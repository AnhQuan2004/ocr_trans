import math
import torch
import torch.nn as nn
from torchvision import models, transforms
from utils import PositionalEncoding, count_parameters, log_config
from config import DEVICE, ALPHABET

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()  # You can also use F.gelu

        self.attn_weights = None  # To store attention weights

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory,
                                                 attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)
        self.attn_weights = attn_weights  # Save the weights here
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights

class TransformerOCR(nn.Module):
    def __init__(self,
                 outtoken,
                 hidden=256,
                 dec_layers=3,
                 nhead=8,
                 dropout=0.1,
                 pretrained=True,
                 use_ctc=True,
                 beam_width=5):
        super(TransformerOCR, self).__init__()
        self.hidden_dim = hidden
        self.dec_layers = dec_layers
        self.use_ctc = use_ctc
        self.beam_width = beam_width
        self.backbone_name = "ViT-B/16"  # Add this attribute for log_config

        # ViT backbone
        vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = models.vit_b_16(weights=vit_weights)
        vit_feature_dim = 768  # Fixed value for ViT-B/16
        self.feature_projection = nn.Linear(vit_feature_dim, hidden)

        # Freeze all ViT layers except the last 4 encoder blocks
        self._freeze_vit_except_last_4_blocks()

        # Data transforms: resize, augment, to tensor, normalize
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Full transform for non-tensor inputs
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            self.normalize
        ])

        # Decoder embedding + positional
        self.decoder_embedding = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        # Transformer decoder
        decoder_layer = CustomTransformerDecoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=dec_layers
        )

        # Output heads
        self.fc_attn = nn.Linear(hidden, outtoken)
        if use_ctc:
            # CTC head directly on encoder features
            self.fc_ctc = nn.Linear(hidden, outtoken)
            self.ctc_loss = nn.CTCLoss(blank=ALPHABET.index('PAD'), zero_infinity=True)

        log_config(self)
        print(f"Initialized TransformerOCR with ViT (hidden={hidden}, layers={dec_layers})")
        print(f"ViT backbone frozen except for the last 4 encoder blocks")
        print(f"Total params: {count_parameters(self):,}")

    def _freeze_vit_except_last_4_blocks(self):
        """Freeze all ViT parameters except for the last 4 encoder blocks"""
        # Freeze all parameters first
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # ViT-B/16 has 12 encoder blocks, we want to unfreeze the last 4
        total_blocks = len(self.vit.encoder.layers)
        blocks_to_unfreeze = 4
        
        # Make sure we don't try to unfreeze more blocks than exist
        blocks_to_unfreeze = min(blocks_to_unfreeze, total_blocks)
        
        # Unfreeze the last 4 encoder blocks
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.vit.encoder.layers[i].parameters():
                param.requires_grad = True
                
        # Also unfreeze the feature projection layer
        for param in self.feature_projection.parameters():
            param.requires_grad = True
            
        print(f"Frozen {total_blocks - blocks_to_unfreeze} blocks, trainable {blocks_to_unfreeze} blocks")

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def _get_features(self, src):
        # src: [B,C,H,W] raw images or tensor
        # If src is already a tensor in the right format, just use it directly
        if isinstance(src, torch.Tensor):
            # Check if tensor has the right dimensions and normalize if needed
            if src.dim() == 4:
                # Resize if needed
                if src.shape[2] != 224 or src.shape[3] != 224:
                    src = torch.nn.functional.interpolate(src, size=(224, 224), mode='bilinear', align_corners=False)
                # Assuming input is already normalized, if not, uncommenting below would normalize it
                # src = self.normalize(src)
            else:
                raise ValueError(f"Input tensor should have 4 dimensions [B,C,H,W], got {src.dim()}")
        else:
            # For non-tensor inputs (PIL images, numpy arrays, etc.)
            src = torch.stack([self.transform(img) for img in src])

        # Forward through ViT
        batch_size = src.shape[0]
        
        # Process through ViT backbone
        x = self.vit._process_input(src)
        batch_class_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        
        # Remove class token and project features
        x = x[:, 1:, :]  # Remove class token
        x = self.feature_projection(x)  # [B, S, hidden]
        x = x.permute(1, 0, 2)  # [S, B, hidden]
        return x

    def forward(self, src, trg):
        memory = self._get_features(src)  # [S,B,E]
        
        # CTC branch
        if self.use_ctc:
            self.ctc_logits = self.fc_ctc(memory)  # [S,B,V]
        else:
            self.ctc_logits = None

        # Attention branch
        trg_mask = self.generate_square_subsequent_mask(trg.size(0), trg.device)
        trg_emb = self.pos_decoder(self.decoder_embedding(trg))

        # Initialize a list to store attention weights
        attention_weights = []

        # Modify the transformer_decoder call to store attention weights
        dec_out = trg_emb
        for layer in self.transformer_decoder.layers:
            dec_out, attn_weight = layer(dec_out, memory, tgt_mask=trg_mask)
            attention_weights.append(attn_weight)  # Save attention weights for each layer

        # Concatenate attention weights if needed (e.g., for visualization across layers)
        attn_logits = self.fc_attn(dec_out)  # [T,B,V]

        return attn_logits, attention_weights

    def compute_loss(self, output, trg, trg_lengths, src_lengths, alpha=0.5):
        # output is now a tuple: (attn_logits, attention_weights)
        # Extract attn_logits from the tuple
        attn_logits = output[0] if isinstance(output, tuple) else output
        
        # attn_logits: [T,B,V], trg: [T,B]
        # self.ctc_logits: [S,B,V]
        # trg_lengths: lengths of each trg seq (without SOS), src_lengths: S lengths (constant)
        # CrossEntropyLoss for attention
        T, B, V = attn_logits.size()
        attn_flat = attn_logits.view(-1, V)
        trg_flat = trg.view(-1)
        ce_loss = nn.functional.cross_entropy(attn_flat, trg_flat, ignore_index=ALPHABET.index('PAD'))
        if self.use_ctc and self.ctc_logits is not None:
            # CTC expects [S, B, V]
            log_probs = self.ctc_logits.log_softmax(2)
            ctc_loss = self.ctc_loss(log_probs, trg, src_lengths, trg_lengths)
            return alpha * ce_loss + (1 - alpha) * ctc_loss
        else:
            return ce_loss

    def predict(self, batch_images, max_len=100):
        self.eval()
        with torch.no_grad():
            memory = self._get_features(batch_images)
            B = memory.size(1)  # Get batch size from memory
            device = memory.device
            results = []
            
            # Beam search placeholder
            # TODO: implement beam search over attn_logits
            for i in range(B):
                idxs = [ALPHABET.index('SOS')]
                for _ in range(max_len):
                    trg = torch.LongTensor(idxs).unsqueeze(1).to(device)
                    mask = self.generate_square_subsequent_mask(len(idxs), device)
                    emb = self.pos_decoder(self.decoder_embedding(trg))
                    
                    # Process through each decoder layer manually to get the output
                    dec_out = emb
                    for layer in self.transformer_decoder.layers:
                        dec_out, _ = layer(dec_out, memory[:, i:i+1, :], tgt_mask=mask)
                    
                    logits = self.fc_attn(dec_out)
                    token = logits[-1, 0].argmax().item()
                    idxs.append(token)
                    if token == ALPHABET.index('EOS'):
                        break
                results.append(idxs)
        return results
