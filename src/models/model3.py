import math
import torch
import torch.nn as nn
from torchvision import models
from utils import PositionalEncoding, count_parameters, log_config
from config import DEVICE, ALPHABET

class TransformerModel(nn.Module):
    def __init__(self, outtoken, hidden=256, dec_layers=3, nhead=8, dropout=0.1, pretrained=True):
        super(TransformerModel, self).__init__()

        self.dec_layers = dec_layers
        
        # ResNet50 backbone
        self.backbone_name = 'resnet50'
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Conv2d(2048, hidden, 1)
        
        # Feature projection sẽ được khởi tạo trong lần gọi _get_features đầu tiên
        self._feature_projection_initialized = False
        # Placeholder cho feature_projection
        self.feature_projection = None
        
        # Transformer Decoder components
        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.pos_encoder = self.pos_decoder
        
        # Chỉ sử dụng TransformerDecoder thay vì Transformer đầy đủ
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, 
            nhead=nhead, 
            dim_feedforward=hidden * 4, 
            dropout=dropout, 
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=dec_layers
        )

        # Output layer
        self.fc_out = nn.Linear(hidden, outtoken)
        
        # Masks
        self.trg_mask = None
        
        log_config(self)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=DEVICE), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)
    
    def _get_features(self, src):
        '''
        Trích xuất đặc trưng từ ảnh đầu vào bằng ResNet50
        
        params
        ---
        src : Tensor [B, C, H, W]
            B - batch, C - channel, H - height, W - width

        returns
        ---
        features : Tensor [S, B, E]
            S - sequence length (width), B - batch size, E - embedding dim
        '''
        x = self.backbone.conv1(src)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # [B, 2048, H/32, W/32]
            
        x = self.backbone.fc(x)  # [B, hidden, H/32, W/32]
        
        # Chuyển đổi feature map thành sequence
        # [B, hidden, H/32, W/32] -> [B, W/32, hidden, H/32] -> [B, W/32, hidden*H/32]
        x = x.permute(0, 3, 1, 2)  
        batch_size, seq_len, channels, height = x.size()
        
        # Điều chỉnh: Nếu kích thước đặc trưng là 512, điều chỉnh self.feature_projection để phù hợp
        input_dim = channels * height
        
        # Nếu chưa khởi tạo lớp projection đúng kích thước, tạo mới
        # và đảm bảo nó được đưa đến thiết bị đúng
        if not self._feature_projection_initialized:
            device = src.device  # Lấy thiết bị từ tensor đầu vào
            self.feature_projection = torch.nn.Linear(input_dim, channels).to(device)
            self._feature_projection_initialized = True
            print(f"Đã điều chỉnh feature projection: {input_dim} -> {channels} (trên {device})")
        
        x = x.contiguous().view(batch_size, seq_len, input_dim)
        
        # Chiếu đặc trưng vào không gian của decoder
        x = self.feature_projection(x)  # [B, S, E]
        
        # Đổi chiều để phù hợp với transformer decoder: [S, B, E]
        x = x.permute(1, 0, 2)
        
        return x

    def predict(self, batch):
        '''
        Dự đoán chuỗi văn bản từ ảnh đầu vào
        
        params
        ---
        batch : Tensor [B, C, H, W]
            B - batch, C - channel, H - height, W - width
        
        returns
        ---
        result : List [B, -1]
            chuỗi tokens dự đoán
        '''
        result = []
        for item in batch:
            # Lấy đặc trưng từ CNN
            memory = self._get_features(item.unsqueeze(0))
            
            # Bắt đầu với token SOS
            out_indexes = [ALPHABET.index('SOS'), ]
            
            # Tham số cho quá trình sinh văn bản
            max_len = 1024
            consecutive_repeats = 0
            prev_token = -1
            
            # Sinh văn bản tự hồi quy
            for i in range(max_len):
                # Chuẩn bị input cho decoder
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(DEVICE)
                
                # Tạo mask cho decoding trên cùng thiết bị với trg_tensor
                trg_mask = self.generate_square_subsequent_mask(len(out_indexes)).to(trg_tensor.device)
                
                # Embedding và positional encoding
                tgt_emb = self.decoder(trg_tensor)
                tgt_emb = self.pos_decoder(tgt_emb)
                
                # Transformer decoding
                output = self.transformer_decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=trg_mask
                )
                
                # Dự đoán token tiếp theo
                logits = self.fc_out(output)
                out_token = logits[-1].argmax(1).item()
                out_indexes.append(out_token)
                
                # Dừng nếu gặp EOS token
                if out_token == ALPHABET.index('EOS'):
                    break
                    
                # Dừng nếu model bắt đầu lặp lại nhiều lần
                if out_token == prev_token:
                    consecutive_repeats += 1
                    if consecutive_repeats > 10:
                        out_indexes.append(ALPHABET.index('EOS'))
                        break
                else:
                    consecutive_repeats = 0
                
                prev_token = out_token
                
                # Kiểm tra độ dài tối đa
                if len(out_indexes) >= max_len - 1:
                    out_indexes.append(ALPHABET.index('EOS'))
                    break
                    
            result.append(out_indexes)
        return result

    def forward(self, src, trg):
        '''
        Forward pass cho quá trình huấn luyện
        
        params
        ---
        src : Tensor [B, C, H, W]
            B - batch, C - channel, H - height, W - width
        trg : Tensor [L, B]
            L - max length of label, B - batch size
            
        returns
        ---
        logits : Tensor [L, B, V]
            dự đoán cho mỗi token, V - vocabulary size
        '''
        # Trích xuất đặc trưng từ ảnh bằng CNN
        memory = self._get_features(src)
        
        # Tạo mask cho decoder trên cùng thiết bị với dữ liệu
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
        
        # Embedding và positional encoding cho các token target
        trg_emb = self.decoder(trg)
        trg_emb = self.pos_decoder(trg_emb)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=trg_emb,
            memory=memory,
            tgt_mask=self.trg_mask
        )
        
        # Dự đoán output
        logits = self.fc_out(output)
        
        return logits 