import math
import torch
import torch.nn as nn
from torchvision import models, transforms
from utils import PositionalEncoding, count_parameters, log_config # Đảm bảo bạn có file utils.py này
from config import DEVICE, ALPHABET # Đảm bảo bạn có file config.py này

class TransformerModel(nn.Module):
    def __init__(self, outtoken, hidden=256, dec_layers=3, nhead=8, dropout=0.1, pretrained=True):
        super(TransformerModel, self).__init__()

        self.dec_layers = dec_layers
        self.hidden_dim = hidden # Lưu lại hidden dim cho tiện dùng

        # Vision Transformer (ViT) backbone
        self.backbone_name = 'vit_b_16'
        if pretrained:
            vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        else:
            vit_weights = None
        
        self.vit = models.vit_b_16(weights=vit_weights)

        # ViT-Base (vit_b_16) có embedding dimension là 768
        vit_feature_dim = self.vit.hidden_dim # Thông thường là 768 cho vit_b_16

        # Feature projection layer để đưa output của ViT về 'hidden' dimension của decoder
        # nếu chúng khác nhau.
        self.feature_projection = nn.Linear(vit_feature_dim, hidden)
        
        # Thêm resize transform để đảm bảo kích thước đầu vào phù hợp với ViT
        self.resize_transform = transforms.Resize((224, 224), antialias=True)
        
        # Transformer Decoder components
        self.decoder_embedding = nn.Embedding(outtoken, hidden) # Đổi tên để rõ ràng hơn
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        # self.pos_encoder sẽ không cần thiết cho ViT features vì ViT đã có pos_embed riêng
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, 
            nhead=nhead, 
            dim_feedforward=hidden * 4, 
            dropout=dropout, 
            activation='relu',
            batch_first=False # TransformerDecoder của PyTorch mặc định là (seq, batch, feature)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=dec_layers
        )

        # Output layer
        self.fc_out = nn.Linear(hidden, outtoken)
        
        # Masks - trg_mask sẽ được tạo on-the-fly trong forward pass
        # self.trg_mask = None # Không cần lưu trữ ở đây nữa

        log_config(self)
        print(f"Mô hình Transformer với ViT backbone đã được khởi tạo.")
        print(f"Số lượng tham số: {count_parameters(self):,}")
        print(f"ViT feature dimension: {vit_feature_dim}, Decoder hidden dimension: {hidden}")
        if vit_feature_dim != hidden:
            print(f"Sử dụng lớp Linear projection từ {vit_feature_dim} -> {hidden} cho features từ ViT.")


    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1) 
        # Sử dụng device từ trg tensor thay vì global DEVICE
        # mask = torch.triu(torch.ones(sz, sz, device=DEVICE), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    # def make_len_mask(self, inp): # Hàm này không được sử dụng, có thể bỏ
    #     return (inp == 0).transpose(0, 1)
    
    def _get_features(self, src):
        '''
        Trích xuất đặc trưng từ ảnh đầu vào bằng ViT.
        
        params
        ---
        src : Tensor [B, C, H, W]
            B - batch, C - channel, H - height, W - width

        returns
        ---
        features : Tensor [S, B, E]
            S - sequence length (num_patches), B - batch size, E - embedding dim (hidden)
        '''
        # Resize ảnh đầu vào thành 224x224 - kích thước yêu cầu của ViT
        src = self.resize_transform(src)
        
        # ViT của torchvision xử lý input và trích xuất patch embeddings
        # Output của self.vit.encoder là (batch_size, num_patches + 1, vit_hidden_dim)
        # trong đó +1 là class token.
        
        # Đây là cách lấy features từ ViT trong torchvision, bỏ qua classification head
        x = self.vit._process_input(src) # Tạo patch embeddings
        n = x.shape[0] # batch_size

        # Thêm class token (nếu ViT model của bạn có)
        # vit_b_16 của torchvision có class token
        if hasattr(self.vit, 'class_token') and self.vit.class_token is not None:
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
        
        x = self.vit.encoder(x) # Output: [B, num_patches + 1 (class_token), vit_feature_dim]

        # Chúng ta sẽ bỏ class token và chỉ sử dụng patch embeddings cho OCR decoder memory
        # Class token thường ở vị trí đầu tiên (index 0)
        x = x[:, 1:, :]  # Output: [B, num_patches, vit_feature_dim]
        
        # Chiếu đặc trưng vào không gian 'hidden' của decoder
        x = self.feature_projection(x)  # Output: [B, num_patches, hidden]
        
        # Đổi chiều để phù hợp với transformer decoder: [S, B, E]
        # S = num_patches
        x = x.permute(1, 0, 2) # Output: [num_patches, B, hidden]
        
        # ViT đã có positional embeddings riêng cho các patch,
        # nên chúng ta không cần thêm self.pos_encoder ở đây nữa.
        return x

    def predict(self, batch_images): # Đổi tên tham số để rõ ràng hơn
        '''
        Dự đoán chuỗi văn bản từ batch ảnh đầu vào.
        Sửa đổi để xử lý batch hiệu quả hơn thay vì từng ảnh một nếu có thể,
        nhưng giữ logic sinh tự hồi quy cho từng ảnh.
        
        params
        ---
        batch_images : Tensor [B, C, H, W]
        
        returns
        ---
        all_results : List of lists, mỗi list con chứa chuỗi tokens dự đoán cho một ảnh.
        '''
        self.eval() # Chuyển sang evaluation mode
        all_results = []
        
        # Lấy đặc trưng từ CNN cho cả batch (nếu _get_features hỗ trợ batch)
        # Hàm _get_features hiện tại đã hỗ trợ batch input src [B, C, H, W]
        # và trả về memory [S, B, E]
        
        # Tuy nhiên, vòng lặp sinh văn bản (decoding) vẫn cần thực hiện cho từng ảnh
        # vì nó là tự hồi quy và độ dài output có thể khác nhau.
        
        # memory_batch: [S, B, E]
        memory_batch = self._get_features(batch_images)
        batch_size = batch_images.size(0)

        for i in range(batch_size):
            # Lấy memory cho từng ảnh trong batch
            # memory: [S, 1, E]
            memory_single_image = memory_batch[:, i:i+1, :] 
            
            # Bắt đầu với token SOS
            out_indexes = [ALPHABET.index('SOS'), ]
            
            # Tham số cho quá trình sinh văn bản
            max_len = 100 # Giảm max_len để test nhanh hơn, bạn có thể điều chỉnh
            consecutive_repeats = 0
            prev_token = -1 # Token không hợp lệ để tránh match ở lần đầu
            
            for _ in range(max_len):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(DEVICE) # [current_len, 1]
                
                # Tạo mask cho decoding trên cùng thiết bị với trg_tensor
                # Chuyển DEVICE vào generate_square_subsequent_mask
                trg_mask = self.generate_square_subsequent_mask(len(out_indexes)).to(trg_tensor.device)
                
                tgt_emb = self.decoder_embedding(trg_tensor) # [current_len, 1, hidden]
                tgt_emb = self.pos_decoder(tgt_emb)
                
                output_decoder = self.transformer_decoder(
                    tgt=tgt_emb,
                    memory=memory_single_image, # memory cho ảnh hiện tại
                    tgt_mask=trg_mask
                ) # Output: [current_len, 1, hidden]
                
                logits = self.fc_out(output_decoder) # [current_len, 1, outtoken]
                
                # Lấy token cuối cùng được dự đoán
                out_token = logits[-1, 0, :].argmax(0).item()
                
                if out_token == ALPHABET.index('EOS'):
                    out_indexes.append(out_token) # Thêm EOS token vào kết quả
                    break
                
                out_indexes.append(out_token)
                
                if out_token == prev_token:
                    consecutive_repeats += 1
                    if consecutive_repeats > 5: # Giảm ngưỡng lặp lại
                        # print("Warning: Detected excessive repetition. Stopping.")
                        break 
                else:
                    consecutive_repeats = 0
                
                prev_token = out_token
            
            all_results.append(out_indexes)
        return all_results

    def forward(self, src, trg):
        '''
        Forward pass cho quá trình huấn luyện.
        
        params
        ---
        src : Tensor [B, C, H, W]
        trg : Tensor [L, B] (target token indices, L là max target length)
              LƯU Ý: TransformerDecoder của PyTorch thường mong đợi target có dạng [T, N, E]
              cho tgt và [T,N] cho target indices (nếu dùng với CrossEntropyLoss).
              Nếu trg là [B, L], cần permute. Mã hiện tại giả định trg là [L,B].

        returns
        ---
        logits : Tensor [L, B, V] (V là outtoken - vocabulary size)
        '''
        # trg nên có token 'SOS' ở đầu và không có 'EOS' ở cuối khi làm input cho decoder
        # trg_for_loss nên là trg không có 'SOS' ở đầu, và có 'EOS' (nếu mô hình học EOS)
        
        # Ví dụ: nếu raw target là [t1, t2, EOS]
        # decoder_input (trg trong hàm này) sẽ là [SOS, t1, t2]
        # target_for_loss sẽ là [t1, t2, EOS]

        # Trích xuất đặc trưng từ ảnh bằng ViT
        # memory: [S_img, B, hidden]
        memory = self._get_features(src) 
        
        # Tạo target mask cho decoder
        # trg: [L_trg, B]
        # Chuyển DEVICE vào generate_square_subsequent_mask
        # Device của mask nên là device của trg
        trg_mask = self.generate_square_subsequent_mask(trg.size(0)).to(trg.device)
        
        # Embedding và positional encoding cho các token target (input cho decoder)
        # trg_emb: [L_trg, B, hidden]
        trg_emb = self.decoder_embedding(trg)
        trg_emb = self.pos_decoder(trg_emb)
        
        # Transformer decoder
        # output: [L_trg, B, hidden]
        output = self.transformer_decoder(
            tgt=trg_emb,
            memory=memory,
            tgt_mask=trg_mask
            # memory_mask=None, # Optional: mask cho memory nếu cần
            # tgt_key_padding_mask=None, # Optional: mask cho padding trong target
            # memory_key_padding_mask=None # Optional: mask cho padding trong memory
        )
        
        # Dự đoán output
        # logits: [L_trg, B, outtoken]
        logits = self.fc_out(output)
        
        return logits