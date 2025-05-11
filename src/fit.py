from time import time
import numpy as np
import torch
from const import TRAIN_LOG, EPOCH_LOG
from config import DEVICE, CHECKPOINT_FREQ, EARLY_STOPPING, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_METRIC
from utils import indicies_to_text, char_error_rate, evaluate, log_metrics
from tqdm import tqdm
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as T

def load_image(image_path, transform):
    """Tải ảnh từ đường dẫn và áp dụng transform"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Thêm batch dimension
    return image_tensor

def plot_attention_weights_on_image(image_path, model, layer_idx=0, max_len=100):
    # Tải ảnh từ path
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = load_image(image_path, transform)
    
    # Truyền ảnh qua mô hình để lấy attention weights
    model.eval()
    with torch.no_grad():
        # The model now returns attention weights as the second element of the tuple
        _, attention_weights = model(image_tensor, torch.zeros((max_len, 1), dtype=torch.long).to(image_tensor.device))

    # Chọn attention weights từ layer và head đầu tiên
    attn_weights_layer = attention_weights[layer_idx]  # Get weights from specified layer
    
    # Handle if attn_weights_layer is a tensor with multiple heads
    if len(attn_weights_layer.shape) > 2:
        # Extract first head (assuming shape [batch, head, tgt_len, src_len])
        attn_weights_np = attn_weights_layer[0, 0].cpu().detach().numpy()
    else:
        # Weights are already for a single head
        attn_weights_np = attn_weights_layer.cpu().detach().numpy()

    # Chuyển ảnh tensor về dạng PIL để vẽ lên
    image = Image.open(image_path)

    # Vẽ heatmap lên ảnh
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    sns.heatmap(attn_weights_np, cmap='Blues', alpha=0.5, cbar=True, square=True, annot=False)
    plt.title(f"Attention Heatmap for Layer {layer_idx + 1}")
    plt.show()
    
    return attention_weights

def log_epoch(epoch, train_loss, epoch_metrics=None, time_taken=None, lr=None, file_path=EPOCH_LOG):
    """
    Ghi log thông tin của mỗi epoch vào file
    
    params
    ---
    epoch : int
        Số thứ tự epoch
    train_loss : float
        Loss của epoch đó
    epoch_metrics : dict
        Metrics khác nếu có (validation)
    time_taken : float
        Thời gian train
    lr : float
        Learning rate
    file_path : Path
        Đường dẫn tới file log
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] Epoch {epoch+1}: Train Loss = {train_loss:.4f}"
    
    if epoch_metrics:
        log_line += f", Val Loss = {epoch_metrics['loss']:.4f}"
        log_line += f", CER = {epoch_metrics['cer']:.4f}"
        log_line += f", WER = {epoch_metrics['wer']:.4f}"
    
    if time_taken:
        log_line += f", Time = {time_taken:.2f}s"
    
    if lr:
        log_line += f", LR = {lr:.2e}"
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(log_line + "\n")
        
    print(f"Wrote log to {file_path}")

def train(model, optimizer, criterion, train_loader):
    """
    params
    ---
    model : nn.Module
    optimizer : nn.Object
    criterion : nn.Object
    train_loader : torch.utils.data.DataLoader

    returns
    ---
    epoch_loss / len(train_loader) : float
        overall loss
    """
    model.train()
    epoch_loss = 0
    # Thêm progress bar cho training loop
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for src, trg in pbar:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg[:-1, :])
        
        # Handle tuple output (attn_logits, attention_weights)
        if isinstance(output, tuple):
            attn_logits, attention_weights = output
            # Use just the attn_logits for loss calculation
            loss = criterion(attn_logits.view(-1, attn_logits.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
        else:
            # For backward compatibility
            loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
            
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Cập nhật progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / len(train_loader)


# GENERAL FUNCTION FROM TRAINING AND VALIDATION
def fit(model, optimizer, scheduler, criterion, train_loader, val_loader, start_epoch=0, end_epoch=24, val_max_batches=5):
    metrics = []
    print("\nBắt đầu training...")
    
    # Khởi tạo file log nếu chưa tồn tại
    with open(EPOCH_LOG, 'w', encoding='utf-8') as f:
        f.write(f"=== TRAINING STARTED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Optimizer: {optimizer.__class__.__name__}, LR: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Epochs: {start_epoch} to {end_epoch}\n\n")
    
    # Early stopping variables
    best_metric = float('inf')
    no_improvement_count = 0
    best_model_state = None
    
    # Progress bar cho epochs
    epoch_pbar = tqdm(range(start_epoch, end_epoch), desc='Epochs', position=0)
    
    for epoch in epoch_pbar:
        start_time = time()
        train_loss = train(model, optimizer, criterion, train_loader)
        end_time = time()
        time_taken = end_time - start_time
        
        # Try to visualize attention weights if file exists
        try:
            image_path = 'data/test/vt_159.jpg'
            if os.path.exists(image_path):
                attention_weights = plot_attention_weights_on_image(image_path, model, layer_idx=0)
        except Exception as e:
            print(f"Error visualizing attention: {e}")
        
        # Ghi log cho mỗi epoch
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Chỉ validate sau mỗi CHECKPOINT_FREQ epochs
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            print("\nValidating...")
            epoch_metrics, _ = evaluate(model, criterion, val_loader, max_batches=val_max_batches)
            epoch_metrics['train_loss'] = train_loss
            epoch_metrics['epoch'] = epoch + 1
            epoch_metrics['time'] = time_taken
            epoch_metrics['lr'] = current_lr
            metrics.append(epoch_metrics)
            
            # Ghi log cho epoch có validation
            log_epoch(epoch, train_loss, epoch_metrics, time_taken, current_lr)
            
            # Cập nhật thông tin trên progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{epoch_metrics["loss"]:.4f}',
                'CER': f'{epoch_metrics["cer"]:.4f}',
                'WER': f'{epoch_metrics["wer"]:.4f}',
                'LR': f'{current_lr:.2e}'
            })
            
            log_metrics(epoch_metrics, TRAIN_LOG)
            if scheduler is not None:
                scheduler.step(epoch_metrics['train_loss'])
            

            # Early stopping check
            if EARLY_STOPPING:
                current_metric = epoch_metrics[EARLY_STOPPING_METRIC]
                
                if (best_metric - current_metric) > EARLY_STOPPING_MIN_DELTA:
                    best_metric = current_metric
                    no_improvement_count = 0
                    # Save best model state
                    best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
                    print(f"New best {EARLY_STOPPING_METRIC}: {best_metric:.4f}")
                    # Ghi log cho best model
                    with open(EPOCH_LOG, 'a', encoding='utf-8') as f:
                        f.write(f"[NEW BEST] Epoch {epoch+1}: {EARLY_STOPPING_METRIC} = {best_metric:.4f}\n")
                else:
                    no_improvement_count += 1
                    print(f"No improvement in {EARLY_STOPPING_METRIC} for {no_improvement_count} validations")
                    # Ghi log cho việc không cải thiện
                    with open(EPOCH_LOG, 'a', encoding='utf-8') as f:
                        f.write(f"[NO IMPROVEMENT] Epoch {epoch+1}: {no_improvement_count}/{EARLY_STOPPING_PATIENCE}\n")
                    
                if no_improvement_count >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Load best model state
                    if best_model_state is not None:
                        model.load_state_dict({key: value.to(DEVICE) for key, value in best_model_state.items()})
                    # Ghi log cho early stopping
                    with open(EPOCH_LOG, 'a', encoding='utf-8') as f:
                        f.write(f"[EARLY STOPPING] Triggered at epoch {epoch+1}, restored best model from epoch {epoch+1-no_improvement_count*CHECKPOINT_FREQ}\n")
                    break
        else:
            # Nếu không validate, chỉ hiển thị train loss
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}'
            })
            # Ghi log cho epoch không có validation
            log_epoch(epoch, train_loss, time_taken=time_taken, lr=current_lr)
            
    # Ghi log kết thúc training
    with open(EPOCH_LOG, 'a', encoding='utf-8') as f:
        f.write(f"\n=== TRAINING COMPLETED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        if best_metric != float('inf'):
            f.write(f"Best {EARLY_STOPPING_METRIC}: {best_metric:.4f}\n")
            
    return metrics

