from time import time
import numpy as np
import torch
from const import TRAIN_LOG
from config import DEVICE
from utils import indicies_to_text, char_error_rate, evaluate, log_metrics
from tqdm import tqdm

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

        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Cập nhật progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / len(train_loader)


# GENERAL FUNCTION FROM TRAINING AND VALIDATION
def fit(model, optimizer, scheduler, criterion, train_loader, val_loader, start_epoch=0, end_epoch=24):
    metrics = []
    print("\nBắt đầu training...")
    
    # Progress bar cho epochs
    epoch_pbar = tqdm(range(start_epoch, end_epoch), desc='Epochs', position=0)
    
    for epoch in epoch_pbar:
        start_time = time()
        train_loss = train(model, optimizer, criterion, train_loader)
        end_time = time()
        
        # Thêm progress bar cho validation
        print("\nValidating...")
        epoch_metrics, _ = evaluate(model, criterion, val_loader)
        epoch_metrics['train_loss'] = train_loss
        epoch_metrics['epoch'] = epoch
        epoch_metrics['time'] = end_time - start_time
        epoch_metrics['lr'] = optimizer.param_groups[0]["lr"]
        metrics.append(epoch_metrics)
        
        # Cập nhật thông tin trên progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{epoch_metrics["loss"]:.4f}',
            'CER': f'{epoch_metrics["cer"]:.4f}',
            'WER': f'{epoch_metrics["wer"]:.4f}',
            'LR': f'{epoch_metrics["lr"]:.2e}'
        })
        
        log_metrics(epoch_metrics, TRAIN_LOG)
        if scheduler != None:
            scheduler.step(epoch_metrics['train_loss'])
            
    return metrics
