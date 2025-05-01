import time
import os
import torch

def train_epoch(model, criterion, optimizer, scheduler, train_loader, val_loader, epoch):
    """Train one epoch with improved logging and monitoring"""
    model.train()
    total_loss = 0
    start_time = time.time()
    best_val_loss = float('inf')
    
    # Training metrics
    batch_times = []
    losses = []
    
    for batch_idx, (src, trg) in enumerate(train_loader):
        batch_start = time.time()
        
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(src, trg[:-1, :])
        loss = criterion(
            logits.view(-1, logits.shape[-1]),
            torch.reshape(trg[1:, :], (-1,))
        )
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        losses.append(loss.item())
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Logging
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch: {epoch:3d} | Batch: {batch_idx:5d}/{len(train_loader):5d} | '
                  f'Loss: {loss.item():10.4f} | Elapsed: {elapsed:8.2f}s | '
                  f'Batch time: {batch_time:6.3f}s')
            
            # Sample predictions
            if batch_idx % 500 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model.predict(src[:2])
                    for i in range(min(2, len(pred))):
                        true_text = indicies_to_text(trg.T[i][1:], ALPHABET)
                        pred_text = indicies_to_text(pred[i], ALPHABET)
                        print(f"\nSample {i}:")
                        print(f"True:      '{true_text}'")
                        print(f"Predicted: '{pred_text}'")
                model.train()
    
    # Compute epoch statistics
    avg_loss = total_loss / len(train_loader)
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    # Validation
    val_metrics, _ = evaluate(model, criterion, val_loader)
    val_loss = val_metrics['loss']
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print epoch summary
    print(f"\nEpoch {epoch} Summary:")
    print(f"Training Loss: {avg_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation CER: {val_metrics['cer']:.4f}")
    print(f"Validation WER: {val_metrics['wer']:.4f}")
    print(f"Average Batch Time: {avg_batch_time:.3f}s")
    print(f"Learning Rate: {current_lr:.6f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': avg_loss,
        }, f'checkpoints/best_model_epoch_{epoch}.pt')
    
    return avg_loss, val_metrics

def train(model, criterion, train_loader, val_loader, num_epochs=10):
    """Training loop with improved monitoring and early stopping"""
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_cer': [],
        'val_wer': [],
        'learning_rates': []
    }
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Starting training...")
    start_time = time.time()
    
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train one epoch
            train_loss, val_metrics = train_epoch(
                model, criterion, optimizer, scheduler,
                train_loader, val_loader, epoch
            )
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_cer'].append(val_metrics['cer'])
            history['val_wer'].append(val_metrics['wer'])
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Epoch timing
            epoch_time = time.time() - epoch_start
            print(f"Epoch completed in {epoch_time:.2f} seconds")
            
            # Save checkpoint
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                }, f'checkpoints/checkpoint_epoch_{epoch}.pt')
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Final timing
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    
    # Save final model and training history
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'final_train_loss': train_loss,
        'final_val_metrics': val_metrics,
    }, 'checkpoints/final_model.pt')
    
    return history 