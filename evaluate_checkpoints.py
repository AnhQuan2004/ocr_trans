import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.resolve())+'/src')

from const import PATH_TEST_DIR, PATH_TEST_LABELS
from config import MODEL, BATCH_SIZE, N_HEADS, \
                    ENC_LAYERS, DEC_LAYERS, HIDDEN, \
                    DROPOUT, DEVICE, ALPHABET, TEST_TRANSFORMS
from utils import process_data, generate_data
from dataset import TextCollate, TextLoader

def evaluate_checkpoint(model, test_loader):
    model.eval()
    total_cer = 0
    total_wer = 0
    n_batches = 0
    
    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            out_indexes = model.predict(src)
            
            # Calculate CER and WER here
            # ... (using your existing evaluation logic)
            
            n_batches += 1
            
    return total_cer / n_batches, total_wer / n_batches

def main():
    # Load test data
    img2label, _, all_words = process_data(PATH_TEST_DIR, PATH_TEST_LABELS) 
    img_names, labels = list(img2label.keys()), list(img2label.values())
    X_test = generate_data(img_names)
    y_test = labels

    test_dataset = TextLoader(X_test, y_test, TEST_TRANSFORMS, char2idx, idx2char)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                            batch_size=BATCH_SIZE, pin_memory=True,
                                            drop_last=True, collate_fn=TextCollate())

    # Create character mappings
    char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
    idx2char = {idx: char for idx, char in enumerate(ALPHABET)}

    # Load model architecture
    if MODEL == 'model1':
        from models import model1
        model = model1.TransformerModel(len(ALPHABET), hidden=HIDDEN, 
                                      enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                                      nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
    elif MODEL == 'model2':
        from models import model2
        model = model2.TransformerModel(len(ALPHABET), hidden=HIDDEN, 
                                      enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                                      nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)

    # Find all checkpoints
    checkpoints = sorted(Path('.').glob('checkpoint_*.pt'))
    results = []

    print("\nEvaluating checkpoints...")
    print("-" * 50)
    print("Checkpoint\tCER\tWER")
    print("-" * 50)

    # Evaluate each checkpoint
    for ckpt_path in checkpoints:
        try:
            model.load_state_dict(torch.load(ckpt_path))
            cer, wer = evaluate_checkpoint(model, test_loader)
            results.append((ckpt_path, cer, wer))
            print(f"{ckpt_path.name}\t{cer:.4f}\t{wer:.4f}")
        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {str(e)}")

    # Find best checkpoint
    best_ckpt = min(results, key=lambda x: x[1])  # Using CER as metric
    print("\nBest checkpoint:")
    print(f"File: {best_ckpt[0]}")
    print(f"CER: {best_ckpt[1]:.4f}")
    print(f"WER: {best_ckpt[2]:.4f}")

if __name__ == "__main__":
    main() 