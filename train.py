import sys
import torch
import random
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve())+'/src')

from const import PATH_TEST_DIR, PATH_TEST_LABELS, FROM_CHECKPOINT_PATH, \
                  PATH_TRAIN_DIR, PATH_TRAIN_LABELS, CHECKPOINTS_PATH
from config import MODEL, BATCH_SIZE, N_HEADS, \
                    ENC_LAYERS, DEC_LAYERS, LR, \
                    DEVICE, RANDOM_SEED, HIDDEN, \
                    DROPOUT, CHECKPOINT_FREQ, N_EPOCHS, \
                    ALPHABET, TRAIN_TRANSFORMS, TEST_TRANSFORMS, \
                    OPTIMIZER_NAME, SCHUDULER_ON, PATIENCE
from utils import generate_data, process_data 
from dataset import TextCollate, TextLoader
from fit import fit

# Verify data paths
def verify_paths():
    if not PATH_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found at: {PATH_TRAIN_DIR}")
    if not PATH_TRAIN_LABELS.exists():
        raise FileNotFoundError(f"Training labels file not found at: {PATH_TRAIN_LABELS}")
    if not PATH_TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found at: {PATH_TEST_DIR}")
    if not PATH_TEST_LABELS.exists():
        raise FileNotFoundError(f"Test labels file not found at: {PATH_TEST_LABELS}")
    print("All data paths verified successfully!")

# Set random seeds
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Create character mappings
char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
idx2char = {idx: char for idx, char in enumerate(ALPHABET)}

# Print vocabulary size
print(f"Vocabulary size: {len(ALPHABET)} characters")

# Ensure directories exist
PATH_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
PATH_TEST_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from {PATH_TRAIN_DIR} ...")
print(f"Using labels from {PATH_TRAIN_LABELS}")

try:
    # Verify paths before proceeding
    verify_paths()
    
    img2label, _, all_words = process_data(PATH_TRAIN_DIR, PATH_TRAIN_LABELS) 
    if not img2label:
        raise ValueError("No valid image-label pairs found in training data")
        
    img_names, labels = list(img2label.keys()), list(img2label.values())
    
    # Sử dụng toàn bộ dữ liệu huấn luyện
    print(f"Using all {len(img_names)} available samples")
        
    print(f"Loading {len(img_names)} training images")
    X_train = generate_data(img_names)
    y_train = labels

    train_dataset = TextLoader(X_train, y_train, TRAIN_TRANSFORMS, char2idx, idx2char)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                               batch_size=BATCH_SIZE, pin_memory=True,
                                               drop_last=True, collate_fn=TextCollate())

    print(f"Loading test dataset from {PATH_TEST_DIR} ...")
    img2label, _, all_words = process_data(PATH_TEST_DIR, PATH_TEST_LABELS) 
    img_names, labels = list(img2label.keys()), list(img2label.values())
    
    # Sử dụng toàn bộ dữ liệu kiểm thử
    print(f"Using all {len(img_names)} available test samples")
        
    print(f"Loading {len(img_names)} test images")
    X_test = generate_data(img_names)
    y_test = labels

    test_dataset = TextLoader(X_test, y_test, TEST_TRANSFORMS, char2idx ,idx2char)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True,
                                               batch_size=BATCH_SIZE, pin_memory=True,
                                               drop_last=True, collate_fn=TextCollate())

    if MODEL == 'model1':
        from models import model1
        model = model1.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                                nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
    elif MODEL == 'model2':
        from models import model2
        model = model2.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                                nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
    elif MODEL == 'model3':
        from models import model3
        model = model3.TransformerModel(len(ALPHABET), hidden=HIDDEN, dec_layers=DEC_LAYERS,   
                                nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {MODEL}")

    if FROM_CHECKPOINT_PATH is not None:
        if not FROM_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {FROM_CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(FROM_CHECKPOINT_PATH))
        print(f'Loading from checkpoint {FROM_CHECKPOINT_PATH}')

    criterion = torch.nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
    optimizer = torch.optim.__getattribute__(OPTIMIZER_NAME)(model.parameters(), lr=LR)

    if SCHUDULER_ON:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE)
    else:
        scheduler = None

    print(f'Checkpoints will be saved in {CHECKPOINTS_PATH} every {CHECKPOINT_FREQ} epochs')
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Train for all epochs at once
    metrics = fit(model, optimizer, scheduler, criterion, train_loader, test_loader, 0, N_EPOCHS, val_max_batches=2)
    
    # Save final model
    save_path = CHECKPOINTS_PATH / 'final_model.pt'
    torch.save(model.state_dict(), save_path)
    print(f'Saved final model to {save_path}')
        
except Exception as e:
    print(f"Error during training: {str(e)}")
    raise
