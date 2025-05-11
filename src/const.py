from pathlib import Path
import os

# Get the project root directory using relative path
DIR = Path(__file__).parent.parent.resolve()

# Convert to absolute path without Unicode issues
try:
    DIR = Path(os.path.abspath(DIR))
except:
    print(f"Warning: Could not convert path to absolute: {DIR}")

# Data paths relative to project root
PATH_TEST_DIR = DIR / 'data/test'
PATH_TEST_LABELS = DIR / 'data/test.tsv'
PATH_TRAIN_DIR = DIR / 'data/train'
PATH_TRAIN_LABELS = DIR / 'data/train.tsv'
PREDICT_PATH = DIR / 'data/test'
CHECKPOINTS_PATH = DIR
FROM_CHECKPOINT_PATH = None # if not None then training start with this checkpoint
WEIGHTS_PATH = DIR / 'ocr_transformer_rn50_4h2l_64x256.pt'
PATH_TEST_RESULTS = DIR / 'test_rn50_4h2l_result.tsv'
TRAIN_LOG = DIR / 'train_log.tsv'
EPOCH_LOG = DIR / 'epoch_log.tsv'

# Print paths for debugging
print(f"Project directory: {DIR}")
print(f"Training data directory: {PATH_TRAIN_DIR}")
print(f"Training labels file: {PATH_TRAIN_LABELS}")
