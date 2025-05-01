from pathlib import Path

# For Kaggle environment
KAGGLE_DIR = Path('/kaggle/working')
DATA_DIR = KAGGLE_DIR / 'data'

# Data paths
PATH_TEST_DIR = DATA_DIR / 'test'
PATH_TEST_LABELS = DATA_DIR / 'test.tsv'
PATH_TRAIN_DIR = DATA_DIR / 'train'
PATH_TRAIN_LABELS = DATA_DIR / 'train.tsv'
PREDICT_PATH = DATA_DIR / 'test'

# Model and checkpoint paths
CHECKPOINTS_PATH = KAGGLE_DIR / 'checkpoints'
FROM_CHECKPOINT_PATH = None  # if not None then training start with this checkpoint
WEIGHTS_PATH = KAGGLE_DIR / 'ocr_transformer_rn50_4h2l_64x256.pt'
PATH_TEST_RESULTS = KAGGLE_DIR / 'test_rn50_4h2l_result.tsv'
TRAIN_LOG = KAGGLE_DIR / 'train_log.tsv'

# Create necessary directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
PATH_TEST_DIR.mkdir(parents=True, exist_ok=True)
PATH_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

# Print paths for debugging
print(f"Working directory: {KAGGLE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Training data directory: {PATH_TRAIN_DIR}")
print(f"Training labels file: {PATH_TRAIN_LABELS}")
