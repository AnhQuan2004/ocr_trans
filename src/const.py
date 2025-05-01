from pathlib import Path

DIR = Path.cwd() # work directory
PATH_TEST_DIR = Path(DIR, 'data/test')
PATH_TEST_LABELS = Path(DIR, 'data/test.tsv')
PATH_TRAIN_DIR = Path(DIR, 'data/train')
PATH_TRAIN_LABELS = Path(DIR, 'data/train.tsv')
PREDICT_PATH = Path(DIR, 'data/test')
CHECKPOINTS_PATH = Path(DIR)
FROM_CHECKPOINT_PATH = None # if not None then training start with this checkpoint
WEIGHTS_PATH = Path(DIR, 'ocr_transformer_rn50_4h2l_64x256.pt')
PATH_TEST_RESULTS = Path(DIR, 'test_rn50_4h2l_result.tsv')
TRAIN_LOG = Path(DIR, 'train_log.tsv')
