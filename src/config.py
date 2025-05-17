import torch
from torchvision import transforms
import Augmentor
import random


### MODEL ### 
MODEL = 'model3'
HIDDEN = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
N_HEADS = 8
LENGTH = 512  # Increased to handle very long texts like invoices

# Updated alphabet to support both Vietnamese and English
ALPHABET = ['PAD', 'SOS'] + list(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~'
           'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'
           'àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ') + ['EOS']
            
### TRAINING ###
BATCH_SIZE = 64  # Reduced from 32 to 8 to address CUDA out of memory error
DROPOUT = 0.2
N_EPOCHS = 100  # Increased for full dataset
CHECKPOINT_FREQ = 10  # Validate every 5 epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
SCHUDULER_ON = True
PATIENCE = 5
OPTIMIZER_NAME = 'Adam'
LR = 1e-4

### TESTING ###
CASE = True  # Consider case sensitivity
PUNCT = True  # Consider punctuation marks for Vietnamese text

### INPUT IMAGE PARAMETERS ###
WIDTH = 256
HEIGHT = 64
CHANNELS = 3  # Using 3 channels for better feature extraction

### AUGMENTATIONS ###
p = Augmentor.Pipeline()
p.shear(max_shear_left=2, max_shear_right=2, probability=0.7)
p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjusted for better handling of Vietnamese text
    transforms.RandomRotation(degrees=(-5, 5)),  # Reduced rotation to preserve character integrity
    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        shear=5,
        fill=255
    ),
    transforms.ToTensor(),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
