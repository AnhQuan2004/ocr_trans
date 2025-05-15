import torch
from torchvision import transforms
import Augmentor
import random


### MODEL ### 
MODEL = 'model4'
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
BATCH_SIZE = 8  # Reduced from 32 to 8 to address CUDA out of memory error (set to 8 based on comment)
DROPOUT = 0.5
N_EPOCHS = 100  # Increased for full dataset
CHECKPOINT_FREQ = 20  # Validate every 5 epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
SCHUDULER_ON = True
PATIENCE = 5
OPTIMIZER_NAME = 'Adam'
LR = 2e-4  # Reduced learning rate for fine-tuning with frozen layers

# Early stopping configuration
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3  # Stop training if no improvement for 3 consecutive validations
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to qualify as improvement
EARLY_STOPPING_METRIC = 'loss'  # Monitor validation loss

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

VIT_IMAGE_SIZE = 224
VIT_MEAN = [0.485, 0.456, 0.406]
VIT_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjusted for better handling of Vietnamese text
    transforms.RandomRotation(degrees=(-5, 5)),  # Reduced rotation to preserve character integrity
    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        shear=5,
        fill=255 # Consider using mean color for fill if appropriate
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD)
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD)
])
