import torch
from torchvision import transforms
import Augmentor
import random


### MODEL ### 
MODEL = 'model1'
HIDDEN = 512
ENC_LAYERS = 6
DEC_LAYERS = 6
N_HEADS = 8
LENGTH = 42

# Updated alphabet to support both Vietnamese and English
ALPHABET = ['PAD', 'SOS'] + list(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~'
           'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ'
           'àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ') + ['EOS']
            
### TRAINING ###
BATCH_SIZE = 16
DROPOUT = 0.3
N_EPOCHS = 100
CHECKPOINT_FREQ = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
SCHUDULER_ON = True
PATIENCE = 3
OPTIMIZER_NAME = 'AdamW'
LR = 2e-4

### TESTING ###
CASE = True
PUNCT = True

### INPUT IMAGE PARAMETERS ###
WIDTH = 384
HEIGHT = 96
CHANNELS = 3

### AUGMENTATIONS ###
p = Augmentor.Pipeline()
p.random_distortion(probability=0.8, grid_width=4, grid_height=4, magnitude=8)

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(degrees=(-3, 3)),
    transforms.RandomAffine(
        degrees=3,
        translate=(0.02, 0.02),
        scale=(0.98, 1.02),
        shear=3,
        fill=255
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
