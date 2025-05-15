import random
import torch
import numpy as np
from collections import Counter
import Augmentor
from torchvision import transforms
import augmentations
from utils import *
from config import LENGTH, CHANNELS

### AUGMENTATIONS ###
vignet = augmentations.Vignetting()
cutout = augmentations.Cutout(min_size_ratio=[1, 4], max_size_ratio=[2, 5])
un = augmentations.UniformNoise()
tt = ToTensor()
ld = augmentations.LensDistortion()
######################

# text to array of indicies
def text_to_labels(s, char2idx):
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]

# store list of images' names (in directory) and does some operations with images
class TextLoader(torch.utils.data.Dataset):
    def __init__(self, images_name, labels, transforms, char2idx, idx2char):
        """
        params
        ---
        images_name : list
            list of names of images (paths to images)
        labels : list
            list of labels to correspondent images from images_name list
        char2idx : dict
        idx2char : dict
        """
        self.images_name = images_name
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.transform = transforms

    def _transform(self, X):
        j = np.random.randint(0, 3, 1)[0]
        if j == 0:
            return self.transform(X)
        if j == 1:
            return tt(ld(vignet(X)))
        if j == 2:
            return tt(ld(un(X)))
            

    # shows some stats about dataset
    def get_info(self):
        N = len(self.labels)
        max_len = -1
        for label in self.labels:
            if len(label) > max_len:
                max_len = len(label)
        counter = Counter(''.join(self.labels))
        counter = dict(sorted(counter.items(), key=lambda item: item[1]))
        print(
            'Size of dataset: {}\nMax length of expression: {}\nThe most common char: {}\n The least common char: {}'.format( \
                N, max_len, list(counter.items())[-1], list(counter.items())[0]))

    def __getitem__(self, index):
        img = self.images_name[index] # img is a PIL.Image
        img_tensor = self.transform(img) # self.transform is TRAIN_TRANSFORMS or TEST_TRANSFORMS

        label = text_to_labels(self.labels[index], self.char2idx)
        # img_tensor is already a FloatTensor due to transforms.ToTensor()
        return (img_tensor, torch.LongTensor(label))

    def __len__(self):
        return len(self.labels)


# MAKE TEXT TO BE THE SAME LENGTH
class TextCollate():
    def __call__(self, batch):
        x_padded = []
        
        # Tìm độ dài lớn nhất trong batch hiện tại một cách động
        max_y_len = max([batch[i][1].size(0) for i in range(len(batch))])
        
        # Không còn giới hạn cứng bởi LENGTH, chỉ giới hạn một giá trị hợp lý để tránh OOM
        # LENGTH trong config.py trở thành một gợi ý giới hạn tối đa thay vì giá trị cứng
        safe_max_len = min(max_y_len, 1024)  # Giới hạn an toàn để tránh OOM
        
        # Khởi tạo tensor kết quả với kích thước đúng
        y_padded = torch.LongTensor(safe_max_len, len(batch))
        # Fill with PAD index, ALPHABET must be imported or accessible
        # Assuming ALPHABET is imported from config and available in this scope
        from config import ALPHABET # Ensure ALPHABET is in scope
        pad_token_idx = ALPHABET.index('PAD')
        y_padded.fill_(pad_token_idx)

        actual_target_lengths = []

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1] # This is the output of text_to_labels: [SOS, char1, ..., charN, EOS]
            
            # Actual length for CTC is number of chars (N), so len(y) - 2
            # However, nn.CTCLoss target_lengths should be for the targets passed to it.
            # If targets for CTC are y[1:-1], then length is y.size(0) - 2.
            # If targets for CTC are y (including SOS/EOS), then length is y.size(0).
            # Let's assume for now CTCLoss in model.compute_loss will handle SOS/EOS if needed,
            # and target_lengths should be the length of the sequence given to CTCLoss.
            # For CE loss, the full y_padded (with SOS/EOS) is used.
            # For CTC, typically the original characters without SOS/EOS are used.
            # Let's return the length of the original text (number of characters).
            original_text_len = y.size(0) - 2 
            actual_target_lengths.append(original_text_len if original_text_len > 0 else 1) # CTC needs positive lengths

            current_padded_len = min(y.size(0), safe_max_len)
            y_padded[:current_padded_len, i] = y[:current_padded_len]

        x_padded = torch.cat(x_padded)
        y_lengths_tensor = torch.IntTensor(actual_target_lengths)
        return x_padded, y_padded, y_lengths_tensor
