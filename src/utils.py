import os
import random
import string
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import cv2
from PIL import Image
import editdistance
from tqdm import tqdm
from config import ALPHABET, CHANNELS, WIDTH, HEIGHT, DEVICE, BATCH_SIZE
from pathlib import Path

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x) 

    
# convert images and labels into defined data structures
def process_data(image_dir, labels_dir, ignore=[]):
    """
    params
    ---
    image_dir : str or Path
      path to directory with images

    labels_dir : str or Path
      path to tsv file with labels

    returns
    ---
    img2label : dict
      keys are names of images and values are correspondent labels

    chars : list
      all unique chars used in data

    all_labels : list
    """
    chars = []
    img2label = dict()

    try:
        # Convert paths to Path objects if they aren't already
        image_dir = Path(image_dir)
        labels_dir = Path(labels_dir)
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels file not found at: {labels_dir}")
            
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found at: {image_dir}")

        raw = open(labels_dir, 'r', encoding='utf-8').read()
        lines = raw.split('\n')
        for line in lines:
            try:
                if not line.strip():  # Skip empty lines
                    continue
                    
                filename, label = line.split('\t')
                flag = False
                for item in ignore:
                    if item in label:
                        flag = True
                if flag == False:
                    # Use Path to join paths correctly for the OS
                    img_path = image_dir / filename
                    if img_path.exists():  # Only add if image exists
                        img2label[img_path] = label
                        for char in label:
                            if char not in chars:
                                chars.append(char)
                    else:
                        print(f'Warning: Image not found: {img_path}')
            except Exception as e:
                print(f'Bad line: {line}')
                print(f'Error: {str(e)}')
                continue

        if not img2label:
            raise ValueError("No valid image-label pairs found. Please check your data directory and labels file.")

        all_labels = sorted(list(set(list(img2label.values()))))
        chars.sort()
        chars = ['PAD', 'SOS'] + chars + ['EOS']

        return img2label, chars, all_labels
        
    except Exception as e:
        print(f"Error in process_data: {str(e)}")
        raise


# TRANSLATE INDICIES TO TEXT
def indicies_to_text(indexes, idx2char):
    """Convert model output indices to text with better handling of special tokens"""
    text = []
    for idx in indexes:
        char = idx2char[idx]
        # Stop at EOS token
        if char == 'EOS':
            break
        # Skip special tokens
        if char not in ['PAD', 'SOS', 'EOS']:
            text.append(char)
    return "".join(text)


# COMPUTE CHARACTER ERROR RATE
def char_error_rate(p_seq1, p_seq2):
    """
    Calculate Character Error Rate (CER) between two strings
    params
    ---
    p_seq1 : str
        ground truth string
    p_seq2 : str
        predicted string

    returns
    ---
    cer : float
        character error rate
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]

    # Fix the length checking conditions
    if len(c_seq1) == 0 and len(c_seq2) == 0:
        return 0.0
    if len(c_seq1) == 0 or len(c_seq2) == 0:
        return 1.0

    return editdistance.eval(''.join(c_seq1), ''.join(c_seq2)) / max(len(c_seq1), len(c_seq2))


# RESIZE AND NORMALIZE IMAGE
def process_image(img):
    """
    Process image for OCR
    params:
    ---
    img : np.array

    returns
    ---
    img : np.array
    """
    # Convert to float32 for better precision
    img = img.astype('float32')
    
    # Calculate new dimensions while maintaining aspect ratio
    w, h, _ = img.shape
    new_w = HEIGHT
    new_h = int(h * (new_w / w))
    
    # Resize with better interpolation
    img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_LANCZOS4)
    
    # Add padding if needed
    w, h, _ = img.shape
    new_h = WIDTH
    if h < new_h:
        # Use white padding (255)
        add_zeros = np.full((w, new_h - h, 3), 255, dtype=np.float32)
        img = np.concatenate((img, add_zeros), axis=1)
    elif h > new_h:
        # Resize if too large
        img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_LANCZOS4)
    
    return img


# GENERATE IMAGES FROM FOLDER
def generate_data(img_paths):
    """
    params
    ---
    names : list of str
        paths to images

    returns
    ---
    data_images : list of np.array
        images in np.array format
    """
    data_images = []
    for path in tqdm(img_paths):
        try:
            # Convert path to string and normalize it
            path_str = str(path)
            
            # Try loading with PIL first (better Unicode support)
            try:
                with Image.open(path_str) as img:
                    img = np.array(img.convert('RGB'))
            except Exception as e:
                print(f"PIL failed to load {path_str}, trying cv2: {str(e)}")
                # If PIL fails, try cv2 with Unicode path handling
                img = cv2.imdecode(np.fromfile(path_str, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Failed to load image: {path_str}")
            
            img = process_image(img)
            data_images.append(img.astype('uint8'))
        except Exception as e:
            print(f"Error processing image {path_str}: {str(e)}")
            continue
            
    if len(data_images) == 0:
        raise ValueError("No images could be loaded successfully. Please check your image paths and formats.")
        
    return data_images


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, criterion, loader, case=True, punct=True):
    """Evaluate model with detailed metrics"""
    model.eval()
    metrics = {'loss': 0, 'wer': 0, 'cer': 0}
    result = {'true': [], 'predicted': [], 'cer': [], 'correct_chars': 0, 'total_chars': 0}
    
    with torch.no_grad():
        for batch_idx, (src, trg) in enumerate(loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            # Get model predictions
            logits = model(src, trg[:-1, :])
            loss = criterion(logits.view(-1, logits.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
            out_indexes = model.predict(src)
            
            # Convert to text
            true_phrases = [indicies_to_text(trg.T[i][1:], ALPHABET) for i in range(len(trg.T))]
            pred_phrases = [indicies_to_text(out_indexes[i], ALPHABET) for i in range(len(out_indexes))]
            
            if not case:
                true_phrases = [phrase.lower() for phrase in true_phrases]
                pred_phrases = [phrase.lower() for phrase in pred_phrases]
            if not punct:
                true_phrases = [phrase.translate(str.maketrans('', '', string.punctuation)) for phrase in true_phrases]
                pred_phrases = [phrase.translate(str.maketrans('', '', string.punctuation)) for phrase in pred_phrases]
            
            # Calculate metrics
            batch_cer = 0
            batch_wer = 0
            for i in range(len(true_phrases)):
                if not true_phrases[i] or not pred_phrases[i]:
                    continue
                
                # Character level metrics
                current_cer = char_error_rate(true_phrases[i], pred_phrases[i])
                batch_cer += current_cer
                
                # Count correct characters
                for t, p in zip(true_phrases[i], pred_phrases[i]):
                    if t == p:
                        result['correct_chars'] += 1
                result['total_chars'] += len(true_phrases[i])
                
                # Word level metrics
                batch_wer += int(true_phrases[i] != pred_phrases[i])
                
                # Debug output
                if batch_idx < 2 and i < 2:
                    print(f"\nSample {batch_idx}_{i}:")
                    print(f"True:      '{true_phrases[i]}'")
                    print(f"Predicted: '{pred_phrases[i]}'")
                    print(f"CER: {current_cer:.4f}")
            
            # Update metrics
            metrics['loss'] += loss.item()
            metrics['cer'] += batch_cer / len(true_phrases)
            metrics['wer'] += batch_wer / len(true_phrases)
            
            # Store results
            result['true'].extend(true_phrases)
            result['predicted'].extend(pred_phrases)
            result['cer'].extend([char_error_rate(true_phrases[i], pred_phrases[i]) 
                                for i in range(len(true_phrases))])
    
    # Average metrics
    for key in metrics.keys():
        metrics[key] /= len(loader)
    
    # Calculate character accuracy
    char_acc = result['correct_chars'] / max(result['total_chars'], 1) * 100
    
    # Print detailed statistics
    print("\nEvaluation Statistics:")
    print(f"Average CER: {metrics['cer']:.4f}")
    print(f"Average WER: {metrics['wer']:.4f}")
    print(f"Average Loss: {metrics['loss']:.4f}")
    print(f"Character Accuracy: {char_acc:.2f}%")
    print(f"Total Characters: {result['total_chars']}")
    print(f"Correct Characters: {result['correct_chars']}")
    
    return metrics, result


# MAKE PREDICTION
def prediction(model, test_dir, char2idx, idx2char):
    """
    params
    ---
    model : nn.Module
    test_dir : str
        path to directory with images
    char2idx : dict
        map from chars to indicies
    id2char : dict
        map from indicies to chars

    returns
    ---
    preds : dict
        key : name of image in directory
        value : dict with keys ['p_value', 'predicted_label']
    """
    preds = {}
    os.makedirs('/output', exist_ok=True)
    model.eval()

    with torch.no_grad():
        for filename in os.listdir(test_dir):
            img = Image.open(test_dir + filename).convert('RGB')

            img = process_image(np.asarray(img)).astype('uint8')
            img = img / img.max()
            img = np.transpose(img, (2, 0, 1))

            src = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)
            if CHANNELS == 1:
              src = transforms.Grayscale(CHANNELS)(src)
            out_indexes = model.predict(src)
            pred = indicies_to_text(out_indexes[0], idx2char)
            preds[filename] = pred

    return preds


class ToTensor(object):
    def __init__(self, X_type=None, Y_type=None):
        self.X_type = X_type

    def __call__(self, X):
        X = X.transpose((2, 0, 1))
        X = torch.from_numpy(X)
        if self.X_type is not None:
            X = X.type(self.X_type)
        return X


def log_config(model):
    print('transformer layers: {}'.format(model.enc_layers))
    print('transformer heads: {}'.format(model.transformer.nhead))
    print('hidden dim: {}'.format(model.decoder.embedding_dim))
    print('num classes: {}'.format(model.decoder.num_embeddings))
    print('backbone: {}'.format(model.backbone_name))
    print('dropout: {}'.format(model.pos_encoder.dropout.p))
    print(f'{count_parameters(model):,} trainable parameters')


def log_metrics(metrics, path_to_logs=None):
    if path_to_logs != None:
      f = open(path_to_logs, 'a')
    if metrics['epoch'] == 1:
      if path_to_logs != None:
        f.write('Epoch\tTrain_loss\tValid_loss\tCER\tWER\tTime\n')
      print('Epoch   Train_loss   Valid_loss   CER   WER    Time    LR')
      print('-----   -----------  ----------   ---   ---    ----    ---')
    print('{:02d}       {:.2f}         {:.2f}       {:.2f}   {:.2f}   {:.2f}   {:e}'.format(\
        metrics['epoch'], metrics['train_loss'], metrics['loss'], metrics['cer'], \
        metrics['wer'], metrics['time'], metrics['lr']))
    if path_to_logs != None:
      f.write(str(metrics['epoch'])+'\t'+str(metrics['train_loss'])+'\t'+str(metrics['loss'])+'\t'+str(metrics['cer'])+'\t'+str(metrics['wer'])+'\t'+str(metrics['time'])+'\n')
      f.close()
