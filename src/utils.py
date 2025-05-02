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
from config import ALPHABET, CHANNELS, WIDTH, HEIGHT, DEVICE, BATCH_SIZE, LENGTH
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
        very_long_texts = 0
        max_text_len = 0
        
        for line in lines:
            try:
                if not line.strip():  # Skip empty lines
                    continue
                    
                filename, label = line.split('\t')
                
                # Theo dõi độ dài tối đa của văn bản để thông báo
                text_len = len(label)
                max_text_len = max(max_text_len, text_len)
                if text_len > 512:
                    very_long_texts += 1
                    
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

        print(f'Dataset statistics:')
        print(f'- Total samples: {len(img2label)}')
        print(f'- Maximum text length: {max_text_len} characters')
        print(f'- Very long texts (>512 chars): {very_long_texts}')

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
    text = "".join([idx2char[i] for i in indexes])
    text = text.replace('EOS', '').replace('PAD', '').replace('SOS', '')
    return text


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
    params:
    ---
    img : np.array

    returns
    ---
    img : np.array
    """
    w, h, _ = img.shape
    new_w = HEIGHT
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = WIDTH
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

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


def evaluate(model, criterion, loader, case=True, punct=True, max_batches=None):
    """
    Evaluate model performance
    params
    ---
    model : nn.Module
    criterion : nn.Object
    loader : torch.utils.data.DataLoader
    case : bool
        whether to consider case sensitivity
    punct : bool
        whether to consider punctuation
    max_batches : int or None
        if specified, limits evaluation to this many batches

    returns
    ---
    metrics : dict
        evaluation metrics
    result : dict
        detailed results for analysis
    """
    model.eval()
    metrics = {'loss': 0, 'wer': 0, 'cer': 0}
    result = {'true': [], 'predicted': [], 'cer': []}
    
    with torch.no_grad():
        # Thêm progress bar cho validation
        val_pbar = tqdm(loader, desc='Validating', leave=False)
        for batch_idx, (src, trg) in enumerate(val_pbar):
            # Giới hạn số lượng batch để đánh giá
            if max_batches is not None and batch_idx >= max_batches:
                print(f"\nReached max_batches limit ({max_batches}). Stopping validation.")
                break
                
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            logits = model(src, trg[:-1, :])
            loss = criterion(logits.view(-1, logits.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
            out_indexes = model.predict(src)
            
            true_phrases = [indicies_to_text(trg.T[i][1:], ALPHABET) for i in range(BATCH_SIZE)]
            pred_phrases = [indicies_to_text(out_indexes[i], ALPHABET) for i in range(BATCH_SIZE)]
            
            if not case:
                true_phrases = [phrase.lower() for phrase in true_phrases]
                pred_phrases = [phrase.lower() for phrase in pred_phrases]
            if not punct:
                true_phrases = [phrase.translate(str.maketrans('', '', string.punctuation)) for phrase in true_phrases]
                pred_phrases = [phrase.translate(str.maketrans('', '', string.punctuation)) for phrase in pred_phrases]
            
            # Calculate metrics for this batch
            batch_cer = 0
            batch_wer = 0
            for i in range(BATCH_SIZE):
                # Skip empty predictions/targets
                if not true_phrases[i] or not pred_phrases[i]:
                    continue
                    
                current_cer = char_error_rate(true_phrases[i], pred_phrases[i])
                batch_cer += current_cer
                batch_wer += int(true_phrases[i] != pred_phrases[i])
                
                # Debug output for first few batches
                if batch_idx < 2 and i < 2:  # Show first 2 samples of first 2 batches
                    print(f"\nSample {batch_idx}_{i}:")
                    print(f"True:      '{true_phrases[i]}'")
                    print(f"Predicted: '{pred_phrases[i]}'")
                    print(f"CER: {current_cer:.4f}")
            
            metrics['loss'] += loss.item()
            metrics['cer'] += batch_cer / BATCH_SIZE
            metrics['wer'] += batch_wer / BATCH_SIZE
            
            # Store detailed results
            result['true'].extend(true_phrases)
            result['predicted'].extend(pred_phrases)
            result['cer'].extend([char_error_rate(true_phrases[i], pred_phrases[i]) for i in range(BATCH_SIZE)])

            # Cập nhật progress bar với metrics hiện tại
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CER': f'{batch_cer/BATCH_SIZE:.4f}',
                'WER': f'{batch_wer/BATCH_SIZE:.4f}'
            })

    # Average metrics over all batches that were actually processed
    num_batches = min(len(loader), max_batches or float('inf'))
    for key in metrics.keys():
        metrics[key] /= num_batches
        
    # Print some statistics
    print("\nEvaluation Statistics:")
    print(f"Evaluated on {num_batches} batches ({len(result['true'])} samples)")
    print(f"Average CER: {metrics['cer']:.4f}")
    print(f"Average WER: {metrics['wer']:.4f}")
    print(f"Average Loss: {metrics['loss']:.4f}")
    
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
    # Kiểm tra các thuộc tính trước khi truy cập
    if hasattr(model, 'enc_layers'):
        print('encoder layers: {}'.format(model.enc_layers))
    if hasattr(model, 'dec_layers'):
        print('decoder layers: {}'.format(model.dec_layers))
    
    # Kiểm tra transformer hoặc transformer_decoder
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'nhead'):
        print('transformer heads: {}'.format(model.transformer.nhead))
    elif hasattr(model, 'transformer_decoder') and hasattr(model.transformer_decoder.layers[0], 'nhead'):
        print('transformer heads: {}'.format(model.transformer_decoder.layers[0].nhead))
    
    # Kiểm tra các thuộc tính decoder
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'embedding_dim'):
        print('hidden dim: {}'.format(model.decoder.embedding_dim))
        print('num classes: {}'.format(model.decoder.num_embeddings))
    elif hasattr(model, 'decoder_embedding') and hasattr(model.decoder_embedding, 'embedding_dim'):
        print('hidden dim: {}'.format(model.decoder_embedding.embedding_dim))
        print('num classes: {}'.format(model.decoder_embedding.num_embeddings))
    
    # Backbone luôn tồn tại
    print('backbone: {}'.format(model.backbone_name))
    
    # Kiểm tra positional encoder
    if hasattr(model, 'pos_encoder') and hasattr(model.pos_encoder, 'dropout'):
        print('dropout: {}'.format(model.pos_encoder.dropout.p))
    elif hasattr(model, 'pos_decoder') and hasattr(model.pos_decoder, 'dropout'):
        print('dropout: {}'.format(model.pos_decoder.dropout.p))
    
    # Số lượng tham số luôn được tính
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
