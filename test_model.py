import torch
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
sys.path.append(str(Path(__file__).parent.resolve())+'/src')

from config import MODEL, N_HEADS, \
                    ENC_LAYERS, DEC_LAYERS, HIDDEN, \
                    DROPOUT, DEVICE, ALPHABET, TEST_TRANSFORMS
from utils import process_image, indicies_to_text

def load_model(checkpoint_path):
    """Load the OCR model from checkpoint"""
    # Create character mappings
    char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
    idx2char = {idx: char for idx, char in enumerate(ALPHABET)}
    
    # Initialize model
    if MODEL == 'model1':
        from models import model1
        model = model1.TransformerModel(len(ALPHABET), hidden=HIDDEN, 
                                      enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                                      nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
    elif MODEL == 'model2':
        from models import model2
        model = model2.TransformerModel(len(ALPHABET), hidden=HIDDEN, 
                                      enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                                      nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    return model, idx2char

def predict_image(model, image_path, idx2char):
    """Predict text from a single image"""
    try:
        # Load and preprocess image
        try:
            # Try PIL first
            with Image.open(image_path) as img:
                img = np.array(img.convert('RGB'))
        except Exception as e:
            print(f"PIL failed to load image, trying cv2: {str(e)}")
            # If PIL fails, try cv2
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Failed to load image: {image_path}")
        
        # Process image
        img = process_image(img)
        img = img / 255.0  # Normalize
        img = np.transpose(img, (2, 0, 1))  # CHW format
        img = torch.FloatTensor(img).unsqueeze(0).to(DEVICE)
        
        # Get prediction
        with torch.no_grad():
            out_indexes = model.predict(img)
            pred_text = indicies_to_text(out_indexes[0], idx2char)
            
        return pred_text
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test OCR model on a single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_6.pt', help='Path to model checkpoint')
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, idx2char = load_model(args.checkpoint)
    
    print(f"Processing image {args.image}...")
    predicted_text = predict_image(model, args.image, idx2char)
    
    if predicted_text:
        print("\nPredicted text:")
        print("-" * 50)
        print(predicted_text)
        print("-" * 50)

if __name__ == "__main__":
    main() 