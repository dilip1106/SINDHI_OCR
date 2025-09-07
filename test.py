import argparse
import torch
from PIL import Image
import logging
from pathlib import Path
import numpy as np
from train import CRNN  # Import the model architecture from train.py

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_image(image_path, imgH=48, maxW=512, rtl=True):
    """Preprocess the image for model input"""
    try:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = Image.open(image_path).convert('L')
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")
            
        new_w = min(int(imgH * w / h), maxW)
        img = img.resize((new_w, imgH), Image.Resampling.BILINEAR)
        
        img_np = np.array(img, dtype=np.float32) / 255.0
        if rtl:
            img_np = np.fliplr(img_np).copy()
        
        img_tensor = torch.tensor(img_np[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        return img_tensor
        
    except Exception as e:
        logging.error(f"Image preprocessing failed for {image_path}: {str(e)}")
        raise

def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        model = CRNN(num_classes=len(char_to_idx) + 1)  # +1 for CTC blank
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, idx_to_char
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def decode_prediction(output, idx_to_char):
    """Decode model output to text"""
    pred = output.argmax(2).squeeze(0)
    
    decoded = []
    previous = -1
    max_idx = max(idx_to_char.keys())
    
    # Add debug logging
    logging.debug(f"Available indices: {sorted(idx_to_char.keys())}")
    
    for p in pred:
        p_idx = p.item()
        if p_idx != max_idx and p_idx != previous:  # Not blank and not duplicate
            try:
                decoded.append(idx_to_char[p_idx])
            except KeyError:
                logging.warning(f"Unknown character index: {p_idx}")
                decoded.append('?')  # Replace unknown characters with ?
        previous = p_idx
    
    result = ''.join(decoded)
    logging.debug(f"Raw prediction indices: {pred.tolist()}")
    logging.debug(f"Decoded text: {result}")
    return result

def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    try:
        model, idx_to_char = load_model(args.model_path, device)
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        return

    # Process input path (file or directory)
    input_path = Path(args.input)
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = list(input_path.glob("**/*.png"))
    
    if not paths:
        logging.error(f"No images found at {args.input}")
        return

    # Process images
    for img_path in paths:
        try:
            logging.info(f"Processing image: {img_path}")
            
            # Preprocess image
            img_tensor = preprocess_image(img_path, args.imgH, args.maxW, args.rtl)
            img_tensor = img_tensor.to(device)
            
            logging.debug(f"Image tensor shape: {img_tensor.shape}")

            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                output = output.log_softmax(2)
            
            # Decode prediction
            predicted_text = decode_prediction(output, idx_to_char)
            
            logging.info(f"Image: {img_path}")
            logging.info(f"Predicted text: {predicted_text}")
            
        except Exception as e:
            logging.error(f"Error processing {img_path}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Sindhi OCR Model')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to image file or directory of images')
    parser.add_argument('--imgH', type=int, default=48,
                      help='Image height')
    parser.add_argument('--maxW', type=int, default=512,
                      help='Maximum image width')
    parser.add_argument('--rtl', action='store_true',
                      help='Right to left text')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    test_model(args)