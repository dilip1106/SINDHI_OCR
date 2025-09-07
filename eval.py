import torch
from train import CRNN, SindhiOCRDataset
from torch.utils.data import DataLoader
import logging
import argparse
from Levenshtein import distance

def evaluate_model(args):
    # Load checkpoint
    checkpoint = torch.load(args.model_path)
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    # Initialize model
    model = CRNN(num_classes=len(char_to_idx) + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load validation dataset
    val_dataset = SindhiOCRDataset(args.val_images, args.val_labels, rtl=args.rtl)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    total_cer = 0
    total_wer = 0
    num_samples = 0

    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.log_softmax(2).argmax(2)

            # Convert predictions to text
            batch_texts = []
            for pred in predictions:
                text = []
                previous = -1
                for p in pred:
                    p = p.item()
                    if p != len(char_to_idx) and p != previous:  # Remove blanks and duplicates
                        text.append(idx_to_char[p])
                    previous = p
                batch_texts.append(''.join(text))

            # Calculate metrics
            for pred_text, true_text in zip(batch_texts, texts):
                # Character Error Rate
                cer = distance(pred_text, true_text) / len(true_text)
                total_cer += cer
                
                # Word Error Rate
                pred_words = pred_text.split()
                true_words = true_text.split()
                wer = distance(pred_words, true_words) / len(true_words)
                total_wer += wer
                
                num_samples += 1

    avg_cer = total_cer / num_samples
    avg_wer = total_wer / num_samples
    
    logging.info(f"Average Character Error Rate: {avg_cer:.4f}")
    logging.info(f"Average Word Error Rate: {avg_wer:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Sindhi OCR Model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--val-images', type=str, required=True, help='Path to validation images')
    parser.add_argument('--val-labels', type=str, required=True, help='Path to validation labels')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--rtl', action='store_true', help='Right to left text')
    
    args = parser.parse_args()
    evaluate_model(args)