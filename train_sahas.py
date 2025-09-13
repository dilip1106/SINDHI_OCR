import argparse
import unicodedata
from pathlib import Path
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class SindhiOCRDataset(Dataset):
    def __init__(self, root_dir: str, labels_file: str, imgH: int = 48, maxW: int = 512, rtl: bool = True):
        self.root_dir = Path(root_dir)
        self.labels_file = Path(labels_file)
        self.imgH = imgH
        self.maxW = maxW
        self.rtl = rtl

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        # Try different encodings
        encodings = ['utf-16', 'utf-8', 'utf-8-sig']
        for encoding in encodings:
            try:
                with open(self.labels_file, "r", encoding=encoding) as f:
                    self.labels = [line.strip() for line in f if line.strip()]
                logging.info(f"Successfully read labels file with {encoding} encoding")
                break
            except UnicodeError:
                continue
        else:
            raise UnicodeError(f"Could not read labels file with any of these encodings: {encodings}")

        self.samples = self._build_samples()
        logging.info(f"Loaded dataset with {len(self.samples)} samples")

    def _build_samples(self):
        samples = []
        for idx, label in enumerate(self.labels):
            folder_path = self.root_dir / str(idx)
            if not folder_path.exists():
                logging.warning(f"Skipping missing folder: {folder_path}")
                continue

            for img_path in folder_path.glob("*.png"):
                samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("L")
            w, h = img.size
            new_w = min(int(self.imgH * w / h), self.maxW)
            img = img.resize((new_w, self.imgH), Image.Resampling.BILINEAR)
            
            img_np = np.array(img, dtype=np.float32) / 255.0
            if self.rtl:
                img_np = np.fliplr(img_np).copy()
            
            img_tensor = torch.tensor(img_np[np.newaxis, :, :], dtype=torch.float32)
            return img_tensor, label
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
            raise

class CRNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            self._conv_block(1, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512, dropout=dropout_rate),
            nn.AdaptiveAvgPool2d((1, None))
        )
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.classifier = nn.Linear(512, num_classes)

    def _conv_block(self, in_c: int, out_c: int, dropout: float = 0):
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)
        batch, channels, height, width = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        
        # RNN sequence processing
        rnn_out, _ = self.rnn(conv)
        output = self.classifier(rnn_out)
        return output

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Initialize dataset and dataloader
        dataset = SindhiOCRDataset(
            args.images_dir, 
            args.labels,
            imgH=args.imgH,
            maxW=args.maxW,
            rtl=args.rtl
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Create character map
        chars = sorted(set("".join(dataset.labels)))
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        num_classes = len(chars) + 1  # +1 for CTC blank

        # Initialize model
        model = CRNN(num_classes=num_classes)
        
        # Initialize training state
        start_epoch = 0
        best_loss = float('inf')
        
        # Load checkpoint if resuming
        if args.resume:
            if os.path.isfile(args.resume):
                logging.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint['loss']
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                char_to_idx = checkpoint['char_to_idx']
                logging.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                logging.error(f"No checkpoint found at '{args.resume}'")
                return

        model = model.to(device)

        # Loss and optimizer
        criterion = nn.CTCLoss(blank=num_classes-1, zero_infinity=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        best_loss = float('inf')
        for epoch in range(start_epoch, args.epochs):
            try:
                model.train()
                total_loss = 0
                current_loss = 0  # Add this to track current loss
                
                for batch_idx, (images, texts) in enumerate(train_loader):
                    images = images.to(device)
                    
                    # Prepare target sequences
                    target_lengths = []
                    targets = []
                    for text in texts:
                        indices = [char_to_idx[c] for c in text]
                        targets.extend(indices)
                        target_lengths.append(len(indices))
                    
                    targets = torch.tensor(targets, dtype=torch.long).to(device)
                    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
                    
                    # Forward pass
                    outputs = model(images)
                    output_log_probs = outputs.log_softmax(2)
                    input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)
                    
                    # Calculate loss
                    loss = criterion(output_log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    current_loss = total_loss / (batch_idx + 1)  # Update current loss
                    
                    if batch_idx % args.log_interval == 0:
                        logging.info(f'Epoch: {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

                avg_loss = total_loss / len(train_loader)
                scheduler.step(avg_loss)
                
                # Save model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'char_to_idx': char_to_idx,
                    }, args.model_save_path)
                    logging.info(f'Saved best model with loss: {best_loss:.4f}')

            except KeyboardInterrupt:
                logging.info("\nTraining interrupted by user. Saving checkpoint...")
                # Use current_loss instead of avg_loss for interrupted training
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,  # Use current_loss here
                    'char_to_idx': char_to_idx,
                }, f'interrupted_checkpoint_epoch_{epoch}.pth')
                logging.info(f'Checkpoint saved to interrupted_checkpoint_epoch_sahas_s{epoch}.pth')
                return  # Exit training gracefully

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Sindhi OCR Model')
    parser.add_argument('--images-dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels file')
    parser.add_argument('--model-save-path', type=str, default='sindhi_ocr_model.pth', help='Path to save the model')
    parser.add_argument('--imgH', type=int, default=48, help='Image height')
    parser.add_argument('--maxW', type=int, default=512, help='Maximum image width')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--rtl', action='store_true', help='Right to left text')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train_model(args)