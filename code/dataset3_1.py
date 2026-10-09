import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class SpeedDataset(Dataset):
    def __init__(self, csv_file, image_root, seq_len=10, is_training=True):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.image_root = image_root
        self.seq_len = seq_len
        self.is_training = is_training

        # Define transforms: training set uses color jitter, validation set only scales
        if self.is_training:
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Simulate different lighting/weather conditions
                T.ToTensor(), # Automatically normalizes to (0-1) and permutes to (C,H,W)
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet standard normalization
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        # Ensure no index out of bounds
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        images = []

        for i in range(self.seq_len):
            row = self.df.iloc[idx + i]
            img_path = os.path.join(self.image_root, row["image_path"])

            # Use PIL for compatibility with torchvision transforms
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224)) # Pre-resize
            except Exception as e:
                raise FileNotFoundError(f"Error loading image: {img_path}")

            # Apply transforms
            img_tensor = self.transform(img)
            images.append(img_tensor)

        # Stack sequence: (SEQ_LEN, C, H, W)
        images_seq = torch.stack(images)

        # Target speed: use the speed of the last frame in the sequence
        target = self.df.iloc[idx + self.seq_len - 1]["speed"]
        target = torch.tensor(target).float()

        return images_seq, target