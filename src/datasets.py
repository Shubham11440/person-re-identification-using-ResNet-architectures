# src/datasets.py

"""
Dataset loading and preprocessing utilities for person re-identification.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re

# Define the image transformations as specified in the paper 
# For training, we include data augmentation techniques.
transform_train = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((384, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(scale=(0.02, 0.4), ratio=(0.3, 3.3)),
])

# For evaluation (query and gallery), we only apply the necessary resizing and normalization.
transform_eval = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PersonReIDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading person re-identification data.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        self.dataset = self._process_dir(self.img_paths)

    def _process_dir(self, img_paths):
        """
        Parses image filenames to extract person IDs and camera IDs.
        Example filename: 0002_c1s1_000451_03.jpg
        - Person ID: 0002
        - Camera ID: 1
        """
        dataset = []
        pid_container = set()
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            # Use regex to handle variations like '-1' for junk images
            match = re.match(r'([-\d]+)_c(\d+)', filename)
            if match:
                pid, camid = map(int, match.groups())
                if pid == -1:
                    continue  # Skip junk images
                pid_container.add(pid)
                dataset.append((img_path, pid, camid))
        
        # Create a mapping from original PIDs to a zero-indexed label
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        # Remap the pids in the dataset
        relabeled_dataset = []
        for path, pid, camid in dataset:
            relabeled_dataset.append((path, pid2label[pid], camid))
            
        return relabeled_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, pid, camid = self.dataset[idx]
        
        # Load image with PIL
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, pid, camid


# Optional: Add a small test block to verify the dataset loader
if __name__ == '__main__':
    # This block will only run when you execute `python src/datasets.py` directly
    # It's a good way to test if the class works as expected.
    print("Testing PersonReIDDataset...")
    # Update this path to point to your Market-1501 training directory
    market_train_dir = '../data/Market-1501-v15.09.15/bounding_box_train'
    
    if os.path.exists(market_train_dir):
        # Create a dataset instance for training
        train_dataset = PersonReIDDataset(data_dir=market_train_dir, transform=transform_train)
        
        # Check the total number of images and unique persons
        num_images = len(train_dataset)
        num_pids = len(set([pid for _, pid, _ in train_dataset.dataset]))
        
        print(f"Successfully loaded {num_images} images for {num_pids} unique persons.")
        
        # Get the first item to check its format
        image, pid, camid = train_dataset[0]
        
        print("\n--- Sample Data ---")
        print(f"Image tensor shape: {image.shape}")
        print(f"Person ID (label): {pid}")
        print(f"Camera ID: {camid}")
        print("Test complete. The Dataset class appears to be working correctly. ✅")
    else:
        print(f"Error: Test directory not found at '{market_train_dir}'.")
        print("Please update the path in the test block of src/datasets.py.")
