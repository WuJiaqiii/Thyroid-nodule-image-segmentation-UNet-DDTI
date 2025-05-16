import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
from pathlib import Path
from PIL import Image
import os

class MedicalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = [path.name for path in list(Path(img_dir).rglob('*'))]
        self.mask_names = [name.split('.jpg')[0] + '_mask.jpg' for name in self.img_names]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_names[idx]))
        mask = Image.open(os.path.join(self.mask_dir, self.mask_names[idx]))
        
        if self.transform:
            img, mask = self.transform(img, mask)
            
        return img, mask

class MySampler(Sampler):
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

def collate_fn():
    pass

def create_dataloader(dataset, config, shuffle):
    return DataLoader(dataset=dataset,
                      batch_size=config.batch_size,
                      shuffle=shuffle,
                      num_workers=config.num_workers)
