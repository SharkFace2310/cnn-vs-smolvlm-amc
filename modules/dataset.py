import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os

class ConstellationDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        with open(json_file, 'r') as f:
            self.metadata = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        
        # Mapping κατηγοριών σε νούμερα (Encoding)
        self.mod_mapping = {mod: i for i, mod in enumerate(sorted(list(set(d['modulation'] for d in self.metadata))))}
        self.snr_mapping = {"low": 0, "medium": 1, "high": 2}
        self.iq_mapping = {"none": 0, "low": 1, "medium": 2, "high": 3}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.root_dir, item['file'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Labels για Multi-task
        labels = {
            'modulation': torch.tensor(self.mod_mapping[item['modulation']]),
            'snr': torch.tensor(self.snr_mapping[item['snr_level']]),
            'iq_imbalance': torch.tensor(self.iq_mapping[item['iq_imbalance_level']])
        }
        
        return image, labels