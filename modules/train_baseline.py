import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import json
import os

# --- 1. Dataset Class ---
class ConstellationDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        with open(json_file, 'r') as f:
            self.metadata = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        
        # Mapping κατηγοριών (Πρέπει να ταιριάζει με το dataset_generator.py)
        all_mods = sorted(list(set(d['modulation'] for d in self.metadata)))
        self.mod_mapping = {mod: i for i, mod in enumerate(all_mods)}
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
            
        return image, {
            'modulation': torch.tensor(self.mod_mapping[item['modulation']]),
            'snr': torch.tensor(self.snr_mapping[item['snr_level']]),
            'iq': torch.tensor(self.iq_mapping[item['iq_imbalance_level']])
        }

# --- 2. Multi-Task CNN Architecture ---
class MultiTaskCNN(nn.Module):
    def __init__(self, num_mods):
        super(MultiTaskCNN, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.mod_head = nn.Linear(num_ftrs, num_mods)
        self.snr_head = nn.Linear(num_ftrs, 3)
        self.iq_head = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        features = self.backbone(x)
        return self.mod_head(features), self.snr_head(features), self.iq_head(features)

# --- 3. Main Training Script ---
if __name__ == "__main__":
    # Hyperparameters & Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Εκπαίδευση σε: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Φόρτωση Δεδομένων
    full_dataset = ConstellationDataset('constellation_dataset/labels_train.json', 'constellation_dataset/images', transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Initialize Model & Optimizer
    model = MultiTaskCNN(num_mods=len(full_dataset.mod_mapping)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Simple Training Loop
    for epoch in range(10): # Ξεκίνα με 10 epochs
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            l_mod, l_snr, l_iq = labels['modulation'].to(device), labels['snr'].to(device), labels['iq'].to(device)

            optimizer.zero_grad()
            out_mod, out_snr, out_iq = model(images)
            
            loss = criterion(out_mod, l_mod) + criterion(out_snr, l_snr) + criterion(out_iq, l_iq)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Αποθήκευση του μοντέλου
    torch.save(model.state_dict(), "baseline_cnn.pth")
    print("Το μοντέλο αποθηκεύτηκε ως baseline_cnn.pth")