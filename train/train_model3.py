# train/train_model_3.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.metrics import evaluate_model
from utils.histogram_filter import HistogramBilateralFilter
from models.model_3 import build_model_3
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os

# Data paths
DATA_ROOT = "final_data"
train_path = os.path.join(DATA_ROOT, "train")
val_path = os.path.join(DATA_ROOT, "val")
test_path = os.path.join(DATA_ROOT, "test")

# Preprocessing pipeline
transform = transforms.Compose([
    HistogramBilateralFilter(d=9, sigmaColor=75, sigmaSpace=75),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(train_path, transform=transform)
val_dataset = ImageFolder(val_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = build_model_3().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

# Training loop
def train_model(model, epochs=100, patience=3):
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%")

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100 * correct_val / total_val
        print(f"Validation Acc = {val_acc:.2f}%")
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_3.pth")
            print(" Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= patience:
            print(" Early stopping.")
            break

    return model

# Train the model
trained_model = train_model(model)

# Final evaluation
evaluate_model(trained_model, test_loader, device, model_name="model_3")
