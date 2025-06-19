import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from models.model_2 import AlzheimerCNN
from preprocessing.advanced_preprocessing import transform_with_filter
from utils.metrics import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data root
DATA_ROOT = "final_data"

# Load datasets
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_ROOT, "train"), transform=transform_with_filter)
val_dataset   = datasets.ImageFolder(root=os.path.join(DATA_ROOT, "val"), transform=transform_with_filter)
test_dataset  = datasets.ImageFolder(root=os.path.join(DATA_ROOT, "test"), transform=transform_with_filter)

# Wrap in DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Initialize model
model = AlzheimerCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

# Training loop
def train_model(epochs=100, patience=7):
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_2.pth")
            print("✅ Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔ Early stopping triggered at epoch {epoch+1}")
                break

    return model

# Train the model
trained_model = train_model()

# Evaluate
evaluate_model(trained_model, test_loader, device)
