import os
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.model_01 import AlzheimerCNN
from utils.metrics import evaluate_model
from preprocessing.standard_preprocessing import standard_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_ROOT = "final_data"

# Transforms
transform = standard_transform

# Datasets
train_dataset = ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
val_dataset   = ImageFolder(os.path.join(DATA_ROOT, "val"), transform=transform)
test_dataset  = ImageFolder(os.path.join(DATA_ROOT, "test"), transform=transform)

# Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Model
model = AlzheimerCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

# Training loop
from utils.train_utils import train_val_loop
avg_train_losses, train_accs, val_accs, all_labels, all_preds = train_val_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device=device,
    epochs=100,
    patience=7,
    save_path="best_model.pth"
)

# Evaluation
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
evaluate_model(model, test_loader, device)
