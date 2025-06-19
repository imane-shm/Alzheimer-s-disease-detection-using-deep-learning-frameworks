import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model_4 import get_resnet18_model
from data.data_loaders import get_data_loaders
from utils.metrics import evaluate_model, save_confusion_matrix

def train_model_4(epochs=100, patience=3, lr=1e-4, weight_decay=1e-5):
    # Load subject-split, filtered data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(preprocessing="histogram_bilateral")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item())

        train_acc = 100 * correct / total
        print(f"Train Accuracy: {train_acc:.2f}%")

        # Validate
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.2f}%")
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_5_resnet18.pth")
            print(" Saved best model.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= patience:
            print(f" Early stopping. Best Val Accuracy: {best_acc:.2f}%")
            break

    # Load best model and test
    model.load_state_dict(torch.load("best_model_5_resnet18.pth"))
    model.eval()
    evaluate_model(model, test_loader, device, class_names=class_names, show_report=True)
    save_confusion_matrix(model, test_loader, class_names, "confusion_matrix_model_5_resnet18.png", device=device)
