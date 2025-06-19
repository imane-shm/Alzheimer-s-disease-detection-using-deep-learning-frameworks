import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model_6 import get_mobilenetv3_large
from data.data_loaders import get_data_loaders
from utils.metrics import evaluate_model, save_confusion_matrix

def train_model_6(epochs=50, patience=3, lr=1e-4, weight_decay=1e-5):
    # Load preprocessed data with filters
    train_loader, val_loader, test_loader, class_names = get_data_loaders(preprocessing="histogram_bilateral")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_mobilenetv3_large().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_acc = correct / total
        print(f"Train Accuracy: {train_acc:.2%}")

        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.2%}")
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_6_mobilenetv3.pth")
            print("✅ Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        if patience_counter >= patience:
            print("⛔ Early stopping")
            break

    # Final evaluation
    model.load_state_dict(torch.load("best_model_6_mobilenetv3.pth"))
    model.eval()
    evaluate_model(model, test_loader, device, class_names=class_names, show_report=True)
    save_confusion_matrix(model, test_loader, class_names, "confusion_matrix_model_6_mobilenetv3.png", device=device)
