import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model_7 import VisionTransformer
from data.data_loaders import get_data_loaders
from utils.metrics import evaluate_model, save_confusion_matrix

def train_model_7(epochs=100, patience=5, lr=1e-4, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load standard preprocessing
    train_loader, val_loader, test_loader, class_names = get_data_loaders(preprocessing="standard")

    model = VisionTransformer(num_classes=len(class_names)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader, device)
        print(f" → train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_7_vit.pth")
            print("✅ Saved best model")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

    model.load_state_dict(torch.load("best_model_7_vit.pth"))
    model.eval()
    evaluate_model(model, test_loader, device, class_names=class_names, show_report=True)
    save_confusion_matrix(model, test_loader, class_names, "confusion_matrix_model_7_vit.png", device=device)
