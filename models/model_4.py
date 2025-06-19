import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes=4, dropout_rate=0.4):
    resnet18 = models.resnet18(pretrained=True)

    # Freeze all convolutional layers
    for param in resnet18.parameters():
        param.requires_grad = False

    # Replace the classification head
    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )

    return resnet18
