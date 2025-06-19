import torch.nn as nn
from torchvision import models

def get_resnet_model(version="resnet50", num_classes=4, dropout_rate=0.4):
    if version == "resnet50":
        resnet = models.resnet50(pretrained=True)
    elif version == "resnet18":
        resnet = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported ResNet version: {version}")

    # Freeze all layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace FC layer
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )

    return resnet
