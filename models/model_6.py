import torch.nn as nn
from torchvision import models

def get_mobilenetv3_large(num_classes=4, dropout_rate=0.4):
    mobilenet = models.mobilenet_v3_large(pretrained=True)

    # Freeze feature extractor
    for param in mobilenet.features.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = mobilenet.classifier[0].in_features
    mobilenet.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )

    return mobilenet
