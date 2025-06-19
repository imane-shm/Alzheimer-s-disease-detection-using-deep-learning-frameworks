# models/model_3.py

import torch.nn as nn
from torchvision import models

def build_model_3():
    # Load pretrained DenseNet201
    densenet = models.densenet201(pretrained=True)

    # Freeze all convolutional layers
    for param in densenet.parameters():
        param.requires_grad = False

    # Replace the classifier for 4-class Alzheimer classification
    num_features = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 4)
    )

    return densenet
