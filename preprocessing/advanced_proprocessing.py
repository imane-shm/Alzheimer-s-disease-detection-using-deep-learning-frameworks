# preprocessing/advanced_preprocessing.py

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

class HistogramEqualizationAndBilateralFilter:
    def __call__(self, img):
        """
        Apply histogram equalization + bilateral filter on each channel.
        """
        img_np = np.array(img)

        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            processed_channels = []
            for i in range(3):
                channel = img_np[:, :, i]
                eq = cv2.equalizeHist(channel)
                filt = cv2.bilateralFilter(eq, d=9, sigmaColor=75, sigmaSpace=75)
                processed_channels.append(filt)
            processed_img = cv2.merge(processed_channels)
        else:
            # For grayscale images, apply directly
            processed_img = cv2.equalizeHist(img_np)
            processed_img = cv2.bilateralFilter(processed_img, d=9, sigmaColor=75, sigmaSpace=75)

        return Image.fromarray(processed_img)

def get_advanced_transform():
    """
    Compose histogram equalization + bilateral filtering with base transform.
    """
    base_transforms = transforms.Compose([
        HistogramEqualizationAndBilateralFilter(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return base_transforms
