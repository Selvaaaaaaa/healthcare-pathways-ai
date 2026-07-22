"""
cnn_model.py
============
Custom Small CNN Architecture built from scratch in PyTorch for Retinal DR Severity Classification.

Architecture:
  - 4 Convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool2D)
  - Global/Adaptive Average Pooling
  - Dropout (0.4) for regularization
  - Fully-Connected Classifier head (4 classes)

Input Shape: (3, 128, 128)
Output: Logits for 4 classes [0_no_dr, 1_mild, 2_moderate, 3_severe]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomRetinalCNN(nn.Module):
    """
    Custom 4-layer Convolutional Neural Network for Diabetic Retinopathy classification.
    """

    def __init__(self, num_classes: int = 4):
        super(CustomRetinalCNN, self).__init__()

        # Conv Block 1: 3 -> 32 channels (128x128 -> 64x64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Conv Block 2: 32 -> 64 channels (64x64 -> 32x32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Conv Block 3: 64 -> 128 channels (32x32 -> 16x16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Conv Block 4: 128 -> 256 channels (16x16 -> 8x8)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.4)

        # Fully Connected Classifier
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = CustomRetinalCNN(num_classes=4)
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    print(f"CustomRetinalCNN Smoke Test Output Shape: {output.shape}")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
