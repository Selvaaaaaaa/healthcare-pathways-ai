"""
transfer_model.py
=================
Transfer Learning Model using Pretrained ResNet18 Backbone for DR Severity Classification.

Workflow:
  1. Load ResNet18 pretrained weights (ImageNet initialization).
  2. Freeze feature extraction backbone (conv1 through layer4).
  3. Replace final fully-connected classifier `model.fc` with a custom 4-class head.
  4. Provide helper `unfreeze_top_layers` to unfreeze top block (`layer4`) for fine-tuning.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def build_transfer_resnet18(num_classes: int = 4, freeze_base: bool = True) -> nn.Module:
    """
    Build a ResNet18 Transfer Learning model.

    Parameters
    ----------
    num_classes : int
        Number of target severity classes (4).
    freeze_base : bool
        If True, freeze all convolutional backbone parameters initially.

    Returns
    -------
    nn.Module
        ResNet18 model modified for 4-class DR classification.
    """
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older torchvision or offline environments
        model = models.resnet18(pretrained=True)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def unfreeze_layer4(model: nn.Module) -> None:
    """
    Unfreeze layer4 and classifier head for fine-tuning.
    """
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True


if __name__ == "__main__":
    model = build_transfer_resnet18(num_classes=4, freeze_base=True)
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    print(f"ResNet18 Transfer Model Output Shape: {output.shape}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters (Frozen Base): {trainable:,} / {total:,}")
