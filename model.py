import random

import numpy as np
import torch
import torch.nn as nn

from dataset import CLASSES, IMG_SIZE, to_tensor


class XRayCNN(nn.Module):
    """Small CNN based on Kesim et al. (EBBT 2019): 3 conv layers + 2 dense layers.

    Input: (N, 1, 128, 128) in [0, 1]. Output: logits (softmax is applied in predict_proba and,
    during training, inside CrossEntropyLoss).
    """

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(0.5))
        self.register_buffer("std", torch.tensor(0.5))
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) ** 2, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.classifier(self.features(x))


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model, path, **metadata):
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, **metadata}, path)


def load_model(weights_path, device="cpu"):
    """Loads the trained classifier saved by train.py."""
    checkpoint = torch.load(weights_path, map_location=device)
    if checkpoint.get("classes", CLASSES) != CLASSES:
        raise ValueError(f"Checkpoint classes {checkpoint['classes']} differ from {CLASSES}")
    model = XRayCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, images, device, batch_size=256):
    """Class probabilities for a uint8 array of images (N, H, W)."""
    model.eval()
    probs = []
    for start in range(0, len(images), batch_size):
        batch = to_tensor(images[start:start + batch_size]).to(device)
        probs.append(torch.softmax(model(batch), dim=1).cpu())
    return torch.cat(probs).numpy()
