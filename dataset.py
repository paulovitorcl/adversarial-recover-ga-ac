import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

CLASSES = ["NORMAL", "VIRAL", "BACTERIAL"]
IMG_SIZE = 128
IMG_EXTENSIONS = (".jpeg", ".jpg", ".png")


def label_from_filename(folder, name):
    """The Kaggle dataset only has NORMAL/PNEUMONIA folders; the pneumonia type is in the file name."""
    if folder == "NORMAL":
        return CLASSES.index("NORMAL")
    name = name.lower()
    if "virus" in name:
        return CLASSES.index("VIRAL")
    if "bacteria" in name:
        return CLASSES.index("BACTERIAL")
    return None


def list_samples(split_dir):
    """Returns [(image_path, label), ...] for a split folder such as chest_xray/test."""
    samples = []
    for folder in ["NORMAL", "PNEUMONIA"]:
        folder_path = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Expected folder not found: {folder_path}")
        for name in sorted(os.listdir(folder_path)):
            if not name.lower().endswith(IMG_EXTENSIONS):
                continue
            label = label_from_filename(folder, name)
            if label is not None:
                samples.append((os.path.join(folder_path, name), label))
    return samples


def split_train_val(samples, val_fraction=0.2, seed=42):
    """Stratified 80/20 split of the Kaggle train folder into train and validation."""
    labels = [label for _, label in samples]
    return train_test_split(samples, test_size=val_fraction, stratify=labels, random_state=seed)


def load_image(path, size=IMG_SIZE):
    """Reads an X-ray as a grayscale uint8 array of shape (size, size)."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(images):
    """uint8 array (H, W) or (N, H, W) -> float tensor in [0, 1] with a channel dimension."""
    tensor = torch.from_numpy(np.ascontiguousarray(images)).float().div(255.0)
    return tensor.unsqueeze(-3)


def to_uint8(tensor):
    """Float tensor in [0, 1] of shape (N, 1, H, W) -> uint8 array (N, H, W)."""
    return (tensor.squeeze(1).clamp(0, 1).mul(255.0).round().byte().cpu().numpy())


class XRayDataset(Dataset):
    """Yields (image, label) with image as a (1, 128, 128) tensor in [0, 1].

    Normalization is done inside the model, so attacks and recovery always work in pixel space.
    """

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = to_tensor(load_image(path))
        if self.transform:
            image = self.transform(image)
        return image, label
