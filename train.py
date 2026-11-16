"""Trains the X-ray classifier with the setup from the paper and saves weights/cnn_xray.pth."""
import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset import XRayDataset, list_samples, split_train_val
from model import XRayCNN, get_device, save_checkpoint, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            preds.extend(model(images.to(device)).argmax(1).cpu().numpy())
            labels.extend(targets.numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="chest_xray", help="Folder with train/ and test/ subfolders")
    parser.add_argument("--out", default="weights/cnn_xray.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    logging.info(f"Device: {device}")

    train_samples, val_samples = split_train_val(
        list_samples(os.path.join(args.data, "train")), args.val_fraction, args.seed)
    test_samples = list_samples(os.path.join(args.data, "test"))
    logging.info(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # Data augmentation from the paper: flipping and contrast adjustment
    augmentation = T.Compose([T.RandomHorizontalFlip(), T.ColorJitter(contrast=0.2)])
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers)
    train_loader = DataLoader(XRayDataset(train_samples, augmentation), shuffle=True, **loader_args)
    val_loader = DataLoader(XRayDataset(val_samples), **loader_args)
    test_loader = DataLoader(XRayDataset(test_samples), **loader_args)

    model = XRayCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_acc, val_f1 = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_acc": val_acc, "val_f1": val_f1})
        logging.info(f"Epoch {epoch}/{args.epochs} - loss {np.mean(losses):.4f} - "
                     f"val acc {val_acc:.4f} - val F1 {val_f1:.4f}")

        # Keep the epoch with the best validation F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    logging.info(f"Test acc {test_acc:.4f} - test F1 {test_f1:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    metadata = {
        "config": vars(args),
        "history": history,
        "test_metrics": {"accuracy": test_acc, "f1_macro": test_f1},
    }
    save_checkpoint(model, args.out, **metadata)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Weights saved to {args.out}")


if __name__ == "__main__":
    main()
