"""Runs the four scenarios of the paper on the test set: Baseline, Adversarial (FGSM),
PNG Recovery and WebP Recovery (GA-AC). Results are written to --out."""
import argparse
import csv
import json
import logging
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from attack import fgsm_attack
from dataset import CLASSES, list_samples, load_image, to_tensor, to_uint8
from genetic_compression import FORMATS, GAConfig, run_ga
from model import get_device, load_model, predict_proba, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": precision, "recall": recall, "f1": f1}


def save_confusion_matrix(labels, preds, title, path):
    cm = confusion_matrix(labels, preds, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_examples(images_by_scenario, labels, path):
    """One row per class: original, attacked and recovered images (as in Figure 5 of the paper)."""
    rows = [int(np.flatnonzero(labels == c)[0]) for c in range(len(CLASSES)) if np.any(labels == c)]
    names = list(images_by_scenario)
    fig, axes = plt.subplots(len(rows), len(names), figsize=(3 * len(names), 3 * len(rows)), squeeze=False)
    for r, idx in enumerate(rows):
        for c, name in enumerate(names):
            axes[r, c].imshow(images_by_scenario[name][idx], cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_title(f"{name}\n{CLASSES[labels[idx]]}" if r == 0 else CLASSES[labels[idx]])
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def attack_images(model, images, labels, epsilon, device, batch_size=64):
    adversarial = []
    for start in range(0, len(images), batch_size):
        batch = to_tensor(images[start:start + batch_size]).to(device)
        targets = torch.as_tensor(labels[start:start + batch_size], device=device)
        adversarial.append(to_uint8(fgsm_attack(model, batch, targets, epsilon)))
    # Adversarial images are quantized to 8 bits, as they would be when stored or transmitted
    return np.concatenate(adversarial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="chest_xray", help="Folder with the test/ subfolder")
    parser.add_argument("--weights", default="weights/cnn_xray.pth")
    parser.add_argument("--out", default="results")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    parser.add_argument("--limit", type=int, default=None, help="Use a random subset of N test images")
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--cxpb", type=float, default=0.7)
    parser.add_argument("--mutpb", type=float, default=0.2)
    parser.add_argument("--fitness-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=("PSNR", "SSIM", "CLS"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit(f"Weights not found at {args.weights}. Train the model first: python train.py")

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.weights, device)

    samples = list_samples(os.path.join(args.data, "test"))
    if args.limit:
        samples = random.Random(args.seed).sample(samples, min(args.limit, len(samples)))
    originals = np.stack([load_image(path) for path, _ in samples])
    labels = np.array([label for _, label in samples])
    logging.info(f"Device: {device} | test images: {len(samples)}")

    images = {"Baseline": originals}
    logging.info(f"Applying FGSM (epsilon={args.epsilon})...")
    images["Adversarial"] = attack_images(model, originals, labels, args.epsilon, device)

    ga_config = GAConfig(pop_size=args.pop_size, generations=args.generations, cxpb=args.cxpb,
                         mutpb=args.mutpb, fitness_weights=tuple(args.fitness_weights))
    for fmt in args.formats:
        scenario = f"{fmt.upper() if fmt == 'png' else 'WebP'} Recovery"
        logging.info(f"Running GA-AC with {fmt}...")
        recovered, rows = [], []
        for i, (path, _) in enumerate(samples):
            start = time.perf_counter()
            params, image, scores = run_ga(model, originals[i], images["Adversarial"][i], int(labels[i]),
                                           fmt, device, ga_config)
            recovered.append(image)
            rows.append({"image": path, "label": CLASSES[labels[i]], **params, **scores,
                         "seconds": time.perf_counter() - start})
            if (i + 1) % 50 == 0 or i + 1 == len(samples):
                logging.info(f"  {i + 1}/{len(samples)} images")
        images[scenario] = np.stack(recovered)
        with open(os.path.join(args.out, f"recovery_params_{fmt}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    metrics = {}
    for scenario, scenario_images in images.items():
        preds = predict_proba(model, scenario_images, device).argmax(1)
        metrics[scenario] = compute_metrics(labels, preds)
        slug = scenario.lower().replace(" ", "_")
        save_confusion_matrix(labels, preds, scenario, os.path.join(args.out, f"confusion_matrix_{slug}.pdf"))
        logging.info(f"{scenario}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics[scenario].items()))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"config": vars(args), "metrics": metrics}, f, indent=2)
    save_examples(images, labels, os.path.join(args.out, "examples.png"))
    logging.info(f"Results saved to {args.out}/")


if __name__ == "__main__":
    main()
