"""GA-AC: Genetic Algorithm-based Adaptive Compression.

For each adversarial image, a GA searches for the recovery transformation (rotation, scaling,
brightness, contrast and PNG/WebP compression) that maximizes a weighted combination of
PSNR, SSIM (against the clean image) and the classifier's confidence in the true class.
"""
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from deap import base, creator, tools, algorithms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import predict_proba

FORMATS = ("png", "webp")
ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)
INVERSE_ROTATIONS = (None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE)
PSNR_CAP = 50.0  # PSNR is divided by this value so it is on the same scale as SSIM and probability

# Chromosome: 5 genes in [0, 1], decoded by decode(). Keeping every gene continuous makes
# cxBlend and mutGaussian valid for the whole individual.
GENES = ("quality", "rotation", "scale", "brightness", "contrast")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


@dataclass
class GAConfig:
    pop_size: int = 50
    generations: int = 25
    cxpb: float = 0.7
    mutpb: float = 0.2
    indpb: float = 0.2
    sigma: float = 0.1
    tournsize: int = 3
    elite: int = 2
    # Weights of (PSNR, SSIM, classification confidence) in the fitness
    fitness_weights: tuple = field(default=(1.0, 1.0, 1.0))


def decode(individual, fmt):
    quality, rotation, scale, brightness, contrast = individual
    return {
        "format": fmt,
        # WebP: quality 1-100 (lossy). PNG: compression level 0-9 (lossless).
        "quality": int(round(1 + quality * 99)) if fmt == "webp" else int(round(quality * 9)),
        "rotation": 90 * min(int(rotation * 4), 3),
        "scale": 0.5 + 0.5 * scale,
        "brightness": -50.0 + 100.0 * brightness,
        "contrast": 0.5 + 1.5 * contrast,
    }


def apply_recovery(image, params):
    """Applies a decoded configuration to a uint8 grayscale image and returns a same-size image.

    Rotation and scaling are undone after compression: they only change how the compression
    grid aligns with the image (and the adversarial pattern). Brightness/contrast are kept.
    """
    h, w = image.shape
    k = params["rotation"] // 90
    out = cv2.rotate(image, ROTATIONS[k]) if k else image
    out = cv2.convertScaleAbs(out, alpha=params["contrast"], beta=params["brightness"])

    if params["scale"] < 1.0:
        small = (max(1, round(out.shape[1] * params["scale"])), max(1, round(out.shape[0] * params["scale"])))
        size = (out.shape[1], out.shape[0])
        out = cv2.resize(out, small, interpolation=cv2.INTER_AREA)

    if params["format"] == "webp":
        encode_args = [cv2.IMWRITE_WEBP_QUALITY, params["quality"]]
    else:
        encode_args = [cv2.IMWRITE_PNG_COMPRESSION, params["quality"]]
    ok, buffer = cv2.imencode(f".{params['format']}", out, encode_args)
    if not ok:
        raise RuntimeError(f"Could not encode image as {params['format']}")
    out = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    if params["scale"] < 1.0:
        out = cv2.resize(out, size, interpolation=cv2.INTER_LINEAR)
    if k:
        out = cv2.rotate(out, INVERSE_ROTATIONS[k])
    return out if out.shape == (h, w) else cv2.resize(out, (w, h))


def image_quality(original, recovered):
    return psnr(original, recovered, data_range=255), ssim(original, recovered, data_range=255)


def _build_toolbox(config):
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, random.random, n=len(GENES))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=config.sigma, indpb=config.indpb)
    toolbox.register("select", tools.selTournament, tournsize=config.tournsize)
    return toolbox


def run_ga(model, original, adversarial, label, fmt, device, config=GAConfig()):
    """Optimizes the recovery of one adversarial image.

    original, adversarial: uint8 arrays (H, W). label: true class index, used by the
    classification term of the fitness (set fitness_weights[2] = 0 for a label-free GA).
    Returns (params, recovered_image, scores).
    """
    if fmt not in FORMATS:
        raise ValueError(f"Format must be one of {FORMATS}")
    w_psnr, w_ssim, w_cls = config.fitness_weights
    toolbox = _build_toolbox(config)

    def evaluate_population(individuals):
        # Recover all individuals first so the classifier runs in a single batch
        params = [decode(ind, fmt) for ind in individuals]
        recovered = np.stack([apply_recovery(adversarial, p) for p in params])
        probs = predict_proba(model, recovered, device)[:, label] if w_cls else np.zeros(len(recovered))
        for ind, image, prob in zip(individuals, recovered, probs):
            p, s = image_quality(original, image)
            ind.fitness.values = (w_psnr * min(p, PSNR_CAP) / PSNR_CAP + w_ssim * s + w_cls * prob,)

    pop = toolbox.population(n=config.pop_size)
    evaluate_population(pop)
    hof = tools.HallOfFame(config.elite)
    hof.update(pop)

    for _ in range(config.generations):
        offspring = toolbox.select(pop, len(pop) - config.elite)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=config.cxpb, mutpb=config.mutpb)
        for ind in offspring:
            ind[:] = np.clip(ind, 0.0, 1.0).tolist()
        evaluate_population([ind for ind in offspring if not ind.fitness.valid])
        # Elitism: the best individuals found so far always survive
        pop = offspring + [toolbox.clone(ind) for ind in hof]
        hof.update(pop)

    best = hof[0]
    params = decode(best, fmt)
    recovered = apply_recovery(adversarial, params)
    p, s = image_quality(original, recovered)
    prob = predict_proba(model, recovered[None], device)[0, label]
    scores = {"psnr": float(p), "ssim": float(s), "prob_true": float(prob), "fitness": float(best.fitness.values[0])}
    return params, recovered, scores
