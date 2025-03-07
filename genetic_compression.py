import numpy as np
import cv2
import os
import json
import random
import logging
from deap import base, creator, tools, algorithms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Optimization problem definition
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))  # Maximizar PSNR e SSIM
creator.create("Individual", list, fitness=creator.FitnessMax)

# Generation of random individuals
def generate_individual():
    return [random.choice(["jpeg", "png", "webp"]),  
            random.randint(10, 95),  
            random.choice([0, 90, 180, 270]),  
            random.uniform(0.5, 2.0),  
            random.uniform(0.5, 2.0)]  

# Assessment of the individual
def evaluate(individual, img_original, img_adversarial):
    formato, qualidade, rotacao, brilho, contraste = individual

    # Apply transformations
    img_recuperada = cv2.rotate(img_adversarial, rotacao)
    img_recuperada = cv2.convertScaleAbs(img_recuperada, alpha=contraste, beta=brilho * 50)

    # Image compression
    temp_path = f"temp.{formato}"
    cv2.imwrite(temp_path, img_recuperada, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    img_recuperada = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    os.remove(temp_path)

    # ✅ Ensure the image size is the same as the original
    if img_recuperada.shape != img_original.shape:
        img_recuperada = cv2.resize(img_recuperada, (img_original.shape[1], img_original.shape[0]))

    # Calculate quality metrics
    psnr_value = psnr(img_original, img_recuperada)
    ssim_value = ssim(img_original, img_recuperada)

    return psnr_value, ssim_value


# GA configuration
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# GA execution
def run_ga(images_original, images_adversarial, n_generations=20, pop_size=50):
    logging.info("Iniciando o Algoritmo Genético...")
    pop = toolbox.population(n=pop_size)

    for gen in range(n_generations):
        logging.info(f"Iniciando a geração {gen + 1}/{n_generations}")

        # GA evaluation for each pair of original and adversarial images
        for img_original, img_adversarial in zip(images_original, images_adversarial):
            for ind in pop:
                ind.fitness.values = evaluate(ind, img_original, img_adversarial)

        # Apply crossover and mutation operators
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

    # Select the best individual based on PSNR + SSIM metrics
    best_individual = tools.selBest(pop, k=1)[0]
    logging.info(f"Melhor configuração encontrada: {best_individual}")

    # Save the optimized configuration
    with open("best_compression.json", "w") as f:
        json.dump({
            "format": best_individual[0],
            "quality": best_individual[1],
            "rotation": best_individual[2],
            "brightness": best_individual[3],
            "contrast": best_individual[4]
        }, f)

    return best_individual
