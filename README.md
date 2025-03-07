# X-ray Classification with Adversarial Attacks and Optimized Compression

This project implements a pipeline for classifying X-ray images using a Convolutional Neural Network (CNN). The workflow includes:

- **Data Loading and Preprocessing**: Images are read in grayscale and normalized.
- **Model Definition**: A CNN is defined to classify the images into 3 categories (NORMAL, VIRAL, BACTERIAL).
- **Model Evaluation**: The model is evaluated before and after applying adversarial attacks.
- **Adversarial Attack**: The FGSM technique is used to create adversarial images.
- **Genetic Algorithm Optimization**: A function optimizes image compression to restore quality.
- **Results Visualization**: Confusion matrices are displayed before and after the attack and recovery.

## Project Structure

- **script.py**: Main script that executes the complete pipeline.
- **chest_xray/train**: Directory containing the dataset, organized into subfolders:
  - **NORMAL**: X-ray images without pneumonia.
  - **PNEUMONIA**: X-ray images with pneumonia.  
    *Note: File names must contain the words "virus" or "bacteria" to differentiate the pneumonia types.*
- **genetic_compression.py**: Module containing the `run_ga` function for compression optimization.
- **attacks.py**: Module containing the `fgsm_attack` function for applying the adversarial attack.

## Requirements

- **Python**: Version 3.7 or later.
- **Python Libraries**:
  - numpy
  - opencv-python (cv2)
  - torch
  - torchvision
  - matplotlib
  - scikit-learn

*Note: Ensure that the `genetic_compression.py` and `attacks.py` modules are implemented and located in the same directory as the script or accessible via PYTHONPATH.*

## Installation

1. **Clone the repository**:
   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_DIRECTORY>

2. **Virtual environment**:
   ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows

3. **Install the dependencies**:
   ```bash
    pip install numpy opencv-python torch torchvision matplotlib scikit-learn

## Script Execution

   ```bash
   python adversarial_experiment.py
