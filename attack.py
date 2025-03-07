import torch
import torch.nn as nn

# Function of FGSM adversarial attack
def fgsm_attack(model, image, label, epsilon=0.03):
    model.eval()

    if image.ndim == 5:
        image = image.squeeze(0)  

    image.requires_grad = True

    # Forward pass
    output = model(image) 
    loss = nn.CrossEntropyLoss()(output, label)

    
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * image.grad.sign()

    adversarial_image = image + perturbation

    adversarial_image = torch.clamp(adversarial_image, 0, 1)

    return adversarial_image.detach()  # Remove o gradiente da imagem adversarial