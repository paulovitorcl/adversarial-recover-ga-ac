import torch.nn.functional as F
import torch


def fgsm_attack(model, images, labels, epsilon=0.03):
    """FGSM (Goodfellow et al., 2015).

    images: (N, 1, H, W) tensor in [0, 1] pixel space, so epsilon=0.03 is ~7.7 gray levels.
    Returns the adversarial images, clamped to [0, 1].
    """
    model.eval()
    images = images.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(images), labels)
    (grad,) = torch.autograd.grad(loss, images)
    adversarial = images + epsilon * grad.sign()
    return adversarial.clamp(0, 1).detach()
