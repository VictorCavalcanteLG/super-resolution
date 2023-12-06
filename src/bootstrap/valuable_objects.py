from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CyclicLR, CosineAnnealingLR
from src.models.autoencoder import ConvAutoencoder


LOSS = {
    "bce_loss": nn.BCELoss
}

MODEL = {
    "autoencoder": ConvAutoencoder
}

LEARNING_RATE_SCHEDULER = {
    "reduce_on_plateau": ReduceLROnPlateau,
    "exponential": ExponentialLR,
    "cyclic": CyclicLR,
    "cosine": CosineAnnealingLR
}
