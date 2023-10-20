from torchvision import transforms
import torch.nn as nn
from src.datasets.autoencoder_dataset import AutoencoderDataset
from src.training.autoencoder_training import ModelTrainer
from src.models.autoencoder import ConvAutoencoder
from src.infra.valuable_objects import MonitorExperimentsConfig
from src.infra.monitor.monitor import WandbExperimentMonitor

transform = transforms.Compose([
    transforms.ToTensor()
])

x_train_dataset = "./databases/DIV2K/DIV2K_train_LR_240p"
y_train_dataset = "./databases/DIV2K/DIV2K_train_LR_480p"

autoencoder_dataset = AutoencoderDataset(x_train_dataset, y_train_dataset, transform)

experiment_monitor = WandbExperimentMonitor(MonitorExperimentsConfig(
    project_name="super-resolution",
    project_configs={
        "learning_rate": 0.00001,
        "architecture": "AutoEncoder",
        "epochs": 5,
    }
))

criterion = nn.BCELoss()

trainer = ModelTrainer(
    ConvAutoencoder,
    autoencoder_dataset,
    experiment_monitor,
    nn.BCELoss,
    0.2,
    5,
    0.00001
)

trainer.train(10)

trainer.save_model("/home/victor/pythonProjects/super-resolution/models_zoo/train_4.pth")
