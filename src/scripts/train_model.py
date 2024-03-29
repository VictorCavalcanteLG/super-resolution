import os
from torchvision import transforms

from src.datasets.image_patch_dataset import ImagesPatchDataset
from src.training.autoencoder_training import ModelTrainer
from src.infra.valuable_objects import MonitorExperimentsConfig
from src.infra.monitor.monitor import WandbExperimentMonitor
from src.bootstrap.bootstrap import Bootstrap

transform = transforms.Compose([
    transforms.ToTensor()
])

CONFIG_FILE = os.getenv('MODEL_CONFIG_PATH')

variables = Bootstrap(CONFIG_FILE)

autoencoder_dataset = ImagesPatchDataset(variables.x_train_dataset_path, variables.y_train_dataset_path, window_size=120, stride=120, transforms=transform)

entity = "victor-cavalcante"
project = "super-resolution"
job_name = "walkthrough_example"

experiment_monitor = WandbExperimentMonitor(MonitorExperimentsConfig(
    project_name=project,
    project_configs={
        "learning_rate": variables.learning_rate,
        # "architecture": "AutoEncoder",
        "epochs": variables.num_epochs,
    },
    entity=entity,
    job_name=job_name
))

trainer = ModelTrainer(
    variables.model,
    autoencoder_dataset,
    experiment_monitor,
    variables.criterion,
    variables.lr_scheduler_function,
    variables.lr_scheduler_configs,
    variables.validation_split,
    variables.batch_size,
    variables.learning_rate
)

trainer.train(variables.num_epochs)

trainer.save_model("/home/victor/pythonProjects/super-resolution/models_zoo/train_6_0.pth")
