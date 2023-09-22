import wandb
from src.infra.contracts import ExperimentMonitor
from src.infra.valuable_objects import MonitorExperimentsConfig


class WandbExperimentMonitor(ExperimentMonitor):

    def __init__(self, configs: MonitorExperimentsConfig):
        self.run = wandb.init(
            project=configs.project_name,
            config=configs.project_configs
        )

    def log_loss_and_accuracy(self, loss: float, acc: float):
        self.run.log({
            "loss": loss,
            "acc": acc
        })

    def watch_model(self, model, criterion, log="all", log_freq=10):
        self.run.watch(model, criterion, log=log, log_freq=log_freq)
