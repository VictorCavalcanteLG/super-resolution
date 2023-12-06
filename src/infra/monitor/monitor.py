import wandb
from src.infra.contracts import ExperimentMonitor
from src.infra.valuable_objects import MonitorExperimentsConfig


class WandbExperimentMonitor(ExperimentMonitor):

    def __init__(self, configs: MonitorExperimentsConfig):
        self.run = wandb.init(
            project=configs.project_name,
            config=configs.project_configs,
            entity=configs.entity,
            settings=wandb.Settings(job_name=configs.job_name)
        )
        self.log_table = wandb.Table(columns=["Image Name", "Output Image", "Ground Truth Image", "PSNR"])

    def log_loss(self, train_loss: float, validate_loss: float):
        self.run.log({
            "train_loss": train_loss,
            "validate_loss": validate_loss
        })

    def watch_model(self, model, criterion, log="all", log_freq=10):
        self.run.watch(model, criterion, log=log, log_freq=log_freq)

    def log_image_comparison_table(self, image_name, output_image, ground_truth_image, psnr_score):
        self.log_table.add_data(image_name,
                                wandb.Image(output_image, caption="Output Image"),
                                wandb.Image(ground_truth_image, caption="Ground Truth Image"),
                                psnr_score)

    def finalize(self):
        self.run.log({"Image Comparison Table": self.log_table})
        self.run.finish()
