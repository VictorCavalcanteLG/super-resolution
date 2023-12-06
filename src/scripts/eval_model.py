from src.bootstrap.bootstrap import Bootstrap
from src.infra.valuable_objects import MonitorExperimentsConfig
from src.infra.monitor.monitor import WandbExperimentMonitor
from numpy import asarray
from pathlib import Path
from PIL import Image
from src.metrics.image_metrics import (signal_to_noise_ratio, structural_similarity_index, mean_absolute_error,
                                       peak_signal_to_noise_ratio)

print("Evaluating Model")

CONFIG_FILE = '../../configs/config.yaml'

variables = Bootstrap(CONFIG_FILE)

experiment_monitor = WandbExperimentMonitor(MonitorExperimentsConfig(
    project_name="super-resolution",
    project_configs={
        "learning_rate": variables.learning_rate,
        # "architecture": "AutoEncoder",
        "epochs": variables.num_epochs,
    }
))

reference_image_dataset_path = Path("../../databases/DIV2K/DIV2K_valid_LR_480p")
output_image_dataset_path = Path("../../databases/DIV2K/DIV2K_valid_LR_OUTPUT_480p")

psnr_mean = 0.0
i = 0

for image_path in output_image_dataset_path.glob("*.png"):
    print("Evaluating image: " + image_path.name)

    output_image = asarray(Image.open(image_path))
    reference_image = asarray(Image.open(reference_image_dataset_path / image_path.name))

    SNR = signal_to_noise_ratio(reference_image, output_image)

    # SSIM = structural_similarity_index(reference_image, output_image, 241)

    MAE = mean_absolute_error(reference_image, output_image)

    psnr_score = peak_signal_to_noise_ratio(reference_image, output_image)

    psnr_mean = psnr_mean + psnr_score
    i = i + 1
    print(SNR, MAE, psnr_score)
    experiment_monitor.log_image_comparison_table(image_path.name, output_image, reference_image, psnr_score)

experiment_monitor.finalize()
print(psnr_mean/i)
