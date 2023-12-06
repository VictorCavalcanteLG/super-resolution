from torchvision import transforms
from src.bootstrap.bootstrap import Bootstrap
from PIL import Image
from pathlib import Path
import torch

print("Generating images with high resolution")

CONFIG_FILE = '../../configs/config.yaml'

dataset_path = Path("../../databases/DIV2K/DIV2K_valid_LR_240p")

variables = Bootstrap(CONFIG_FILE)

model = variables.model()
model.load_state_dict(torch.load("../../models_zoo/train_4.pth"))
model.eval()

output_dataset_path = Path("../../databases/DIV2K/DIV2K_valid_LR_OUTPUT_480p")
output_dataset_path.mkdir(parents=True, exist_ok=True)

for image_path in dataset_path.glob("*.png"):
    print("execute image: " + image_path.name)
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    output = model(image)
    output_image = transforms.ToPILImage()(output)
    output_image.save(output_dataset_path / image_path.name)

