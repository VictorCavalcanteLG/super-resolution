from torchvision import transforms
from src.bootstrap.bootstrap import Bootstrap
from PIL import Image
from pathlib import Path
import torch
import os

print("Generating images with high resolution")

CONFIG_FILE = os.getenv('MODEL_CONFIG_PATH')

variables = Bootstrap(CONFIG_FILE)

dataset_path = Path(variables.x_test_dataset_path)

model = variables.model()
model.load_state_dict(torch.load(variables.model_zoo))
model.eval()

output_dataset_path = Path(variables.output_dataset_path)
output_dataset_path.mkdir(parents=True, exist_ok=True)

for image_path in dataset_path.glob("*.png"):
    print("execute image: " + image_path.name)
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    output = model(image)
    output_image = transforms.ToPILImage()(output)
    output_image.save(output_dataset_path / image_path.name)

