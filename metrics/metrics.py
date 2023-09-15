import torch
from torchvision import transforms
from models.autoencoder import ConvAutoencoder
import glob
from PIL import Image
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
model_weights_path = "/home/victor/pythonProjects/super-resolution/models_zoo/train_4.pth"
model.load_state_dict(torch.load(model_weights_path))
model.eval()

imgs_path = "/home/victor/pythonProjects/super-resolution/datasets/DIV2K/DIV2K_valid_LR_240p"
imgs_path_dataset = glob.glob(imgs_path + "/*.png")[:100]

output_path = Path("/home/victor/pythonProjects/super-resolution/datasets/DIV2K/DIV2K_valid_output_LR")
output_path.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor()
])

x_img = None
for image_path in imgs_path_dataset:
    actual_path = Path(image_path)

    x_img = Image.open(image_path)
    x_img = transform(x_img)
    x_img = x_img.to(device)

    output = model(x_img)

    output_img = transforms.ToPILImage()(output[0].cpu())

    output_img.save(output_path / actual_path.name)

