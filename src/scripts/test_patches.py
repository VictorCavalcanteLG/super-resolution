from src.bootstrap.bootstrap import Bootstrap
from src.datasets.image_patch_dataset import ImagesPatchDataset
from numpy import asarray
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

print("Evaluating Model")

CONFIG_FILE = '../../configs/config.yaml'

variables = Bootstrap(CONFIG_FILE)

transform = transforms.Compose([
    transforms.ToTensor()
])

autoencoder_dataset = ImagesPatchDataset(variables.x_train_dataset_path, variables.y_train_dataset_path, 120, 120, transform)


print("Testing image patch")

train_loader = DataLoader(autoencoder_dataset)

for data in train_loader:
    x_tensor, y_tensor = data

    x_img = transforms.ToPILImage()(x_tensor.squeeze(0))
    y_img = transforms.ToPILImage()(y_tensor.squeeze(0))

    plt.imshow(x_img)
    plt.title("Input Image")
    plt.show()

    plt.imshow(y_img)
    plt.title("Output Image")
    plt.show()
