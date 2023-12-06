from torch.utils.data import Dataset
import random
from PIL import Image
import glob
from pathlib import Path


class AutoencoderDataset(Dataset):
    def __init__(self, input_image_folder_dataset, output_image_folder_dataset, transforms=None):
        self.input_image_folder_dataset = glob.glob(input_image_folder_dataset + "/*.png")
        random.shuffle(self.input_image_folder_dataset)
        self.output_image_folder_dataset = Path(output_image_folder_dataset)
        self.transforms = transforms
        self.current_index = 0

    def __getitem__(self, index):
        x_img_path = self.input_image_folder_dataset[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.input_image_folder_dataset)
        x_img = Image.open(x_img_path)
        x_img = x_img.convert("RGB")
        actual_path = Path(x_img_path)

        y_img = Image.open(self.output_image_folder_dataset / actual_path.name)
        y_img = y_img.convert("RGB")

        if self.transforms is not None:
            x_img = self.transforms(x_img)
            y_img = self.transforms(y_img)

        return x_img, y_img

    def __len__(self):
        return len(self.input_image_folder_dataset)
