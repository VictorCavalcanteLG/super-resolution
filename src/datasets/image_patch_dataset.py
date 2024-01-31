from torch.utils.data import IterableDataset
import random
from PIL import Image
import glob
from pathlib import Path
import numpy as np


class ImagesPatchDataset(IterableDataset):
    def __init__(self, input_image_folder_dataset, output_image_folder_dataset, window_size, stride, transforms=None):
        self.input_image_folder_dataset = glob.glob(input_image_folder_dataset + "/*.png")
        random.shuffle(self.input_image_folder_dataset)
        self.output_image_folder_dataset = Path(output_image_folder_dataset)
        self.transforms = transforms
        self.current_index = 0
        self.window_size = window_size
        self.stride = stride

    def generator(self):
        while self.current_index < len(self.input_image_folder_dataset):
            x_img_path = self.input_image_folder_dataset[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.input_image_folder_dataset)
            x_img = Image.open(x_img_path)
            x_img = x_img.convert("RGB")
            x_img_np = np.array(x_img)

            actual_path = Path(x_img_path)
            y_img = Image.open(self.output_image_folder_dataset / actual_path.name)
            y_img = y_img.convert("RGB")
            y_img_np = np.array(y_img)

            augmentation_factor = int(y_img.size[0] / x_img.size[0])
            window_location = (0, 0)

            x_img_window = None
            y_img_window = None

            while window_location[0] < x_img.size[1]:  # In numpy the array is [y,x]
                while window_location[1] < x_img.size[0]:
                    print(window_location)
                    x_img_window = x_img_np[window_location[0]: window_location[0] + self.window_size,
                                   window_location[1]: window_location[1] + self.window_size]
                    y_img_window = y_img_np[window_location[0] * augmentation_factor: window_location[
                                                                                          0] * augmentation_factor + self.window_size * augmentation_factor,
                                   window_location[1] * augmentation_factor: window_location[
                                                                                 1] * augmentation_factor + self.window_size * augmentation_factor]

                    window_location = (window_location[0], window_location[1] + self.stride)

                    x_img_tensor = None
                    y_img_tensor = None

                    if self.transforms is not None:
                        x_img_tensor = self.transforms(x_img_window)
                        y_img_tensor = self.transforms(y_img_window)

                    yield x_img_tensor, y_img_tensor

                window_location = (window_location[0] + self.stride, 0)

    def __iter__(self):
        return self.generator()
