from PIL import Image
from pathlib import Path


def _resize_image(img_input_path, new_width, new_height):
    image = Image.open(img_input_path)
    new_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return new_image


class DatasetHelper:

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def create_resized_dataset_images(self, output_dataset_path, new_width, new_height):
        output_dataset_path = Path(output_dataset_path)
        output_dataset_path.mkdir(parents=True, exist_ok=True)

        for image_path in self.dataset_path.glob("*.png"):
            print("execute image: " + image_path.name)
            new_image = _resize_image(image_path, new_width, new_height)
            new_image.save(output_dataset_path / image_path.name)
