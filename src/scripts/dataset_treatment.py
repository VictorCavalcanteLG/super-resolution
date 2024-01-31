from src.helpers.helpers import DatasetHelper

dataset_path = "/databases/DIV2K/DIV2K_valid_LR_480p"

dataset_helpers = DatasetHelper(dataset_path)

print("Treating current dataset")

dataset_helpers.create_resized_dataset_images("/home/victor/pythonProjects/super-resolution/databases/DIV2K/DIV2K_valid_LR_240p", 360, 240)
