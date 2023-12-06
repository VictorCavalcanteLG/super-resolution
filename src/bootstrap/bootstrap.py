import yaml
from src.bootstrap.valuable_objects import LOSS, MODEL, LEARNING_RATE_SCHEDULER


class Bootstrap:

    def __init__(self, config_path):
        self.lr_scheduler_configs = None
        self.lr_scheduler_function = None
        self.validation_split = None
        self.batch_size = None
        self.num_epochs = None
        self.criterion = None
        self.learning_rate = None
        self.model = None
        self.x_train_dataset_path = None
        self.y_train_dataset_path = None
        self.x_test_dataset_path = None
        self.y_test_dataset_path = None

        with open(config_path, 'r') as file:
            self.__data = yaml.safe_load(file)

        self.set_train_variables()

    def set_train_variables(self):
        self.x_train_dataset_path = self.__data['x_train_dataset_path']
        self.y_train_dataset_path = self.__data['y_train_dataset_path']
        self.x_test_dataset_path = self.__data['x_test_dataset_path']
        self.y_test_dataset_path = self.__data['y_test_dataset_path']

        model_configs = self.__data['model_configs']

        self.model = MODEL[model_configs['model']]
        self.learning_rate = model_configs['learning_rate']
        self.criterion = LOSS[model_configs['criterion']]
        self.batch_size = model_configs['batch_size']
        self.validation_split = model_configs['validation_split']
        self.num_epochs = model_configs['num_epochs']

        lr_scheduler = self.__data['learning_rate_scheduler']

        self.lr_scheduler_function = LEARNING_RATE_SCHEDULER[lr_scheduler['function']]
        self.lr_scheduler_configs = lr_scheduler['configs']
