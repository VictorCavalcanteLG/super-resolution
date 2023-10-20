import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from src.infra.contracts import ExperimentMonitor


class ModelTrainer:

    def __init__(self, model_class: nn.Module, dataset: Dataset, experiment_monitor: ExperimentMonitor, criterion_class,
                 validation_split=0.2, batch_size=1, learning_rate=0.001):
        """
        Initializes the class.

        Args:
            model_class: Model class.
            dataset: Instance of Dataset.
            experiment_monitor: Instance of ExperimentMonitor.
            criterion_class: Criterion class (loss function).
            batch_size: Batch size to train the model.
            validation_split: Database division percentage for validation.
            learning_rate: Learning rate (default is 0.001).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            print("CUDA (GPU) not available. Switching to CPU.")

        torch.cuda.empty_cache()

        self.model = model_class().to(self.device)
        self.criterion = criterion_class()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        dataset_size = len(dataset)
        validation_size = int(validation_split * dataset_size)
        train_size = dataset_size - validation_size

        train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.experiment_monitor = experiment_monitor

    def validate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in self.val_loader:
                x_img, y_img = data
                x_img = x_img.to(self.device)
                y_img = y_img.to(self.device)

                # Forward Pass
                outputs = self.model(x_img)

                # Compute Loss
                loss = self.criterion(outputs, y_img)
                val_loss += loss.item() * x_img.size(0)

        val_loss = val_loss / len(self.val_loader.dataset)

        return val_loss

    def train(self, num_epochs):
        self.experiment_monitor.watch_model(self.model, criterion=self.criterion)

        for epoch in range(num_epochs):
            self.model.train()

            print("Epoch: ", epoch)
            train_loss = 0.0
            for data in self.train_loader:
                print("teste")
                self.optimizer.zero_grad()

                x_img, y_img = data
                x_img = x_img.to(self.device)
                y_img = y_img.to(self.device)

                # Forward Pass
                output = self.model(x_img)

                # Compute Loss
                loss = self.criterion(output, y_img)

                # Backward Pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * x_img.size(0)

            train_loss = train_loss / len(self.train_loader.dataset)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

            val_loss = self.validate()
            print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch + 1, val_loss))

            self.experiment_monitor.log_loss(train_loss, val_loss)

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
