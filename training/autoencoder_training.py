import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.autoencoder import ConvAutoencoder
from data_loader.autoencoder_data_loader import AutoencoderDataset
from infra.monitor.monitor import WandbExperimentMonitor
from infra.valuable_objects import MonitorExperimentsConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.ToTensor()
])

x_train_dataset = "/home/victor/pythonProjects/super-resolution/datasets/DIV2K/DIV2K_train_LR_240p"
y_train_dataset = "/home/victor/pythonProjects/super-resolution/datasets/DIV2K/DIV2K_train_LR_480p"

autoencoder_dataset = AutoencoderDataset(x_train_dataset, y_train_dataset, transform)

train_loader = torch.utils.data.DataLoader(autoencoder_dataset, batch_size=10, shuffle=True)

torch.cuda.empty_cache()

experiment_monitor = WandbExperimentMonitor(MonitorExperimentsConfig(
    project_name="super-resolution",
    project_configs={
        "learning_rate": 0.001,
        "architecture": "AutoEncoder",
        "epochs": 5,
    }
))

experiment_monitor.watch_model(model, criterion=criterion)

num_epochs = 5
for epoch in range(num_epochs):
    print("Epoch: ", epoch)
    train_loss = 0.0
    for data in train_loader:
        print("teste")
        optimizer.zero_grad()

        x_img, y_img = data
        x_img = x_img.to(device)
        y_img = y_img.to(device)

        # Forward Pass
        output = model(x_img)

        # Compute Loss
        loss = criterion(output, y_img)

        # Backward Pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x_img.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    experiment_monitor.log_loss_and_accuracy(train_loss, 0.0)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

torch.save(model.state_dict(), "/home/victor/pythonProjects/super-resolution/models_zoo/train_4.pth")
