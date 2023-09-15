import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout2d(p=0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Decoder
        x = self.relu(self.dropout(self.deconv1(x)))
        x = self.relu(self.dropout(self.deconv2(x)))
        x = self.relu(self.dropout(self.deconv3(x)))
        x = self.sigmoid(self.deconv4(x))
        return x
