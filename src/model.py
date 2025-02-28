import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_channels: int = 1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1, bias=False),
            nn.Conv2d(img_channels, img_channels, 5, 1, 2, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(batch_size, self.latent_dim, 1, 1)
        img = self.model(z)
        return img[:, :, 2:-2, 2:-2]

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, img):
        logits = self.model(img)
        return logits.view(-1, 1)
