import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, z_dim, channel_dim):
        super().__init__()

        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Option 1: 256 x 8 x 8
            nn.Conv2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
        )
        self.out = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)


class Transition(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.out = nn.Linear(z_dim, z_dim, bias=False)

    def forward(self, x):
        return self.out(x)


class Decoder(nn.Module):

    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, channel_dim, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = z.view(-1, self.z_dim, 1, 1)
        output = self.main(x)
        return output