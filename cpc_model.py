import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    prefix = 'encoder'

    def __init__(self, z_dim, channel_dim, squash=False):
        super().__init__()
        print('squash', squash)

        self.squash = squash
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1),
        #    nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1),
        #    nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Option 1: 256 x 8 x 8
            nn.Conv2d(256, 256, 4, 2, 1),
        #    nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
        )
        self.out = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        if self.squash:
            x = torch.tanh(x)
        return x


class Transition(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, squash=False, trans_type='linear'):
        super().__init__()
        self.squash = squash
        self.trans_type = trans_type
        print('squash', self.squash, 'trans_type', self.trans_type)
        self.z_dim = z_dim

        if self.trans_type == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        elif self.trans_type == 'mlp':
            hidden_size = 32
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, z_dim)
            )
        else:
            raise Exception('Invalid trans_type', trans_type)

    def forward(self, x):
        x = self.model(x)
        if self.squash:
            x = torch.tanh(x)
        return x


def quantize(x, n_bit):
    x = x * 0.5 + 0.5 # to [0, 1]
    x *= n_bit ** 2 - 1 # [0, 15] for n_bit = 4
    x = torch.floor(x + 1e-4) # [0, 15]
    return x


class Decoder(nn.Module):
    prefix = 'decoder'

    def __init__(self, z_dim, channel_dim, discrete=False, n_bit=4):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim
        self.discrete = discrete
        self.n_bit = n_bit
        self.discrete_dim = 2 ** n_bit

        out_dim = self.discrete_dim * self.channel_dim if discrete else channel_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 256, 4, 1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, out_dim, 4, 2, 1),
        )

    def forward(self, z):
        x = z.view(-1, self.z_dim, 1, 1)
        output = self.main(x)

        if self.discrete:
            output = output.view(output.shape[0], self.discrete_dim,
                                 self.channel_dim, *output.shape[2:])
        else:
            output = torch.tanh(output)

        return output


    def loss(self, x, z):
        recon = self(z)
        if self.discrete:
            loss = F.cross_entropy(recon, quantize(x, self.n_bit).long())
        else:
            loss = F.mse_loss(recon, x)
        return loss


    def predict(self, z):
        recon = self(z)
        if self.discrete:
            recon = torch.max(recon, dim=1)[1].float()
            recon = (recon / (self.discrete_dim - 1) - 0.5) / 0.5
        return recon


class InverseModel(nn.Module):
    prefix = 'inv'

    def __init__(self, z_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(2 * z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z, z_next):
        x = torch.cat((z, z_next), dim=1)
        return self.model(x)


class ForwardModel(nn.Module):
    prefix = 'forward'

    def __init__(self, z_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(z_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
        )

    def forward(self, z, action):
        x = torch.cat((z, action), dim=1)
        return self.model(x)


class BetaVAE(nn.Module):
    def __init__(self, z_dim, channel_dim, beta=1.0):
        super().__init__()
        self.encoder = Encoder(2 * z_dim, channel_dim)
        self.decoder = Decoder(z_dim, channel_dim)
        self.beta = beta

    def encode(self, x):
        return self.encoder(x).chunk(2, dim=1)[0] # Only get mu

    def decode(self, z):
        return self.decoder(z)

    def loss(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = torch.randn_like(mu) * (0.5 * log_var).exp() + mu
        recon = self.decoder(z)

        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_loss = 0.5 * torch.sum(-log_var - 1 + torch.exp(log_var) + mu ** 2)
        return recon_loss + self.beta * kl_loss, recon_loss.item(), kl_loss.item()
