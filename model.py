import dill
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils import from_numpy_to_var


class FCN_mse(nn.Module):
    """
    Predict whether pixels are part of the object or the background.
    """

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        c1 = torch.tanh(self.conv1(x))
        c2 = torch.tanh(self.conv2(c1))
        score = (self.classifier(c2))  # size=(N, n_class, H, W)
        return score  # size=(N, n_class, x.H/1, x.W/1)


class D(nn.Module):
    """
    Predict whether image pairs are realistic.

    dtype 1:
        2*channel_dim x 64 x 64
        --> 64 x 32 x 32
        --> 128 x 16 x 16
        --> 256 x 8 x 8
        --> 512 x 4 x 4
        --> 1

        Img1, Img2 --> D --> 0/1

    """

    def __init__(self, dtype, channel_dim):
        super(D, self).__init__()
        self.dtype = dtype
        if dtype == 1:
            self.main = nn.Sequential(
                # input size (2 or 6) x 64 x64
                nn.Conv2d(2 * channel_dim, 64, 4, 2, 1),
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
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # 512 x 4 x 4
                nn.Conv2d(512, 1, 4),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError("dtype %d has not been implemented." % dtype)

    def forward(self, o, o_next):
        if self.dtype == 1:
            x = torch.cat([o, o_next], dim=1)
            return self.main(x)
        else:
            raise NotImplementedError("dtype %d has not been implemented." % self.dtype)


class GaussianPosterior(nn.Module):
    """
    Estimate the Gaussian posterior distribution on states.

    qtype 1:
        same as dtype 1
    """

    def __init__(self, con_c_dim, qtype, channel_dim):
        super(GaussianPosterior, self).__init__()
        self.conv = nn.Conv2d(512, 128, 4, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        # self.conv_disc = nn.Conv2d(128, dis_c_dim, 1)
        self.conv_mu = nn.Conv2d(128, con_c_dim, 1)
        self.conv_var = nn.Conv2d(128, con_c_dim, 1)
        self.con_c_dim = con_c_dim
        self.qtype = qtype

        if qtype == 1:
            self.main = nn.Sequential(
                # input size 3 x 64 x64
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
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # 512 x 4 x 4

                # Option 2: 256 x 8 x 8
                # nn.Conv2d(256, 1024, 8, bias=False),
                # nn.BatchNorm2d(1024),
                # nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            raise NotImplementedError("qtype %d has not been implemented." % qtype)

    def forward(self, x):
        x = self.main(x)
        y = self.lReLU(self.bn(self.conv(x)))
        mu = var = None
        if self.con_c_dim > 0:
            mu = self.conv_mu(y).squeeze()
            var = self.conv_var(y).squeeze().exp()
        return mu, var

    def forward_soft(self, x):
        mu, var = self.forward(x)
        return mu + var.sqrt() * from_numpy_to_var(np.random.randn(*list(var.size())))

    def forward_hard(self, x):
        mu, _ = self.forward(x)
        return mu.detach()

    def log_prob(self, x, s):
        """
        Compute log q(s|x)
        :param s: has size (bs, con_c_dim)
        """
        mu, var = self.forward(x)
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (s - mu).pow(2).div(var.mul(2.0) + 1e-6)
        return logli.sum(1)


class G(nn.Module):
    """
    Generate current and next observations from latent codes and noise.

    gtype 1:
        z_dim x 1 x 1
        --> 512 x 4 x 4
        --> 256 x 8 x 8
        --> 128 x 16 x 16
        --> 64 x 32 x 32
        --> 3 x 64 x 64

        z --G--> Img1, Img2
    """

    def __init__(self, c_dim, z_dim, gtype, channel_dim):
        super(G, self).__init__()
        self.latent_dim = z_dim + 2 * c_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.gtype = gtype
        self.channel_dim = channel_dim

        if gtype == 1:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(self.latent_dim, 512, 4, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 2 * channel_dim, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:
            raise NotImplementedError("gtype %d has not been implemented." % self.gtype)

    def forward(self, z, c, c_next):
        x = torch.cat([z, c, c_next], dim=1).view(-1, self.latent_dim, 1, 1)
        if self.gtype == 1:
            output = self.main(x)
            return output[:, :self.channel_dim, :, :], output[:, self.channel_dim:, :, :]
        else:
            raise NotImplementedError("gtype %d has not been implemented." % self.gtype)


class GaussianTransition(nn.Module):
    """
    Transition function in the latent space (Gaussian)
    """

    def __init__(self, s_dim=10, tau=0.1, hidden=(10, 10), learn_var=False, learn_mu=False):
        super(GaussianTransition, self).__init__()
        self.s_dim = s_dim
        self.output_dim = (learn_var + learn_mu) * s_dim
        if self.output_dim > 0:
            self.layers = [
                nn.Linear(s_dim, hidden[0]),
                nn.ReLU(True),
            ]
            for i, layer in enumerate(hidden):
                if i < len(hidden) - 1:
                    self.layers.append(nn.Linear(hidden[i], hidden[i + 1]))
                    self.layers.append(nn.ReLU(True))
            self.layers.append(nn.Linear(hidden[-1], self.output_dim))
            self.main = nn.Sequential(*self.layers)

        self.default_var = tau ** 2
        self.learn_var = learn_var
        self.learn_mu = learn_mu

    def get_mu_and_var(self, s):
        out = None
        if self.output_dim > 0:
            out = self.main(s)
        mu = s + F.tanh(out[:, :self.s_dim]) * 0.1 if self.learn_mu else s
        var = out[:, -self.s_dim:].exp() if self.learn_var else torch.ones_like(s)*self.default_var
        return mu, var

    def forward(self, s, a=None):
        bs = list(s.size())[0]
        mu, var = self.get_mu_and_var(s)
        return mu + var.sqrt() * from_numpy_to_var(np.random.randn(bs, self.s_dim))
        # return s + from_numpy_to_var(np.random.randn(bs, self.s_dim)*np.sqrt(self.var))

    def get_var(self, s):
        mu, var = self.get_mu_and_var(s)
        return var

    def log_prob(self, s, a, s_next):
        mu, var = self.get_mu_and_var(s)
        logli = -0.5 * torch.log((2 * np.pi * var) + 1e-6) - \
                (s_next - mu).pow(2).div(var * 2.0 + 1e-6)
        return logli.sum(1)


class UniformDistribution(nn.Module):
    def __init__(self, s_dim=10, unif_range=(-1, 1)):
        super(UniformDistribution, self).__init__()
        self.unif_range = unif_range
        self.s_dim = s_dim

    def sample(self, batch_size):
        s = np.random.uniform(*self.unif_range, size=(batch_size, self.s_dim))
        return from_numpy_to_var(s)

    def log_prob(self, s):
        bs = list(s.size())[0]
        log_prob = from_numpy_to_var(-np.ones(bs) * np.log(self.unif_range[1] - self.unif_range[0]))
        return log_prob


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Classifier(nn.Module):
    """
    Classifier is trained to predict the score between two color rope images.
    The score is high if they are within a few steps apart, and low other wise.
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.LeNet = nn.Sequential(
            # input size 6 x 64 x 64. Take 2 color images.
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            # Option 1: 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4),
            Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        stacked = torch.cat([x1, x2], dim=1)
        return self.LeNet(stacked)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_causal_classifier(path, default):
    """
    loads the weights saved in numpy (by ml_logger). This code is device independent, much better than the native
    state_dict objects. -- Ge
    """
    if not os.path.exists(path):
        return default
    classifier = Classifier().cuda()
    classifier.load_state_dict(torch.load(path))
    return classifier


class WGAN(nn.Module):

    def __init__(self, z_dim, channel_dim, c_dim=0, lambda_=10):
        super().__init__()
        self.lambda_ = lambda_
        self.G = SingleG(z_dim, channel_dim, c_dim=c_dim)
        self.D = SingleD(channel_dim)

    def sample(self, n, cond=None):
        samples = self.generate(n, cond=cond)
        samples = samples * 0.5 + 0.5
        return samples.cpu()

    def generate(self, n, cond=None):
        return self.G.sample(n, cond=cond)

    def discriminate(self, x):
        return self.D(x)

    def sample_eps(self, n):
        return torch.rand(n).cuda()

    def grad_penalty(self, x_hat):
        Dx_hat = self.discriminate(x_hat)
        grads = autograd.grad(Dx_hat, x_hat, torch.ones_like(Dx_hat),
                              retain_graph=True, create_graph=True, only_inputs=True)[0]
        grads = grads.view(grads.size(0), -1)
        grad_norms = torch.sqrt((grads ** 2).sum(-1))
        grad_penalty = self.lambda_ * (grad_norms - 1) ** 2
        return grad_penalty.mean()

    def gan_loss(self, x_tilde, x):
        Dx_tilde = self.discriminate(x_tilde)
        Dx = self.discriminate(x)
        return (Dx_tilde - Dx).mean()

    def generator_loss(self, gz):
        D_gz = self.discriminate(gz)
        return -D_gz.mean()


class GAN(nn.Module):

    def __init__(self, z_dim, channel_dim, c_dim=0):
        super().__init__()
        self.G = SingleG(z_dim, channel_dim, c_dim=c_dim)
        self.D = SingleD(channel_dim, sigmoid=True)

    def sample(self, n, cond=None):
        samples = self.generate(n, cond=cond)
        samples = samples * 0.5 + 0.5
        return samples.cpu()

    def generate(self, n, cond=None):
        return self.G.sample(n, cond=cond)

    def discriminate(self, x):
        return self.D(x)

    def gan_loss(self, x_tilde, x):
        Dx_tilde = self.discriminate(x_tilde)
        Dx = self.discriminate(x)
        pred = torch.cat((Dx_tilde, Dx), dim=0)
        labels = torch.cat((torch.zeros(Dx_tilde.shape[0]), torch.ones(Dx.shape[0])), dim=0).cuda()
        return F.binary_cross_entropy(pred, labels)

    def generator_loss(self, gz):
        D_gz = self.discriminate(gz)
        labels = torch.ones(D_gz.shape[0]).cuda()
        return F.binary_cross_entropy(D_gz, labels)

class SingleD(nn.Module):
    def __init__(self, channel_dim, sigmoid=False):
        super().__init__()
        self.model = nn.Sequential(
            # input size (1 or 3) x 64 x64
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
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4),
            # nn.Sigmoid()
        )
        self.sigmoid = sigmoid

    def forward(self, x):
        return torch.sigmoid(self.model(x)) if self.sigmoid else self.model(x)


class SingleG(nn.Module):
    def __init__(self, z_dim, channel_dim, c_dim=0):
        super(SingleG, self).__init__()
        self.latent_dim = z_dim + c_dim
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channel_dim, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)

    def sample(self, n, cond=None):
        z = torch.randn(n, self.z_dim).cuda()
        if cond is not None:
            z = torch.cat((z, cond), dim=-1)
        z = z.unsqueeze(-1).unsqueeze(-1)
        out = self(z)
        return out
