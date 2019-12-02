import os
from os.path import join, exists
import numpy as np
from tqdm import tqdm
import argparse
from scipy.ndimage.morphology import grey_dilation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision import transforms, utils, datasets

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


class Decoder(nn.Module):

    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 512, 4, 1, bias=False),
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
        x = z.view(-1, self.z_dim, 1, 1)
        output = self.main(x)
        return output


def get_data():
    def filter_background(x):
        x[:, (x < 0.3).any(dim=0)] = 0.0
        return x

    def dilate(x):
        x = x.squeeze(0).numpy()
        x = grey_dilation(x, size=3)
        x = x[None, :, :]
        return torch.from_numpy(x)

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        filter_background,
        lambda x: x.mean(dim=0)[None, :, :],
        dilate,
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dset = datasets.ImageFolder(args.train, transform=transform)
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=2)

    test_dset = datasets.ImageFolder(args.test, transform=transform)
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=2)

    start_dset = datasets.ImageFolder(args.start, transform=transform)
    goal_dset = datasets.ImageFolder(args.goal, transform=transform)

    start_images = torch.stack([start_dset[i][0] for i in range(len(start_dset))], dim=0)
    goal_images = torch.stack([goal_dset[i][0] for i in range(len(goal_dset))], dim=0)


    return train_loader, test_loader, start_images, goal_images


def train(model, optimizer, train_loader, encoder, epoch):
    model.train()

    train_losses = []
    pbar = tqdm(total=len(train_loader.dataset))
    for x, _ in train_loader:
        x = x.cuda()
        z = encoder(x).detach()
        recon = model(z)
        loss = F.mse_loss(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        avg_loss = np.mean(train_losses[-50:])

        pbar.set_description('Epoch {}, Train Loss {:.4f}'.format(epoch, avg_loss))
        pbar.update(x.shape[0])
    pbar.close()


def test(model, test_loader, encoder, epoch):
    model.eval()

    test_loss = 0
    for x, _ in test_loader:
        x = x.cuda()
        z = encoder(x).detach()
        recon = model(z)
        loss = F.mse_loss(recon, x)
        test_loss += loss.item() * x.shape[0]
    test_loss /= len(test_loader.dataset)
    print('Epoch {}, Test Loss: {:.4f}'.format(epoch, test_loss))


def save_recon(model, train_loader, test_loader, encoder, epoch, folder_name):
    model.eval()

    train_batch = next(iter(train_loader))[0][:16]
    test_batch = next(iter(test_loader))[0][:16]
    train_batch, test_batch = train_batch.cuda(), test_batch.cuda()

    with torch.no_grad():
        train_z, test_z = encoder(train_batch), encoder(test_batch)
        train_recon, test_recon = model(train_z), model(test_z)

    real_imgs = torch.cat((train_batch, test_batch), dim=0)
    recon_imgs = torch.cat((train_recon, test_recon), dim=0)
    imgs = torch.stack((real_imgs, recon_imgs), dim=1)
    imgs = imgs.view(-1, *real_imgs.shape[1:]).cpu()

    folder_name = join(folder_name, 'reconstructions')
    if not exists(folder_name):
        os.makedirs(folder_name)

    filename = join(folder_name, 'recon_epoch{}.png'.format(epoch))
    utils.save_image(imgs * 0.5 + 0.5, filename, nrow=8)


def save_interpolation(model, start_images, goal_images, encoder, epoch, folder_name):
    model.eval()

    z_start = encoder(start_images)  # n x z_dim
    z_goal = encoder(goal_images)  # n x z_dim

    start_size, goal_size = z_start.shape[0], z_goal.shape[0]

    z_start = z_start.repeat(goal_size, 1)
    z_start = z_start.view(start_size, goal_size, -1).permute(1, 0, 2)
    z_start = z_start.contiguous().view(-1, args.z_dim)
    z_goal = z_goal.repeat(start_size, 1)

    lambdas = np.linspace(0, 1, args.n_interp + 2)
    zs = torch.stack([(1 - lambda_) * z_start + lambda_ * z_goal
                      for lambda_ in lambdas], dim=1)  # n x n_interp+2 x z_dim
    zs = zs.view(-1, args.z_dim)  # n * (n_interp+2) x z_dim

    with torch.no_grad():
        imgs = model(zs).cpu()

    folder_name = join(folder_name, 'interpolations')
    if not exists(folder_name):
        os.makedirs(folder_name)

    filename = join(folder_name, 'interp_epoch{}.png'.format(epoch))
    utils.save_image(imgs * 0.5 + 0.5, filename, nrow=args.n_interp + 2)


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    assert exists(folder_name)

    train_loader, test_loader, start_images, goal_images = get_data()
    start_images, goal_images = start_images.cuda(), goal_images.cuda()

    model = Decoder(args.z_dim, 1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    encoder = torch.load(join(folder_name, 'encoder.pt'), map_location='cuda')
    encoder.eval()

    for epoch in range(args.epochs):
        train(model, optimizer, train_loader, encoder, epoch)
        test(model, test_loader, encoder, epoch)

        if epoch % args.log_interval == 0:
            save_recon(model, train_loader, test_loader, encoder, epoch, folder_name)
            save_interpolation(model, start_images, goal_images, encoder, epoch, folder_name)
            torch.save(model, join(folder_name, 'decoder.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/rope/train_data')
    parser.add_argument('--test', type=str, default='data/rope/test_data')
    parser.add_argument('--start', type=str, default='data/rope/seq_data/start')
    parser.add_argument('--goal', type=str, default='data/rope/seq_data/goal')
    parser.add_argument('--n_interp', type=int, default=6)

    parser.add_argument('--z_dim', type=int, default=16)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='recon')
    args = parser.parse_args()
    main()
