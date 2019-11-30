import argparse
import os
from os.path import join, exists
import numpy as np
from scipy.ndimage.morphology import grey_dilation
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataset import CPCDataset


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


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def get_dataloader():
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

    train_dset = CPCDataset(root=args.root, transform=transform,
                            n_frames_apart=args.k, n=args.n, n_repeat=args.n_repeat)
    print('Dset Size', len(train_dset))
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=2)

    other_dset = ImageFolder(args.root, transform=transform)
    other_loader = data.DataLoader(other_dset, batch_size=args.batch_size, shuffle=True,
                                   pin_memory=True, num_workers=2)

    start_dset = ImageFolder(args.start, transform=transform)
    goal_dset = ImageFolder(args.goal, transform=transform)

    start_images = torch.stack([start_dset[i][0] for i in range(len(start_dset))], dim=0)
    goal_images = torch.stack([goal_dset[i][0] for i in range(len(goal_dset))], dim=0)


    return infinite_dataloader(train_loader), other_loader, start_images, goal_images


def find_closest_image(z, other_loader, encoder):
    with torch.no_grad():
        dists = []
        for other_x, _ in other_loader:
            other_x = other_x.cuda()
            other_z = encoder(other_x) # bs x z_dim
            dists.append(torch.matmul(z, other_z.t()).cpu())
        dists = torch.cat(dists, dim=1)
        closest = torch.argmin(dists, dim=1).numpy()

        closest_images = torch.stack([other_loader.dataset[i][0] for i in closest], dim=0)
        return closest_images * 0.5 + 0.5



def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    obs_dim = (1, 64, 64)
    encoder = Encoder(args.z_dim, 1).cuda()
    trans = Transition(args.z_dim).cuda()

    optimizer = optim.Adam(list(encoder.parameters()) + list(trans.parameters()),
                           lr=args.lr)

    dataloader, other_loader, start_images, goal_images = get_dataloader()
    start_images, goal_images = start_images.cuda(), goal_images.cuda()
    losses = []

    x, y = next(dataloader)
    x, y = x[:10], y[:10]
    x = x.view(-1, *obs_dim)
    save_image(x * 0.5 + 0.5, join(folder_name, 'train_img.png'), nrow=args.n + 1)

    pbar = tqdm(total=args.itrs)
    for i in range(args.itrs):
        x, y = next(dataloader) # bs x n+1 x 1 x 64 x 64
        x, y = x.cuda(), y.cuda()
        bs = x.shape[0]

        x = x.view(-1, *obs_dim) # bs * (n + 1) x 1 x 64 x 64
        z = encoder(x) # bs * (n + 1) x z_dim
        z = z.view(bs, args.n + 1, args.z_dim)  # bs x n+1 x z_dim
        z_cur, z_other = z[:, 0, :], z[:, 1:, :] # bs x z_dim, bs x n x z_dim
        z_next = trans(z_cur).unsqueeze(1) # bs x 1 x z_dim
        z_other = z_other.permute(0, 2, 1).contiguous() # bs x z_dim x n
        logits = torch.bmm(z_next, z_other).squeeze(1) # bs x n
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        avg_loss = np.mean(losses[-50:])
        pbar.set_description('Loss: {:.4f}'.format(avg_loss))
        pbar.update(1)

        if i % args.log_interval == 0:
            torch.save(encoder, join(folder_name, 'encoder.pt'))
            torch.save(trans, join(folder_name, 'trans.pt'))

            z_start = encoder(start_images) # n x z_dim
            z_goal = encoder(goal_images) # n x z_dim

            start_size, goal_size = z_start.shape[0], z_goal.shape[0]

            z_start = z_start.repeat(goal_size, 1)
            z_start = z_start.view(start_size, goal_size, -1).permute(1, 0, 2)
            z_start = z_start.contiguous().view(-1, args.z_dim)
            z_goal = z_goal.repeat(start_size, 1)

            lambdas = np.linspace(0, 1, args.n_interp + 2)
            zs = torch.stack([(1 - lambda_) * z_start + lambda_ * z_goal
                              for lambda_ in lambdas], dim=1) # n x n_interp+2 x z_dim
            zs = zs.view(-1, args.z_dim) # n * (n_interp+2) x z_dim
            imgs = find_closest_image(zs, other_loader, encoder)
            save_image(imgs, join(folder_name, 'sample_itr{}.png'.format(i)), nrow=args.n_interp + 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope/full_data')
    parser.add_argument('--start', type=str, default='data/rope/seq_data/start')
    parser.add_argument('--goal', type=str, default='data/rope/seq_data/goal')
    parser.add_argument('--n_interp', type=int, default=6)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--itrs', type=int, default=int(1e5))
    parser.add_argument('--log_interval', type=int, default=500)

    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--n_repeat', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--k', type=int, default=1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='cpc')
    args = parser.parse_args()
    main()
