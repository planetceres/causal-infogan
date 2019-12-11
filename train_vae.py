from tqdm import tqdm
import argparse
from os.path import join, exists
import numpy as np
import os

import torch
import torch.optim as optim
import torch.utils.data as data

from torchvision.utils import save_image
from torchvision import datasets

from cpc_model import BetaVAE


def get_dataloaders():
    train_dset = datasets.ImageFolder(join(args.root, 'train_data'))
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True,
                                   num_workers=4)

    test_dset = datasets.ImageFolder(join(args.root, 'test_data'))
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=True,
                                  num_workers=4)

    return train_loader, test_loader


def train(model, optimizer, train_loader, epoch, device):
    model.train()

    recon_losses, kl_losses = [], []
    pbar = tqdm(total=len(train_loader.dataset))
    for x, _ in train_loader:
        x = x.to(device)
        loss, recon_loss, kl_loss = model.loss(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_losses.append(recon_loss)
        kl_losses.append(kl_loss)
        avg_recon_loss = np.mean(recon_losses[-50:])
        avg_kl_loss = np.mean(kl_losses[-50:])

        pbar.set_description('Epoch {}, Recon Loss {:.4f}, KL Loss {:.4f}'.format(epoch, avg_recon_loss, avg_kl_loss))
        pbar.update(x.shape[0])
    pbar.close()


def test(model, test_loader, epoch, device):
    model.eval()

    test_recon_loss, test_kl_loss = 0, 0
    for x, _ in test_loader:
        with torch.no_grad():
            x = x.to(device)
            _, recon_loss, kl_loss = model.loss(x)
            test_recon_loss += recon_loss * x.shape[0]
            test_kl_loss += kl_loss * x.shape[0]
    test_recon_loss /= len(test_loader.dataset)
    test_kl_loss /= len(test_loader.dataset)
    print('Epoch {}, Test Recon Loss: {:.4f}, KL Loss: {:.4f}'.format(epoch, test_recon_loss, test_kl_loss))


def save_recon(model, train_loader, test_loader, epoch, folder_name, device, n=32):
    assert n % 2 == 0
    x_train = next(iter(train_loader))[0][:n // 2]
    x_test = next(iter(test_loader))[0][:n // 2]
    x = torch.cat((x_train, x_test), dim=0).to(device)

    with torch.no_grad():
        z = model.encode(x)
        recon_x = model.decode(z)
    imgs = torch.stack((x, recon_x), dim=1).view(-1, *x.shape[1:])
    imgs = imgs.cpu() * 0.5 + 0.5

    folder_name = join(folder_name, 'reconstructions')
    if not exists(folder_name):
        os.makedirs(folder_name)
    save_image(imgs, join(folder_name, 'recon_e{}.png'.format(epoch)))


def save_interpolation(model, train_loader, test_loader, epoch, folder_name, device, n=10):
    assert n % 2 == 0
    x_train = next(iter(train_loader))[:n]
    x_test = next(iter(test_loader))[:n]

    x_train_start, x_train_end = x_train.chunk(2, dim=0)
    x_test_start, x_test_end = x_test.chunk(2, dim=0)

    x_start = torch.cat((x_train_start, x_test_start), dim=0).to(device)
    x_end = torch.cat((x_test_start, x_test_end), dim=0).to(device)

    with torch.no_grad():
        z_start = model.encode(x_start)
        z_end = model.encode(x_end)

    alphas = np.linspace(0, 1, args.n_interp)
    zs = torch.stack([(1 - alpha) * z_start + alpha * z_end
                      for alpha in alphas], dim=1)
    zs = zs.view(-1, args.z_dim)
    with torch.no_grad():
        imgs = model.decode(zs)
    imgs = imgs.cpu() * 0.5 + 9.5

    folder_name = join(folder_name, 'interpolations')
    if not exists(folder_name):
        os.makedirs(folder_name)
    save_image(imgs, join(folder_name, 'interp_e{}.png'.format(epoch)))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    assert exists(folder_name)

    device = torch.device('cuda')
    train_loader, test_loader = get_dataloaders()

    model = BetaVAE(args.z_dim, 1, beta=args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(model, optimizer, train_loader, epoch, device)
        test(model, test_loader, epoch, device)

        save_recon(model, train_loader, test_loader, epoch, folder_name, device)
        save_interpolation(model, train_loader, test_loader, epoch, folder_name, device)
        torch.save(model, join(folder_name, 'vae.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope')
    parser.add_argument('--n_interp', type=int, default=10)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--beta', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='recon')
    args = parser.parse_args()

    main()
