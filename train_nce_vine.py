import argparse
import os
from os.path import join, exists
import numpy as np
from scipy.ndimage.morphology import grey_dilation
from tqdm import tqdm
import glob

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from dataset import NCEVineDataset
from cpc_model import Encoder, Decoder, Transition
from cpc_util import *


def get_dataloaders():
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

    train_dset = NCEVineDataset(root=join(args.root, 'train_data'), n_neg=args.n,
                                transform=transform)
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dset = NCEVineDataset(root=join(args.root, 'test_data'), n_neg=args.n, transform=transform)
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    dec_train_dset = ImageFolder(join(args.root, 'train_data'), transform=transform)
    dec_train_loader = data.DataLoader(dec_train_dset, batch_size=args.batch_size, shuffle=True,
                                       pin_memory=True, num_workers=2) # for training decoder

    dec_test_dset = ImageFolder(join(args.root, 'test_data'), transform=transform)
    dec_test_loader = data.DataLoader(dec_test_dset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=True, num_workers=2)

    return train_loader, test_loader, dec_train_loader, dec_test_loader


def compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions):
    bs = obs.shape[0]

    z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    # obs_neg is b x n x 1 x 64 x 64
    obs_neg = obs_neg.view(-1, *obs_neg.shape[2:]) # b * n x 1 x 64 x 64
    z_neg = encoder(obs_neg)  # b * n x z_dim

    z = torch.cat((z, actions), dim=1)
    z_next = trans(z)  # b x z_dim

    pos_log_density = (z_next * z_pos).sum(dim=1)
    if args.mode == 'cos':
        pos_log_density /= torch.norm(z_next, dim=1) * torch.norm(z_pos, dim=1)
    pos_log_density = pos_log_density.unsqueeze(1)

    z_next = z_next.unsqueeze(1)
    z_neg = z_neg.view(bs, args.n, args.z_dim).permute(0, 2, 1).contiguous() # b x z_dim x n
    neg_log_density = torch.bmm(z_next, z_neg).squeeze(1)  # b x n
    if args.mode == 'cos':
        neg_log_density /= torch.norm(z_next, dim=2) * torch.norm(z_neg, dim=1)

    loss = torch.cat((torch.zeros(bs, 1).cuda(), neg_log_density - pos_log_density), dim=1)  # b x n+1
    loss = torch.logsumexp(loss, dim=1).mean()
    return loss


def train_cpc(encoder, trans, optimizer, train_loader, epoch):
    encoder.train()
    trans.train()

    train_losses = []
    pbar = tqdm(total=len(train_loader.dataset))
    for batch in train_loader:
        obs, obs_pos, actions, obs_neg = [b.cuda() for b in batch]
        loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        avg_loss = np.mean(train_losses[-50:])

        pbar.set_description('CPC Epoch {}, Train Loss {:.4f}'.format(epoch, avg_loss))
        pbar.update(obs.shape[0])
    pbar.close()


def test_cpc(encoder, trans, test_loader, epoch):
    encoder.eval()
    trans.eval()

    test_loss = 0
    for batch in test_loader:
        obs, obs_pos, actions, obs_neg = [b.cuda() for b in batch]
        loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=actions)
        test_loss += loss.item() * obs.shape[0]
    test_loss /= len(test_loader.dataset)
    print('CPC Epoch {}, Test Loss {:.4f}'.format(epoch, test_loss))


def train_decoder(decoder, optimizer, train_loader, encoder, epoch):
    decoder.train()
    encoder.eval()

    train_losses = []
    pbar = tqdm(total=len(train_loader.dataset))
    for x, _ in train_loader:
        x = x.cuda()

        z = encoder(x).detach()
        recon = decoder(z)
        loss = F.mse_loss(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        avg_loss = np.mean(train_losses[-50:])

        pbar.set_description('Dec Epoch {}, Train Loss {:.4f}'.format(epoch, avg_loss))
        pbar.update(x.shape[0])
    pbar.close()


def test_decoder(decoder, test_loader, encoder, epoch):
    decoder.eval()
    encoder.eval()

    test_loss = 0
    for x, _ in test_loader:
        x = apply_fcn_mse(x) if args.thanard_dset else x.cuda()
        z = encoder(x).detach()
        recon = decoder(z)
        loss = F.mse_loss(recon, x)
        test_loss += loss.item() * x.shape[0]
    test_loss /= len(test_loader.dataset)
    print('Dec Epoch {}, Test Loss: {:.4f}'.format(epoch, test_loss))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    obs_dim = (1, 64, 64)
    action_dim = 4

    encoder = Encoder(args.z_dim, obs_dim[0]).cuda()
    trans = Transition(args.z_dim, action_dim).cuda()
    decoder = Decoder(args.z_dim, obs_dim[0]).cuda()

    optim_cpc = optim.Adam(list(encoder.parameters()) + list(trans.parameters()),
                           lr=args.lr)
    optim_dec = optim.Adam(decoder.parameters(), lr=args.lr)

    train_loader, test_loader, dec_train_loader, dec_test_loader = get_dataloaders()

    # Save training images
    imgs = next(iter(dec_train_loader))[0][:64]
    save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_img.png'), nrow=8)

    batch = next(iter(train_loader))
    obs, obs_next, _, obs_neg = batch
    imgs = torch.stack((obs, obs_next), dim=1).view(-1, *obs.shape[1:])
    save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_seq_img.png'), nrow=8)

    obs_neg = obs_neg.view(-1, *obs_dim)[:100]
    save_image(obs_neg * 0.5 + 0.5, join(folder_name, 'neg.png'), nrow=10)

    for epoch in range(args.epochs):
        train_cpc(encoder, trans, optim_cpc, train_loader, epoch)
        test_cpc(encoder, trans, test_loader, epoch)

        if epoch % args.log_interval == 0:
            train_decoder(decoder, optim_dec, dec_train_loader, encoder, epoch)
            test_decoder(decoder, dec_test_loader, encoder, epoch)

            save_recon(decoder, dec_train_loader, dec_test_loader, encoder,
                       epoch, folder_name)
            start_images, goal_images = next(iter(dec_train_loader))[:20].cuda().chunk(2, dim=0)
            save_interpolation(args.n_interp, decoder, start_images,
                               goal_images, encoder, epoch, folder_name)
            save_run_dynamics(decoder, encoder, trans,
                              dec_train_loader, epoch, folder_name, args.root,
                              include_actions=True)
            save_nearest_neighbors(encoder, dec_train_loader, dec_test_loader,
                                   epoch, folder_name, metric='dotproduct')

            torch.save(encoder, join(folder_name, 'encoder.pt'))
            torch.save(trans, join(folder_name, 'trans.pt'))
            torch.save(trans, join(folder_name, 'decoder.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope2')
    parser.add_argument('--n_interp', type=int, default=8)
    parser.add_argument('--mode', type=str, default='dotproduct')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)

    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--z_dim', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='cpc')
    args = parser.parse_args()

    assert args.mode in ['dotproduct', 'cos']

    main()
