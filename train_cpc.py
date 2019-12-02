import argparse
import os
from os.path import join, exists
import numpy as np
from scipy.ndimage.morphology import grey_dilation
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataset import ImagePairs
from cpc_model import Encoder, Decoder, Transition


def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


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

    train_dset = ImagePairs(root=args.train, transform=transform, n_frames_apart=args.k)
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dset = ImagePairs(root=args.test, transform=transform, n_frames_apart=args.k)
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    neg_train_dset = ImageFolder(args.train, transform=transform)
    neg_train_loader = data.DataLoader(neg_train_dset, batch_size=args.batch_size, shuffle=True,
                                       pin_memory=True, num_workers=2) # for training decoder
    neg_train_inf = infinite_loader(data.DataLoader(neg_train_dset, batch_size=args.n, shuffle=True,
                                                    pin_memory=True, num_workers=2)) # to get negative samples

    neg_test_dset = ImageFolder(args.train, transform=transform)
    neg_test_loader = data.DataLoader(neg_test_dset, batch_size=args.batch_size, shuffle=True,
                                       pin_memory=True, num_workers=2)
    neg_test_inf = infinite_loader(data.DataLoader(neg_test_dset, batch_size=args.n, shuffle=True,
                                                   pin_memory=True, num_workers=2))


    start_dset = ImageFolder(args.start, transform=transform)
    goal_dset = ImageFolder(args.goal, transform=transform)

    start_images = torch.stack([start_dset[i][0] for i in range(len(start_dset))], dim=0)
    goal_images = torch.stack([goal_dset[i][0] for i in range(len(goal_dset))], dim=0)


    return train_loader, test_loader, neg_train_loader, neg_test_loader, neg_train_inf, neg_test_inf, start_images, goal_images


def compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans):
    bs = obs.shape[0]

    z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    z_neg = encoder(obs_neg)  # n x z_dim
    z_next = trans(z)  # b x z_dim

    z_next = z_next.unsqueeze(1)  # b x 1 x z_dim
    z_pos = z_pos.unsqueeze(2)  # b x z_dim x 1
    pos_log_density = torch.bmm(z_next, z_pos).squeeze(-1)  # b x 1

    z_neg = z_neg.t().unsqueeze(0).repeat(bs, 1, 1)  # b x z_dim x n
    neg_log_density = torch.bmm(z_next, z_neg).squeeze(1)  # b x n

    loss = torch.cat((torch.zeros(bs, 1).cuda(), neg_log_density - pos_log_density), dim=1)  # b x n+1
    loss = torch.logsumexp(loss, dim=1).mean()
    return loss


def train_cpc(encoder, trans, optimizer, train_loader, neg_train_inf, epoch):
    encoder.train()
    trans.train()

    train_losses = []
    pbar = tqdm(total=len(train_loader.dataset))
    for (obs, _), (obs_pos, _) in train_loader:
        obs, obs_pos = obs.cuda(), obs_pos.cuda() # b x 1 x 64 x 64
        obs_neg = next(neg_train_inf)[0].cuda() # n x 1 x 64 x 64

        loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        avg_loss = np.mean(train_losses[-50:])

        pbar.set_description('CPC Epoch {}, Train Loss {:.4f}'.format(epoch, avg_loss))
        pbar.update(obs.shape[0])
    pbar.close()


def test_cpc(encoder, trans, test_loader, neg_test_inf, epoch):
    encoder.eval()
    trans.eval()

    test_loss = 0
    for (obs, _), (obs_pos, _) in test_loader:
        obs, obs_pos = obs.cuda(), obs_pos.cuda()  # b x 1 x 64 x 64
        obs_neg = next(neg_test_inf)[0].cuda()  # n x 1 x 64 x 64

        loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans)
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
        x = x.cuda()
        z = encoder(x).detach()
        recon = decoder(z)
        loss = F.mse_loss(recon, x)
        test_loss += loss.item() * x.shape[0]
    test_loss /= len(test_loader.dataset)
    print('Dec Epoch {}, Test Loss: {:.4f}'.format(epoch, test_loss))


def save_recon(decoder, train_loader, test_loader, encoder, epoch, folder_name):
    decoder.eval()
    encoder.eval()

    train_batch = next(iter(train_loader))[0][:16]
    test_batch = next(iter(test_loader))[0][:16]
    train_batch, test_batch = train_batch.cuda(), test_batch.cuda()

    with torch.no_grad():
        train_z, test_z = encoder(train_batch), encoder(test_batch)
        train_recon, test_recon = decoder(train_z), decoder(test_z)

    real_imgs = torch.cat((train_batch, test_batch), dim=0)
    recon_imgs = torch.cat((train_recon, test_recon), dim=0)
    imgs = torch.stack((real_imgs, recon_imgs), dim=1)
    imgs = imgs.view(-1, *real_imgs.shape[1:]).cpu()

    folder_name = join(folder_name, 'reconstructions')
    if not exists(folder_name):
        os.makedirs(folder_name)

    filename = join(folder_name, 'recon_epoch{}.png'.format(epoch))
    save_image(imgs, filename, nrow=8)


def save_interpolation(decoder, start_images, goal_images, encoder, epoch, folder_name):
    decoder.eval()
    encoder.eval()

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
        imgs = decoder(zs).cpu()

    folder_name = join(folder_name, 'interpolations')
    if not exists(folder_name):
        os.makedirs(folder_name)

    filename = join(folder_name, 'interp_epoch{}.png'.format(epoch))
    save_image(imgs, filename, nrow=args.n_interp + 2)


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    obs_dim = (1, 64, 64)
    encoder = Encoder(args.z_dim, obs_dim[0]).cuda()
    trans = Transition(args.z_dim).cuda()
    decoder = Decoder(args.z_dim, obs_dim[0]).cuda()

    optim_cpc = optim.Adam(list(encoder.parameters()) + list(trans.parameters()),
                           lr=args.lr)
    optim_dec = optim.Adam(decoder.parameters(), lr=args.lr)

    train_loader, test_loader, neg_train_loader, neg_test_loader, neg_train_inf, neg_test_inf, start_images, goal_images = get_dataloaders()
    start_images, goal_images = start_images.cuda(), goal_images.cuda()

    # Save training images
    imgs = next(iter(neg_train_loader))[0][:64]
    save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_img.png'), nrow=8)

    (obs, _), (obs_next, _) = next(iter(train_loader))
    imgs = torch.stack((obs, obs_next), dim=1).view(-1, *obs_dim)
    save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_seq_img.png'), nrow=8)

    imgs = next(neg_train_inf)[0]
    save_image(imgs * 0.5 + 0.5, join(folder_name, 'neg.png'), nrow=10)

    for epoch in range(args.epochs):
        train_cpc(encoder, trans, optim_cpc, train_loader, neg_train_inf, epoch)
        test_cpc(encoder, trans, test_loader, neg_test_inf, epoch)

        if epoch % args.log_interval == 0:
            train_decoder(decoder, optim_dec, neg_train_loader, encoder, epoch)
            test_decoder(decoder, neg_test_loader, encoder, epoch)

            save_recon(decoder, neg_train_loader, neg_test_loader, encoder, epoch, folder_name)
            save_interpolation(decoder, start_images, goal_images, encoder, epoch, folder_name)

            torch.save(encoder, join(folder_name, 'encoder.pt'))
            torch.save(trans, join(folder_name, 'trans.pt'))
            torch.save(trans, join(folder_name, 'decoder.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/rope2/train_data')
    parser.add_argument('--test', type=str, default='data/rope2/test_data')
    parser.add_argument('--start', type=str, default='data/rope2/seq_data/start')
    parser.add_argument('--goal', type=str, default='data/rope2/seq_data/goal')
    parser.add_argument('--n_interp', type=int, default=8)

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
    main()
