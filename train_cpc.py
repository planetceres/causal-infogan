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

from dataset import ImagePairs
from cpc_model import Encoder, Decoder, Transition
from model import FCN_mse


def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


def apply_fcn_mse(img):
    o = fcn(img.cuda()).detach()
    return torch.clamp(2 * (o - 0.5), -1 + 1e-3, 1 - 1e-3)


def get_dataloaders():
    def filter_background(x):
        x[:, (x < 0.3).any(dim=0)] = 0.0
        return x

    def dilate(x):
        x = x.squeeze(0).numpy()
        x = grey_dilation(x, size=3)
        x = x[None, :, :]
        return torch.from_numpy(x)

    if args.thanard_dset:
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            filter_background,
            lambda x: x.mean(dim=0)[None, :, :],
            dilate,
            transforms.Normalize((0.5,), (0.5,)),
        ])

    train_dset = ImagePairs(root=join(args.root, 'train_data'), include_actions=args.include_actions,
                            thanard_dset=args.thanard_dset, transform=transform, n_frames_apart=args.k)
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dset = ImagePairs(root=join(args.root, 'test_data'), include_actions=args.include_actions,
                           thanard_dset=args.thanard_dset, transform=transform, n_frames_apart=args.k)
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    neg_train_dset = ImageFolder(join(args.root, 'train_data'), transform=transform)
    neg_train_loader = data.DataLoader(neg_train_dset, batch_size=args.batch_size, shuffle=True,
                                       pin_memory=True, num_workers=2) # for training decoder
    neg_train_inf = infinite_loader(data.DataLoader(neg_train_dset, batch_size=args.n, shuffle=True,
                                                    pin_memory=True, num_workers=2)) # to get negative samples

    neg_test_dset = ImageFolder(join(args.root, 'test_data'), transform=transform)
    neg_test_loader = data.DataLoader(neg_test_dset, batch_size=args.batch_size, shuffle=True,
                                       pin_memory=True, num_workers=2)
    neg_test_inf = infinite_loader(data.DataLoader(neg_test_dset, batch_size=args.n, shuffle=True,
                                                   pin_memory=True, num_workers=2))


    start_dset = ImageFolder(join(args.root, 'seq_data', 'start'), transform=transform)
    goal_dset = ImageFolder(join(args.root, 'seq_data', 'goal'), transform=transform)

    start_images = torch.stack([start_dset[i][0] for i in range(len(start_dset))], dim=0)
    goal_images = torch.stack([goal_dset[i][0] for i in range(len(goal_dset))], dim=0)

    n = min(start_images.shape[0], goal_images.shape[0])
    start_images, goal_images = start_images[:n], goal_images[:n]

    return train_loader, test_loader, neg_train_loader, neg_test_loader, neg_train_inf, neg_test_inf, start_images, goal_images


def compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=None):
    assert (args.include_actions and actions is not None) or (not args.include_actions and actions is None)
    bs = obs.shape[0]

    z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    z_neg = encoder(obs_neg)  # n x z_dim

    z = torch.cat((z, actions), dim=1) if args.include_actions else z
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
    for batch in train_loader:
        if args.include_actions:
            (obs, _, actions), (obs_pos, _, _) = batch
            actions = actions.cuda()
        else:
            (obs, _), (obs_pos, _) = batch
            actions = None

        if args.thanard_dset:
            obs, obs_pos = apply_fcn_mse(obs), apply_fcn_mse(obs_pos)
            obs_neg = apply_fcn_mse(next(neg_train_inf)[0])
        else:
            obs, obs_pos = obs.cuda(), obs_pos.cuda() # b x 1 x 64 x 64
            obs_neg = next(neg_train_inf)[0].cuda() # n x 1 x 64 x 64

        loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=actions)
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
    for batch in test_loader:
        if args.include_actions:
            (obs, _, actions), (obs_pos, _, _) = batch
            actions = actions.cuda()
        else:
            (obs, _), (obs_pos, _) = batch
            actions = None

        if args.thanard_dset:
            obs, obs_pos = apply_fcn_mse(obs), apply_fcn_mse(obs_pos)
            obs_neg = apply_fcn_mse(next(neg_test_inf)[0])
        else:
            obs, obs_pos = obs.cuda(), obs_pos.cuda()  # b x 1 x 64 x 64
            obs_neg = next(neg_test_inf)[0].cuda()  # n x 1 x 64 x 64

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
        x = apply_fcn_mse(x) if args.thanard_dset else x.cuda()

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


def save_nearest_neighbors(encoder, train_loader, test_loader,
                           epoch, folder_name, k=100):
    encoder.eval()

    train_batch = next(iter(train_loader))[0][:5]
    test_batch = next(iter(test_loader))[0][:5]

    with torch.no_grad():
        batch = torch.cat((train_batch, test_batch), dim=0)
        batch = apply_fcn_mse(batch) if args.thanard_dset else batch.cuda()
        z = encoder(batch) # 10 x z_dim
        zz = (z ** 2).sum(-1).unsqueeze(1) # z^Tz, 10 x 1

        pbar = tqdm(total=len(train_loader.dataset) + len(test_loader.dataset))
        pbar.set_description('Computing NN')
        dists = []
        for loader in [train_loader, test_loader]:
            for x, _ in loader:
                x = apply_fcn_mse(x) if args.thanard_dset else x.cuda()
                zx = encoder(x) # b x z_dim
                zzx = torch.matmul(z, zx.t()) # z_1^Tz_2, 10 x b
                zxzx = (zx ** 2).sum(-1).unsqueeze(0) #zx^Tzx, 1 x b
                dist = zz - 2 * zzx + zxzx # norm squared distance, 10 x b
                dists.append(dist.cpu())
                pbar.update(x.shape[0])
        dists = torch.cat(dists, dim=1) # 10 x dset_size
        topk = torch.topk(dists, k + 1, dim=1, largest=False)[1]

        pbar.close()

    folder_name = join(folder_name, 'nn_epoch{}'.format(epoch))
    if not exists(folder_name):
        os.makedirs(folder_name)

    train_size = len(train_loader.dataset)
    for i in range(10):
        imgs = []
        for idx in topk[i]:
            if idx >= train_size:
                imgs.append(test_loader.dataset[idx - train_size][0])
            else:
                imgs.append(train_loader.dataset[idx][0])
        imgs = torch.stack(imgs, dim=0)
        if args.thanard_dset:
            imgs = apply_fcn_mse(imgs).cpu()
        save_image(imgs * 0.5 + 0.5, join(folder_name, 'nn_{}.png'.format(i)), nrow=10)


def save_recon(decoder, train_loader, test_loader, encoder, epoch, folder_name):
    decoder.eval()
    encoder.eval()

    train_batch = next(iter(train_loader))[0][:16]
    test_batch = next(iter(test_loader))[0][:16]
    if args.thanard_dset:
        train_batch, test_batch = apply_fcn_mse(train_batch), apply_fcn_mse(test_batch)
    else:
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
    save_image(imgs * 0.5 + 0.5, filename, nrow=8)


def save_interpolation(decoder, start_images, goal_images, encoder, epoch, folder_name):
    decoder.eval()
    encoder.eval()

    with torch.no_grad():
        z_start = encoder(start_images)
        z_goal = encoder(goal_images)

        lambdas = np.linspace(0, 1, args.n_interp + 2)
        zs = torch.stack([(1 - lambda_) * z_start + lambda_ * z_goal
                          for lambda_ in lambdas], dim=1)  # n x n_interp+2 x z_dim
        zs = zs.view(-1, args.z_dim)  # n * (n_interp+2) x z_dim

        imgs = decoder(zs).cpu()

    folder_name = join(folder_name, 'interpolations')
    if not exists(folder_name):
        os.makedirs(folder_name)

    filename = join(folder_name, 'interp_epoch{}.png'.format(epoch))
    save_image(imgs * 0.5 + 0.5, filename, nrow=args.n_interp + 2)


def save_run_dynamics(decoder, encoder, trans, start_images,
                      train_loader, epoch, folder_name):
    decoder.eval()
    encoder.eval()

    if args.include_actions:
        dset = train_loader.dataset
        transform = dset.transform
        with torch.no_grad():
            actions, images = [], []
            n_ep = 5
            for i in range(n_ep):
                class_name = [name for name, idx in dset.class_to_idx.items() if idx == i]
                assert len(class_name) == 1
                class_name = class_name[0]

                a = np.load(join(args.root, 'train_data', class_name, 'actions.npy'))
                a = torch.FloatTensor(a)
                actions.append(a)
                ext = 'jpg' if args.thanard_dset else 'png'
                img_files = glob.glob(join(args.root, 'train_data', class_name, '*.{}'.format(ext)))
                img_files = sorted(img_files)
                image = torch.stack([transform(default_loader(f)) for f in img_files], dim=0)
                images.append(image)
            min_length = min(min([img.shape[0] for img in images]), 10)
            actions = [a[:min_length] for a in actions]
            images = [img[:min_length] for img in images]
            actions, images = torch.stack(actions, dim=0), torch.stack(images, dim=0)
            images = images.view(-1, *images.shape[2:])
            images = apply_fcn_mse(images) if args.thanard_dset else images.cuda()
            images = images.view(n_ep, min_length, *images.shape[1:])
            actions = actions.cuda()

            zs = [encoder(images[:, 0])]
            for i in range(min_length - 1):
                zs.append(trans(torch.cat((zs[-1], actions[:, i]), dim=1)))
            zs = torch.stack(zs, dim=1)
            zs = zs.view(-1, args.z_dim)
            recon = decoder(zs)
            recon = recon.view(n_ep, min_length, *images.shape[2:])

            all_imgs = torch.stack((images, recon), dim=1)
            all_imgs = all_imgs.view(-1, *all_imgs.shape[3:])

            folder_name = join(folder_name, 'run_dynamics')
            if not exists(folder_name):
                os.makedirs(folder_name)

            filename = join(folder_name, 'dyn_epoch{}.png'.format(epoch))
            save_image(all_imgs * 0.5 + 0.5, filename, nrow=min_length)
    else:
        with torch.no_grad():
            zs = [encoder(start_images)]
            for _ in range(args.n_interp):
                zs.append(trans(zs[-1]))
            zs = torch.stack(zs, dim=1)
            zs = zs.view(-1, args.z_dim)
            imgs = decoder(zs).cpu()

        folder_name = join(folder_name, 'run_dynamics')
        if not exists(folder_name):
            os.makedirs(folder_name)

        filename = join(folder_name, 'dyn_epoch{}.png'.format(epoch))
        save_image(imgs * 0.5 + 0.5, filename, nrow=args.n_interp + 1)


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    obs_dim = (1, 64, 64)
    action_dim = 5 if args.thanard_dset else 4

    encoder = Encoder(args.z_dim, obs_dim[0]).cuda()
    trans = Transition(args.z_dim, args.include_actions * action_dim).cuda()
    decoder = Decoder(args.z_dim, obs_dim[0]).cuda()

    optim_cpc = optim.Adam(list(encoder.parameters()) + list(trans.parameters()),
                           lr=args.lr)
    optim_dec = optim.Adam(decoder.parameters(), lr=args.lr)

    train_loader, test_loader, neg_train_loader, neg_test_loader, neg_train_inf, neg_test_inf, start_images, goal_images = get_dataloaders()
    if args.thanard_dset:
        start_images, goal_images = apply_fcn_mse(start_images), apply_fcn_mse(goal_images)
    else:
        start_images, goal_images = start_images.cuda(), goal_images.cuda()

    # Save training images
    imgs = next(iter(neg_train_loader))[0][:64]
    if args.thanard_dset:
        imgs = apply_fcn_mse(imgs).cpu()
    save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_img.png'), nrow=8)

    batch = next(iter(train_loader))
    if args.include_actions:
        (obs, _, _), (obs_next, _, _) = batch
    else:
        (obs, _), (obs_next, _) = batch
    imgs = torch.stack((obs, obs_next), dim=1).view(-1, *obs.shape[1:])
    if args.thanard_dset:
        imgs = apply_fcn_mse(imgs).cpu()
    save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_seq_img.png'), nrow=8)

    imgs = next(neg_train_inf)[0]
    if args.thanard_dset:
        imgs = apply_fcn_mse(imgs).cpu()
    save_image(imgs * 0.5 + 0.5, join(folder_name, 'neg.png'), nrow=10)

    for epoch in range(args.epochs):
        train_cpc(encoder, trans, optim_cpc, train_loader, neg_train_inf, epoch)
        test_cpc(encoder, trans, test_loader, neg_test_inf, epoch)

        if epoch % args.log_interval == 0:
            train_decoder(decoder, optim_dec, neg_train_loader, encoder, epoch)
            # test_decoder(decoder, neg_test_loader, encoder, epoch)

            save_recon(decoder, neg_train_loader, neg_test_loader, encoder, epoch, folder_name)
            save_interpolation(decoder, start_images, goal_images, encoder, epoch, folder_name)
            save_run_dynamics(decoder, encoder, trans, start_images, neg_train_loader, epoch, folder_name)
            save_nearest_neighbors(encoder, neg_train_loader, neg_test_loader, epoch, folder_name)

            torch.save(encoder, join(folder_name, 'encoder.pt'))
            torch.save(trans, join(folder_name, 'trans.pt'))
            torch.save(trans, join(folder_name, 'decoder.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope2')
    parser.add_argument('--n_interp', type=int, default=8)
    parser.add_argument('--thanard_dset', action='store_true')
    parser.add_argument('--include_actions', action='store_true')

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

    if args.thanard_dset:
        fcn = FCN_mse(2).cuda()
        fcn.load_state_dict(torch.load('/home/wilson/causal-infogan/data/FCN_mse'))
        fcn.eval()

    main()
