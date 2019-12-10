import argparse
from tqdm import tqdm
import numpy as np
from os.path import join

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from dataset import NCEVineDataset
from cpc_model import InverseModel, ForwardModel
from cpc_util import *


def get_dataloaders():
    transform = get_transform(False)

    train_dset = NCEVineDataset(root=join(args.root, 'train_data'), n_neg=0,
                                transform=transform)
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4,
                                   pin_memory=True)

    test_dset = NCEVineDataset(root=join(args.root, 'test_data'), n_neg=0,
                               transform=transform)
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4,
                                  pin_memory=True)


    return train_loader, test_loader


def compute_losses(fwd_model, inv_model, encoder, obs, obs_next, actions):
    z, z_next = encoder(obs).detach(), encoder(obs_next).detach()
    a_pred = inv_model(z, z_next)
    loss_inv = F.mse_loss(a_pred, actions)

    z_pred = fwd_model(z, actions)
    loss_fwd = F.mse_loss(z_pred, z_next)

    return loss_inv, loss_fwd

def train(fwd_model, inv_model, encoder, opt_fwd, opt_inv, train_loader, epoch, device):
    fwd_model.train()
    inv_model.train()

    pbar = tqdm(total=len(train_loader.dataset))
    inv_losses, fwd_losses = [], []
    for batch in train_loader:
        obs, obs_next, actions = [b.to(device) for b in batch[:-1]]
        loss_inv, loss_fwd = compute_losses(fwd_model, inv_model, encoder,
                                            obs, obs_next, actions)
        opt_inv.zero_grad()
        loss_inv.backward()
        opt_inv.step()

        opt_fwd.zero_grad()
        loss_fwd.backward()
        opt_fwd.step()

        inv_losses.append(loss_inv.item())
        fwd_losses.append(loss_fwd.item())
        avg_inv_loss = np.mean(inv_losses[-50:])
        avg_fwd_loss = np.mean(fwd_losses[-50:])

        pbar.set_description('Epoch {}, Inv Loss {:.4f}, Fwd Loss {:.4f}'.format(epoch, avg_inv_loss, avg_fwd_loss))
        pbar.update(obs.shape[0])
    pbar.close()


def test(fwd_model, inv_model, encoder, test_loader, epoch, device):
    fwd_model.eval()
    inv_model.eval()

    inv_loss, fwd_loss = 0, 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_next, actions = [b.to(device) for b in batch[:-1]]
            loss_inv, loss_fwd = compute_losses(fwd_model, inv_model, encoder,
                                                obs, obs_next, actions)
            inv_loss += loss_inv.item() * obs.shape[0]
            fwd_loss += loss_fwd.item() * obs.shape[0]
    inv_loss /= len(test_loader.dataset)
    fwd_loss /= len(test_loader.dataset)

    print('Test Epoch {}, Inv Loss {:.4f}, Fwd Loss {:.4f}'.format(epoch, inv_loss, fwd_loss))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    assert exists(folder_name)

    action_dim = 4
    device = torch.device('cuda')
    train_loader, test_loader = get_dataloaders()

    encoder = torch.load(join(folder_name, 'encoder.pt'), map_location=device)
    encoder.eval()

    fwd_model = ForwardModel(encoder.z_dim, action_dim).to(device)
    inv_model = InverseModel(encoder.z_dim, action_dim).to(device)

    opt_fwd = optim.Adam(fwd_model.parameters(), lr=args.lr)
    opt_inv = optim.Adam(inv_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(fwd_model, inv_model, encoder, opt_fwd, opt_inv,
              train_loader, epoch, device)
        test(fwd_model, inv_model, encoder, test_loader, epoch, device)

        torch.save(fwd_model, join(folder_name, 'fwd_model.pt'))
        torch.save(inv_model, join(folder_name, 'inv_model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()

    main()
