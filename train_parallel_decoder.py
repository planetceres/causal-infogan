from tqdm import tqdm
import argparse
from scipy.ndimage.morphology import grey_dilation
from mpi4py import MPI

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import horovod.torch as hvd

from torchvision import transforms, utils, datasets

from cpc_model import Decoder
from cpc_util import *


def get_data():
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

    train_dset = datasets.ImageFolder(join(args.root, 'train_data'), transform=transform)
    train_sampler = data.distributed.DistributedSampler(train_dset, num_replicas=hvd.size(),
                                                        rank=hvd.rank())
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4, sampler=train_sampler)

    test_dset = datasets.ImageFolder(join(args.root, 'test_data'), transform=transform)
    test_sampler = data.distributed.DistributedSampler(test_dset, num_replicas=hvd.size(),
                                                       rank=hvd.rank())
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4, sampler=test_sampler)

    return train_loader, test_loader


def metric_average(val, name):
    tensor = val.clone()
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train(model, optimizer, train_loader, encoder, epoch, device):
    model.train()

    train_loader.sampler.set_epoch(epoch)
    if hvd.rank() == 0:
        train_losses = []
        pbar = tqdm(total=len(train_loader.sampler))
    for x, _ in train_loader:
        with torch.no_grad():
            x = apply_fcn_mse(x, device=device) if args.thanard_dset else x.to(device)
            z = encoder(x).detach()
        recon = model(z)
        loss = F.mse_loss(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hvd.rank() == 0:
            train_losses.append(loss.item())
            avg_loss = np.mean(train_losses[-50:])

            pbar.set_description('Epoch {}, Train Loss {:.4f}'.format(epoch, avg_loss))
            pbar.update(x.shape[0])
    if hvd.rank() == 0:
        pbar.close()


def test(model, test_loader, encoder, epoch, device):
    model.eval()

    test_loss = 0
    for x, _ in test_loader:
        with torch.no_grad():
            x = apply_fcn_mse(x, device=device) if args.thanard_dset else x.to(device)
            z = encoder(x).detach()
            recon = model(z)
            loss = F.mse_loss(recon, x)
            test_loss += loss.item() * x.shape[0]
    test_loss /= len(test_loader.sampler)
    test_loss = metric_average(test_loss, 'avg_loss')

    if hvd.rank() == 0:
        print('Epoch {}, Test Loss: {:.4f}'.format(epoch, test_loss))


def main():
    hvd.init()
    np.random.seed(args.seed + hvd.rank())
    torch.cuda.set_device(hvd.local_rank())
    torch.manual_seed(args.seed + hvd.rank())
    torch.cuda.manual_seed(args.seed + hvd.rank())

    folder_name = join('out', args.name)
    assert exists(folder_name)

    train_loader, test_loader = get_data()
    device = torch.cuda('cuda:{}'.format(hvd.local_rank()))

    encoder = torch.load(join(folder_name, 'encoder.pt'), map_location=device)
    encoder.eval()
    trans = torch.load(join(folder_name, 'trans.pt'), map_location=device)
    trans.eval()

    model = Decoder(encoder.z_dim, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    if hvd.rank() == 0:
        imgs = next(iter(train_loader))[0]
        if args.thanard_dset:
            imgs = apply_fcn_mse(imgs).cpu()
        utils.save_image(imgs * 0.5 + 0.5, join(folder_name, 'dec_train_img.png'))

        save_nearest_neighbors(encoder, train_loader, test_loader,
                               -1, folder_name, thanard_dset=args.thanard_dset,
                               metric='dotproduct')

    for epoch in range(args.epochs):
        MPI.COMM_WORLD.Barrier()
        train(model, optimizer, train_loader, encoder, epoch, device)
        test(model, test_loader, encoder, epoch, device)

        if epoch % args.log_interval == 0 and hvd.rank() == 0:
            save_recon(model, train_loader, test_loader, encoder,
                       epoch, folder_name, thanard_dset=args.thanard_dset)
            start_images, goal_images = next(iter(train_loader))[0][:20].to(device).chunk(2, dim=0)
            save_interpolation(args.n_interp, model, start_images, goal_images, encoder,
                               epoch, folder_name)
            save_run_dynamics(model, encoder, trans, train_loader,
                              epoch, folder_name, args.root,
                              include_actions=args.include_actions,
                              thanard_dset=args.thanard_dset)
            torch.save(model, join(folder_name, 'decoder.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope')
    parser.add_argument('--n_interp', type=int, default=8)
    parser.add_argument('--thanard_dset', action='store_true')
    parser.add_argument('--include_actions', action='store_true')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='recon')
    args = parser.parse_args()

    main()
