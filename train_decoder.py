from tqdm import tqdm
import argparse
from scipy.ndimage.morphology import grey_dilation

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision import transforms, utils, datasets

from cpc_model import Decoder
from cpc_util import *


def get_dataloaders():
    transform = get_transform(args.thanard_dset)

    train_dset = datasets.ImageFolder(join(args.root, 'train_data'), transform=transform)
    if args.horovod:
        train_sampler = data.distributed.DistributedSampler(train_dset, num_replicas=hvd.size(),
                                                            rank=hvd.rank())
    else:
        train_sampler = None
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=not args.horovod, pin_memory=True,
                                   num_workers=4, sampler=train_sampler)

    test_dset = datasets.ImageFolder(join(args.root, 'test_data'), transform=transform)
    if args.horovod:
        test_sampler = data.distributed.DistributedSampler(test_dset, num_replicas=hvd.size(),
                                                           rank=hvd.rank())
    else:
        test_sampler = None
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  shuffle=not args.horovod, pin_memory=True,
                                  num_workers=4, sampler=test_sampler)

    return train_loader, test_loader


def train(model, optimizer, train_loader, encoder, epoch, device):
    model.train()

    if not args.horovod or hvd.rank() == 0:
        train_losses = []
        pbar = tqdm(total=len(train_loader.sampler if args.horovod else train_loader.dataset))
    for x, _ in train_loader:
        x = apply_fcn_mse(x, device) if args.thanard_dset else x.to(device)
        z = encoder(x).detach()
        loss = model.loss(x, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not args.horovod or hvd.rank() == 0:
            train_losses.append(loss.item())
            avg_loss = np.mean(train_losses[-50:])

            pbar.set_description('Epoch {}, Train Loss {:.4f}'.format(epoch, avg_loss))
            pbar.update(x.shape[0])
    if not args.horovod or hvd.rank() == 0:
        pbar.close()


def test(model, test_loader, encoder, epoch, device):
    model.eval()

    test_loss = 0
    for x, _ in test_loader:
        x = apply_fcn_mse(x, device) if args.thanard_dset else x.to(device)
        z = encoder(x).detach()
        loss = model.loss(x, z)
        test_loss += loss * x.shape[0]
    test_loss /= len(test_loader.sampler if args.horovod else test_loader.dataset)
    if args.horovod:
        test_loss = metric_average(test_loss, 'avg_loss')
    if not args.horovod or hvd.rank() == 0:
        print('Epoch {}, Test Loss: {:.4f}'.format(epoch, test_loss))


def main():
    if args.horovod:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    assert exists(folder_name)

    device = torch.device('cuda:{}'.format(hvd.rank())) if args.horovod else torch.device('cuda')
    train_loader, test_loader = get_dataloaders()
    load_fcn_mse(device)

    encoder = torch.load(join(folder_name, 'encoder.pt'), map_location=device)
    encoder.eval()
    trans = torch.load(join(folder_name, 'trans.pt'), map_location=device)
    trans.eval()

    model = Decoder(encoder.z_dim, 1, discrete=args.discrete, n_bit=args.n_bit).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.horovod:
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters()
        )

        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    if not args.horovod or hvd.rank() == 0:
        imgs = next(iter(train_loader))[0]
        if args.thanard_dset:
            imgs = apply_fcn_mse(imgs, device).cpu()
        utils.save_image(imgs * 0.5 + 0.5, join(folder_name, 'dec_train_img.png'))

        save_nearest_neighbors(encoder, train_loader, test_loader,
                               -1, folder_name, device, thanard_dset=args.thanard_dset,
                               metric='dotproduct')

    for epoch in range(args.epochs):
        if args.horovod:
            MPI.COMM_WORLD.Barrier()
        train(model, optimizer, train_loader, encoder, epoch, device)
        test(model, test_loader, encoder, epoch, device)

        if epoch % args.log_interval == 0 and (not args.horovod or hvd.rank() == 0):
            save_recon(model, train_loader, test_loader, encoder,
                       epoch, folder_name, device, thanard_dset=args.thanard_dset)
            start_images, goal_images = next(iter(train_loader))[0][:20].to(device).chunk(2, dim=0)
            save_interpolation(args.n_interp, model, start_images, goal_images, encoder,
                               epoch, folder_name)
            save_run_dynamics(model, encoder, trans, train_loader,
                              epoch, folder_name, args.root, device,
                              include_actions=args.include_actions,
                              thanard_dset=args.thanard_dset, vine=args.vine)
            torch.save(model, join(folder_name, 'decoder.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope')
    parser.add_argument('--n_interp', type=int, default=8)
    parser.add_argument('--thanard_dset', action='store_true')
    parser.add_argument('--include_actions', action='store_true')
    parser.add_argument('--vine', action='store_true')

    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--n_bit', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)

    parser.add_argument('--horovod')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='recon')
    args = parser.parse_args()

    if args.horovod:
        import horovod.torch as hvd
        from mpi4py import MPI

    main()
