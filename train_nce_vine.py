import argparse

import torch.nn.functional as F

import torch.utils.data as data
import torch.optim as optim

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from dataset import NCEVineDataset
from cpc_model import Encoder, Transition, InverseModel
from cpc_util import *


def get_dataloaders():
    transform = get_transform(False)

    train_dset = NCEVineDataset(root=join(args.root, 'train_data'), n_neg=args.n_neg,
                                transform=transform)
    if args.horovod:
        train_sampler = data.distributed.DistributedSampler(train_dset, num_replicas=hvd.size(),
                                                            rank=hvd.rank())
    else:
        train_sampler = None
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=not args.horovod, num_workers=4,
                                   pin_memory=True, sampler=train_sampler)

    test_dset = NCEVineDataset(root=join(args.root, 'test_data'), n_neg=args.n_neg,
                               transform=transform)
    if args.horovod:
        test_sampler = data.distributed.DistributedSampler(test_dset, num_replicas=hvd.size(),
                                                           rank=hvd.rank())
    else:
        test_sampler = None
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  shuffle=not args.horovod, num_workers=4,
                                  pin_memory=True, sampler=test_sampler)


    return train_loader, test_loader


def compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, inv, actions, device):
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
    z_neg = z_neg.view(bs, args.n_neg, args.z_dim).permute(0, 2, 1).contiguous() # b x z_dim x n
    neg_log_density = torch.bmm(z_next, z_neg).squeeze(1)  # b x n
    if args.mode == 'cos':
        neg_log_density /= torch.norm(z_next, dim=2) * torch.norm(z_neg, dim=1)

    loss = torch.cat((torch.zeros(bs, 1).to(device), neg_log_density - pos_log_density), dim=1)  # b x n+1
    loss = torch.logsumexp(loss, dim=1).mean()

    # loss += F.mse_loss(z_next, z_pos.detach())
    pred_a = inv(z, z_pos)
    print(actions.min(dim=1), actions.max(dim=1))
    loss += F.mse_loss(pred_a, actions)

    return loss


def train(encoder, trans, inv, optimizer, train_loader, epoch, device):
    encoder.train()
    trans.train()

    if not args.horovod or hvd.rank() == 0:
        train_losses = []
        pbar = tqdm(total=len(train_loader.sampler if args.horovod else train_loader.dataset))
    for batch in train_loader:
        obs, obs_pos, actions, obs_neg = [b.to(device) for b in batch]
        loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder,
                                trans, inv, actions, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not args.horovod or hvd.rank() == 0:
            train_losses.append(loss.item())
            avg_loss = np.mean(train_losses[-50:])

            pbar.set_description('Epoch {}, Train Loss {:.4f}'.format(epoch, avg_loss))
            pbar.update(obs.shape[0])
    if not args.horovod or hvd.rank() == 0:
        pbar.close()


def test(encoder, trans, inv, test_loader, epoch, device):
    encoder.eval()
    trans.eval()

    test_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_pos, actions, obs_neg = [b.to(device) for b in batch]
            loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder,
                                    trans, inv, actions, device)
            test_loss += loss * obs.shape[0]
    test_loss /= len(test_loader.sampler if args.horovod else test_loader.dataset)
  #  if args.horovod:
  #      test_loss = metric_average(test_loss, 'avg_loss')
    if not args.horovod or hvd.rank() == 0:
        print('Epoch {}, Test Loss: {:.4f}'.format(epoch, test_loss.item()))


def test_distance(encoder, trans, train_loader, device):
    encoder.eval()
    trans.eval()

    with torch.no_grad():
        batch = next(iter(train_loader))
        obs, obs_pos, actions, obs_neg = [b.to(device) for b in batch]
        bs = obs.shape[0]

        z, z_pos = encoder(obs), encoder(obs_pos)
        obs_neg = obs_neg.view(-1, *obs_neg.shape[2:]) # b * n x 1 x 64 x 64
        z_neg = encoder(obs_neg)  # b * n x z_dim

        inp = torch.cat((z, actions), dim=1)
        z_next = trans(inp)  # b x z_dim

        pos_log_density = (z_next * z_pos).sum(dim=1)
        pos_log_density_norm = pos_log_density / (torch.norm(z_next, dim=1) * torch.norm(z_pos, dim=1))

        z_next = z_next.unsqueeze(1)
        z_neg = z_neg.view(bs, args.n_neg, args.z_dim).permute(0, 2, 1).contiguous() # b x z_dim x n
        neg_log_density = torch.bmm(z_next, z_neg).squeeze(1)  # b x n
        neg_log_density_norm = neg_log_density / (torch.norm(z_next, dim=2) * torch.norm(z_neg, dim=1))

        print('Pos DP', pos_log_density.min().item(), pos_log_density.max().item())
        print('Neg DP', neg_log_density.min().item(), neg_log_density.max().item())

        print('Pos DP cos', pos_log_density_norm.min().item(), pos_log_density_norm.max().item())
        print('Neg DP cos', neg_log_density_norm.min().item(), neg_log_density_norm.max().item())

        pos_dist = torch.norm(z - z_pos, dim=1)
        trans_dist = torch.norm(z - z_next, dim=1)
        other_dist = torch.norm(z_pos - z_next, dim=1)

        print('z-z_pos dist', pos_dist.min().item(), pos_dist.max().item())
        print('z-z_next dist', trans_dist.min().item(), trans_dist.max().item())
        print('z_next-z_pos dist', other_dist.min().item(), other_dist.max().item())


def main():
    if args.horovod:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    obs_dim = (1, 64, 64)
    action_dim = 4

    device = torch.device('cuda:{}'.format(hvd.rank())) if args.horovod else torch.device('cuda')
    load_fcn_mse(device)

    encoder = Encoder(args.z_dim, obs_dim[0]).to(device)
    trans = Transition(args.z_dim, action_dim).to(device)
    parameters = list(encoder.parameters()) + list(trans.parameters())
    if args.inv_model:
        inv = InverseModel(args.z_dim, action_dim).to(device)
        parameters += list(inv.parameters())
    else:
        inv = None

    optimizer = optim.Adam(parameters, lr=args.lr)
    if args.horovod:
        enc_np = encoder.named_parameters(prefix=encoder.prefix)
        trans_np = trans.named_parameters(prefix=trans.prefix)
        named_parameters = list(enc_np) + list(trans_np)
        if args.inv_model:
            named_parameters += list(inv.named_parameters(prefix=inv.prefix))
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=named_parameters
        )

        hvd.broadcast_parameters(encoder.state_dict(), root_rank=0)
        hvd.broadcast_parameters(trans.state_dict(), root_rank=0)
        if args.inv_model:
            hvd.broadcast_parameters(inv.state_dict(0, root_rank=0))
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    train_loader, test_loader = get_dataloaders()
    if not args.horovod or hvd.rank() == 0:
        # Save training images
        batch = next(iter(train_loader))
        obs, obs_next, _, obs_neg = batch
        imgs = torch.stack((obs, obs_next), dim=1).view(-1, *obs.shape[1:])
        save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_seq_img.png'), nrow=8)

        obs_neg = obs_neg.view(-1, *obs_dim)[:100]
        save_image(obs_neg * 0.5 + 0.5, join(folder_name, 'neg.png'), nrow=10)

    if not args.horovod or hvd.rank() == 0:
        test_distance(encoder, trans, train_loader, device)
    for epoch in range(args.epochs):
        if args.horovod:
            MPI.COMM_WORLD.Barrier()
        train(encoder, trans, inv, optimizer, train_loader, epoch, device)
        test(encoder, trans, inv, test_loader, epoch, device)

        if epoch % args.log_interval == 0 and (not args.horovod or hvd.rank() == 0):
            test_distance(encoder, trans, train_loader, device)

            torch.save(encoder, join(folder_name, 'encoder.pt'))
            torch.save(trans, join(folder_name, 'trans.pt'))
            if args.inv_model:
                torch.save(inv, join(folder_name, 'inv.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope2')
    parser.add_argument('--n_interp', type=int, default=8)
    parser.add_argument('--mode', type=str, default='dotproduct')
    parser.add_argument('--inv_model', action='store_true')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)

    parser.add_argument('--n_neg', type=int, default=50)
    parser.add_argument('--z_dim', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)

    parser.add_argument('--horovod', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='cpc')
    args = parser.parse_args()

    assert args.mode in ['dotproduct', 'cos']
    if args.horovod:
        import horovod.torch as hvd
        from mpi4py import MPI

    main()
