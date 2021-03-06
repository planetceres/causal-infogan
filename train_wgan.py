import os
from os.path import join, exists
from tqdm import tqdm
import argparse
from scipy.ndimage.morphology import grey_dilation

import torch
import torch.optim as optim
import torch.utils.data as data

from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from model import WGAN, FCN_mse, BigWGAN

def inf_iterator(data_loader):
    epoch = 0
    while True:
        for batch in data_loader:
            yield batch
        epoch += 1

def train(model, fcn, data_loader):
    itrs = args.itrs
    log_interval = args.log_interval
    n_critic = 5

    optimizerG = optim.Adam(model.gen.parameters(), lr=args.lr, betas=(0, 0.9))
    optimizerD = optim.Adam(model.disc.parameters(), lr=args.lr, betas=(0, 0.9))

    data_gen = inf_iterator(data_loader)
    filepath = join('out', args.name)
    if not exists(filepath):
        os.makedirs(filepath)

    saved = False
    pbar = tqdm(total=itrs)
    model.train()
    for itr in range(itrs):
        for _ in range(n_critic):
            x,  _ = next(data_gen)
            x  = x.cuda()
            #x = apply_fcn_mse(fcn, x)
            batch_size = x.size(0)

            if not saved:
                save_image(x * 0.5 + 0.5, join(filepath, 'example_dset.png'))
                saved = True

            optimizerD.zero_grad()
            x_tilde = model.generate(batch_size)
            eps = model.sample_eps(batch_size).view(batch_size, 1, 1, 1)
            x_hat = eps * x + (1 - eps) * x_tilde
            disc_loss = model.gan_loss(x_tilde, x)
            grad_penalty = model.grad_penalty(x_hat)
            (disc_loss + grad_penalty).backward()
            optimizerD.step()

        optimizerG.zero_grad()
        gz = model.generate(batch_size)
        gen_loss = model.generator_loss(gz)
        gen_loss.backward()
        optimizerG.step()

        pbar.set_description('G: {:.4f}, D: {:.4f}, Pen: {:.4f}'.format(gen_loss.item(), disc_loss.item(), grad_penalty.item()))

        if itr % log_interval == 0:
            model.eval()
            samples = model.sample(64)
            save_image(samples, join(filepath, 'samples_itr{}.png'.format(itr)))
            model.train()

        pbar.update(1)
    pbar.close()

def apply_fcn_mse(fcn, img):
    o = fcn(img).detach()
    return torch.clamp(2 * (o - 0.5), -1 + 1e-3, 1 - 1e-3)

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #fcn = FCN_mse(2).cuda()
    #fcn.load_state_dict(torch.load('/home/wilson/causal-infogan/data/FCN_mse'))
    #fcn.eval()
    fcn = None

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

    dataset = ImageFolder(args.root, transform=transform)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=2)

   # model = WGAN(32, 1).cuda()
    model = BigWGAN((1, 64, 64), z_dim=32).cuda()
    train(model, fcn, loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope/full_data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--itrs', type=int, default=int(1e5))
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='wgan')
    args = parser.parse_args()
    main()
