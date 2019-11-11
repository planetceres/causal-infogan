from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms

from model import Classifier
from dataset import ImagePairs


def train(model, optimizer, train_loader, epoch):
    model.train()

    train_losses = []
    pbar = tqdm(total=len(train_loader.dataset))
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_loss = np.mean(train_losses[-10:])
        pbar.set_description("Epoch {}, Train Loss: {:.4f}".format(epoch, train_loss))
        pbar.update(x.shape[0])
    pbar.close()


def test(model, test_loader, epoch):
    model.eval()

    test_loss = 0
    for x, y in test_loader:
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y)
            test_loss += loss.item() * x.shape[0]
    test_loss /= len(test_loader.dataset)
    print("Epoch {}, Test Loss: {:4f}".format(epoch, test_loss))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = Classifier().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dset = ImagePairs(root=args.train_root, transform=transform,
                            n_frames_apart=args.n_frames_apart, include_neg=True)
    test_dset = ImagePairs(root=args.test_root, transform=transform,
                           n_frames_apart=args.n_frames_apart, include_neg=True)
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2)

    for epoch in range(args.epochs):
        train(model, optimizer, train_loader, epoch)
        test(model, test_loader, epoch)

        torch.save(model.state_dict(), 'classifier.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--train_root', type=str, default='data/train_rope')
    parser.add_argument('--test_root', type=str, default='data/test_rope')
    parser.add_argument('--n_frames_apart', type=int, default=1)
    args = parser.parse_args()

    main()

