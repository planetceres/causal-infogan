import argparse
from os.path import join
from tqdm import tqdm
from scipy.ndimage.morphology import grey_dilation
import h5py
import numpy as np
import pickle

import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader as loader

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
args = parser.parse_args()

with open(join(args.root, 'pos_neg_pairs.pkl'), 'rb') as f:
    data = pickle.load(f)
all_images = data['all_images']

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

dset = h5py.File(join(args.root, 'images.hdf5'), 'x')
dset.create_dataset('images', (len(all_images), 1, 64, 64), 'uint8')
stored = []
for i, img in enumerate(tqdm(all_images)):
    img = transform(loader(img))
    img = img.numpy() * 0.5 + 0.5
    img *= 255
    img = img.astype(np.uint8)
    dset['images'][i] = img
