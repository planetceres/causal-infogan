import sys
import torch
import numpy as np
import cv2
import glob
from os.path import join
import random
from dataset import NCEVineDataset
import torch.utils.data as data

n = 32
n_neg = 12
root = sys.argv[1]

dset = NCEVineDataset(root, n_neg)
data_loader = data.DataLoader(dset, batch_size=n, shuffle=True)

obs, obs_pos, actions, obs_neg = next(iter(data_loader))
obs, obs_pos, obs_neg = obs * 0.5 + 0.5, obs_pos * 0.5 + 0.5, obs_neg * 0.5 + 0.5
obs = obs.permute(0, 2, 3, 1).contiguous().repeat(1, 1, 1, 3)
obs, actions = obs.numpy(), actions.numpy()
obs = (obs * 255).astype(np.uint8)

actions = actions * dset.std[None, :] + dset.mean[None, :]

for i in range(obs.shape[0]):
    act = actions[i]
    loc = act[:2] * 63
    act = act[2:]
    act[1] = -act[1]
    act = act[[1, 0]]
    act *= 0.25 * 64

    startr, startc = loc
    endr, endc = loc + act
    startr, startc, endr, endc = int(startr), int(startc), int(endr), int(endc)
    cv2.arrowedLine(obs[i], (startc, startr), (endc, endr), (125, 125, 125), 3)
obs = obs.astype('float32') / 255.
obs = torch.FloatTensor(obs)
obs = obs.permute(0, 3, 1, 2).contiguous().mean(dim=1, keepdim=True)
obs, obs_pos = obs.unsqueeze(1), obs_pos.unsqueeze(1)

imgs = torch.cat((obs, obs_pos, obs_neg), dim=1)
print(imgs.shape)
imgs = imgs.view(-1, *imgs.shape[2:])

from torchvision.utils import save_image
save_image(imgs, 'pt_transitions.png', nrow=n_neg + 2)
