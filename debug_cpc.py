from os.path import join
from scipy.ndimage.morphology import grey_dilation

import torch
import torch.utils.data as data

from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataset import NCEDataset
from model import FCN_mse

batch_size = 128
name = 'z16_n15_mlptrans_novine'
root = 'data/rope/'
n_neg = 15
encoder = torch.load(join('out', name, 'encoder.pt'), map_location='cuda')
trans = torch.load(join('out', name, 'trans.pt'), map_location='cuda')

train_dset = NCEDataset(root=join(root, 'train_data'), n_neg=n_neg)
train_loader = data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)

encoder.train()
trans.train()
with torch.no_grad():
    batch = next(iter(train_loader))
    obs, obs_pos, actions, obs_neg = [b.cuda() for b in batch]
    bs = obs.shape[0]

    #z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    #obs_neg = obs_neg.view(-1, *obs_neg.shape[2:])
    #z_neg = encoder(obs_neg)  # b * n x z_dim
    #z_neg = z_neg.view(bs, n_neg, -1) # b x n x z_dim

   # inp = torch.cat((z, actions), dim=1)
   # z_next = trans(inp)  # b x z_dim

 #   dist_pos = torch.norm(z_pos - z_next, dim=1)
  #  dist_neg = torch.norm(z_pos.unsqueeze(1) - z_neg, dim=2)

   # print('pos', dist_pos)
    #print('neg', dist_neg)

    #print('pos dotprod', (z_next * z_pos).sum(1))
    #print('neg dtoprod', torch.sum(z_next.unsqueeze(1) * z_neg, dim=-1).cpu().numpy())

    z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    obs_neg = obs_neg.view(-1, *obs_neg.shape[2:])
    # obs_neg is b x n x 1 x 64 x 64
    z_neg = encoder(obs_neg)  # b * n x z_dim

    inp = torch.cat((z, actions), dim=1)
    z_next = trans(inp)  # b x z_dim

    pos_log_density = (z_next * z_pos).sum(dim=1)
    pos_log_density = pos_log_density.unsqueeze(1)

    z_next = z_next.unsqueeze(1)
    z_neg = z_neg.view(bs, n_neg, -1).permute(0, 2, 1).contiguous() # b x z_dim x n
    neg_log_density = torch.bmm(z_next, z_neg).squeeze(1)  # b x n

    loss = torch.cat((torch.zeros(bs, 1).cuda(), neg_log_density - pos_log_density), dim=1)  # b x n+1
    loss = torch.logsumexp(loss, dim=1).mean()

    print('loss', loss.item())
