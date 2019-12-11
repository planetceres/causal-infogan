import argparse
import cv2
from tqdm import tqdm
import numpy as np
from os.path import join, exists
import glob
import random
from scipy.ndimage.morphology import grey_dilation

import torch
import torch.utils.data as data
from torchvision.utils import save_image
from torchvision import transforms

from dataset import ImageDataset
from cpc_util import *
from rlpyt.envs.dm_control_env import DMControlEnv


def get_dataloaders():
    dset = ImageDataset(root=args.root, include_state=True, transform=transforms.ToTensor())
    data_loader = data.DataLoader(dset, batch_size=128,
                                  shuffle=True)
    return dset, data_loader


def filter_background(x):
    x[:, (x < 0.3).any(dim=0)] = 0.0
    return x

def dilate(x):
    x = x.squeeze(0).numpy()
    x = grey_dilation(x, size=3)
    x = x[None, :, :]
    return torch.from_numpy(x)


# Assumes input is a [0, 255] numpy array of 64 x 64 x 3
# Processes to [-1, 1] FloatTensor of 1 x 64 x 64
def process_obs(o):
    transform = transforms.Compose([
        filter_background,
        lambda x: x.mean(dim=0)[None, :, :],
        dilate,
        transforms.Normalize((0.5,), (0.5,)),
    ])
    o = torch.FloatTensor(o / 255.).permute(2, 0, 1).contiguous()
    return transform(o)



# [0, 1], 3 x 64 x 64 Float Tensor to [0, 255.] numpy array, 64 x 64 x 3
def unprocess_obs(o):
    o = o.permute(1, 2, 0).contiguous().numpy()
    o *= 255.
    return o


def run_single(model, *args):
    return model(*[a.unsqueeze(0) for a in args]).squeeze(0)


def interpolate(z_start, z_end, alpha):
    if args.interp_type == 'linear':
        return (1 - alpha) * z_start + alpha * z_end
    elif args.interp_type == 'slerp':
        omega = (z_start * z_end).sum() / torch.norm(z_start) / torch.norm(z_end)
        omega = torch.acos(torch.clamp(omega, -1, 1))
        if omega.item() == 0:
            return (1 - alpha) * z_start + alpha * z_end
        a1 = torch.sin((1 - alpha) * omega) / torch.sin(omega)
        a2 = torch.sin(alpha * omega) / torch.sin(omega)
        return a1 * z_start + a2 * z_end
    else:
        raise Exception('Invalid interp_type', args.interp_type)


def visualize_inv(encoder, inv_model, data_loader, env, folder_name, mode, n=10):
    print('visualize_inv', mode)
    assert mode in ['open', 'closed']

    folder_name = join(folder_name, 'visualize_inv_{}'.format(mode))
    if not exists(folder_name):
        os.makedirs(folder_name)

    obs, obs_color, env_states = next(iter(data_loader))
    obs, obs_color, env_states = obs[:2 * n].to(device), obs_color[:2 * n], env_states[:2 * n]
    ostart, oend = obs.chunk(2, dim=0)
    estart, eend = env_states.chunk(2, dim=0)
    estart, eend = estart.numpy(), eend.numpy()
    ocstart, ocend = obs_color.chunk(2, dim=0)

    true_trajectories = []
    with torch.no_grad():
        for i in tqdm(np.arange(n)):
            true_trajectories.append(unprocess_obs(ocstart[i]))

            env.set_state(estart[i])
            o = env.get_obs().pixels
            true_trajectories.append(o)

            zstart = run_single(encoder, ostart[i])
            zend = run_single(encoder, oend[i])
            for t in range(args.n_actions):
                if mode == 'open':
                    z1 = interpolate(zstart, zend, t / args.n_actions)
                    z2 = interpolate(zstart, zend, (t + 1) / args.n_actions)
                    action = run_single(inv_model, z1, z2)
                elif mode == 'closed':
                    ztmp = interpolate(zstart, zend, 1.0)
                    action = run_single(inv_model, zstart, ztmp)
                else:
                    raise Exception('Invalid mode', mode)
                action[2:] *= np.sqrt(2)

                act = action.cpu().numpy().copy()
                loc = (act[:2] * 0.5 + 0.5) * 63
                act = act[2:]
                act[1] = -act[1]
                act = act[[1, 0]]
                act *= 0.3 * 64

                startr, startc = loc
                endr, endc = loc + act
                startr, startc, endr, endc = int(startr), int(startc), int(endr), int(endc)
                true_trajectories[-1] = true_trajectories[-1].astype('uint8')
                cv2.arrowedLine(true_trajectories[-1], (startc, startr), (endc, endr), (255, 0, 0), 2)
                true_trajectories[-1][startr-1:startr+1, startc-1:startc+1, :] = (0, 0, 0)
                true_trajectories[-1] = true_trajectories[-1].astype('float32')

                o = env.step(action.cpu().numpy())[0].pixels
                true_trajectories.append(o)
                if mode == 'closed':
                    zstart = run_single(encoder, process_obs(o).to(device))
            true_trajectories.append(unprocess_obs(ocend[i]))

    true_trajectories = np.stack(true_trajectories, axis=0)
    true_trajectories = true_trajectories / 255.
    true_trajectories = torch.FloatTensor(true_trajectories).permute(0, 3, 1, 2)
    save_image(true_trajectories, join(folder_name, 'seed{}.png'.format(args.seed)),
               nrow=args.n_actions + 3)


def sample_nn_state(encoder, env, state, z, n_trials=100):
    cand_obs, cand_states = [], []
    for _ in range(n_trials):
        env.set_state(state)
        o = env.step(env.action_space.sample())[0].pixels
        cand_obs.append(process_obs(o))
        cand_states.append(env.get_state())
    cand_obs, cand_states = torch.stack(cand_obs).to(device), np.stack(cand_states)
    cand_zs = encoder(cand_obs)
    dists = torch.norm(cand_zs - z.unsqueeze(0), dim=1)
    idx = torch.argmin(dists).item()

    return cand_states[idx]


def visualize_fwd(encoder, fwd_model, dset, env, folder_name, n=10):
    print('visualize_fwd')
    folder_name = join(folder_name, 'visualize_fwd')
    if not exists(folder_name):
        os.makedirs(folder_name)

    # Sample trajectories
    run_paths = glob.glob(join(args.root, 'run*'))
    run_paths = [random.choice(run_paths) for _ in range(n)]

    images = []
    with torch.no_grad():
        for run_path in tqdm(run_paths):
            img_paths = glob.glob(join(run_path, '*.png'))
            img_paths = sorted(img_paths)
            # Add true trajectory
            for img_path in img_paths:
                state = dset.get_item_by_path(img_path)[2].numpy()
                env.set_state(state)
                images.append(env.get_obs().pixels)

            actions = np.load(join(run_path, 'actions.npy'))[:, 0]
            actions = torch.FloatTensor(actions).to(device)
            cur_state = dset.get_item_by_path(img_paths[0])[2].numpy()
            env.set_state(cur_state)
            o = env.get_obs().pixels
            images.append(o)
            z = run_single(encoder, process_obs(o).to(device))
            for i in range(actions.shape[0]):
                znext = run_single(fwd_model, z, actions[i])
                cur_state = sample_nn_state(encoder, env, cur_state, znext)
                env.set_state(cur_state)
                o = env.get_obs().pixels
                images.append(o)
                z = run_single(encoder, process_obs(o).to(device))
    images = np.stack(images, axis=0)
    images = images / 255.
    images = torch.FloatTensor(images).permute(0, 3, 1, 2)
    save_image(images, join(folder_name, 'seed{}.png'.format(args.seed)),
               nrow=images.shape[0] // (2 * n))


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dset, data_loader = get_dataloaders()

    folder_name = join('out', args.name)
    assert exists(folder_name)

    if args.type == 'nce':
        encoder = torch.load(join(folder_name, 'encoder.pt'), map_location=device)
    elif args.type == 'vae':
        encoder = torch.load(join(folder_name, 'vae.pt'), map_location=device)
        obs = next(iter(data_loader))[0].to(device)
        with torch.no_grad():
            obs_recon = encoder.decode(encoder.encode(obs))
        save_image(obs_recon * 0.5 + 0.5, join(folder_name, 'test_vae_visdyn.pngs'))
    else:
        raise Exception('Invalid type', args.type)
    fwd_model = torch.load(join(folder_name, 'fwd_model.pt'), map_location=device)
    inv_model = torch.load(join(folder_name, 'inv_model.pt'), map_location=device)
    encoder.eval()
    fwd_model.eval()
    inv_model.eval()

    env_args = dict(
        domain='rope_sac',
        task='easy',
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=True,
                                  render_kwargs=dict(width=64, height=64,
                                                     camera_id=0)),
        task_kwargs=dict(maxq=True)
    )
    env = DMControlEnv(**env_args)
    env.reset()

    visualize_inv(encoder, inv_model, data_loader, env, folder_name, 'open')
    visualize_inv(encoder, inv_model, data_loader, env, folder_name, 'closed')
    visualize_fwd(encoder, fwd_model, dset, env, folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/rope')
    parser.add_argument('--interp_type', type=str, default='slerp')
    parser.add_argument('--n_actions', type=int, default=10)
    parser.add_argument('--type', type=str, default='nce')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()

    assert args.type in ['nce', 'vae']

    device = torch.device('cuda')
    main()
