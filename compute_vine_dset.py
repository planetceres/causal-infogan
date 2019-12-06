import pickle
import glob
import argparse
import os
from os.path import join
import numpy as np
import itertools
from tqdm import tqdm


def org_images(images):
    t_k = dict()
    for image in images:
        img_split = image.split('_')
        t = int(img_split[-2])
        k = int(img_split[-1].split('.')[0])
        if t not in t_k:
            t_k[t] = dict()
        assert k not in t_k[t]
        t_k[t][k] = image
    return t_k


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
args = parser.parse_args()

runs = glob.glob(join(args.root, 'run*'))
runs = sorted(runs)

neg_samples_same_t = dict()
neg_samples_same_traj = dict()
pos_pairs = []
all_images = []
for run in tqdm(runs):
    action_file = join(run, 'actions.npy')
    np.load(action_file) # just to make sure the path is right

    neg_samples_same_t[run] = dict()
    neg_samples_same_traj[run] = dict()

    images = glob.glob(join(run, '*.png'))
    images = sorted(images)
    all_images.extend(images)
    images = org_images(images)
    for t in itertools.count():
        if t + 1 not in images:
            break
        for k in images[t+1]:
            pos_pairs.append((images[t][0], images[t+1][k], action_file))

        neg_samples_same_t[run][t] = [images[t+1][k] for k in images[t+1]]
        neg_samples_same_traj[run][t] = []
        for t_tmp in images:
            if t_tmp == t or t_tmp == t+1:
                continue
            for k_tmp in images[t_tmp]:
                neg_samples_same_traj[run][t].append(images[t_tmp][k_tmp])

data = dict(pos_pairs=pos_pairs, neg_samples_t=neg_samples_same_t,
            neg_samples_traj=neg_samples_same_traj, all_images=all_images)
with open(join(args.root, 'pos_neg_pairs.pkl'), 'wb') as f:
    pickle.dump(data, f)
