import torch.utils.data as data
import h5py
import torch
import numpy as np
import pickle as pkl
import os
import os.path
import gzip
import errno
from tqdm import tqdm
from os.path import join, dirname, basename

from torchvision.datasets.folder import is_image_file, default_loader, \
    IMG_EXTENSIONS, DatasetFolder
from torchvision.datasets.utils import download_url


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


class MergedDataset(data.Dataset):
    """
    Merged multiple datasets into one. Sample together.
    """

    def __init__(self, *datasets):
        self.datasets = datasets
        assert all(len(d) == self.__len__() for d in self.datasets)

    def __getitem__(self, index):
        return [d[index] for d in self.datasets]

    def __len__(self):
        return len(self.datasets[0])

    def __repr__(self):
        fmt_str = ''
        for dataset in self.datasets:
            fmt_str += dataset.__repr__() + '\n'
        return fmt_str


def make_dataset(dir, class_to_idx):
    actions = []
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            # if root[-2:] not in ['66', '67', '68']:
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                if fname == 'actions.npy':
                    path = os.path.join(root, fname)
                    actions.append(np.load(path))
                    actions[-1][:, -1] = 1.0
                    actions[-1][-1, -1] = 0.0

    return images, np.concatenate(actions, axis=0)


def make_pair(imgs, resets, k, get_img, root):
    """
    Return a list of image pairs. The pair is picked if they are k steps apart,
    and there is no reset from the first to the k-1 frames.
    Cases:
        If k = -1, we just randomly pick two images.
        If k >= 0, we try to load img pairs that are k frames apart.
    """
    if k < 0:
        return list(zip(imgs, np.random.permutation(imgs)))

    filename = os.path.join(root, 'rope_pairs_%d.pkl' % k)
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)

    image_pairs = []
    pbar = tqdm(total=len(imgs))
    for i, img in enumerate(imgs):
        if np.sum(resets[i:i + k]) == 0 and (get_img(imgs[i + k][0]) - get_img(img[0])).abs().max() > 0.5:
            next_img = imgs[i + k]
            image_pairs.append(((img[0], np.array(1.0, dtype='float32')),
                                (next_img[0], np.array(1.0, dtype='float32'))))
        pbar.update(1)
    pbar.close()

    with open(filename, 'wb') as f:
        pkl.dump(image_pairs, f)
    return image_pairs


def make_negative_pairs(imgs, resets, root):
    """
    Return a list of negative image pairs. For each pair, the second image is picked
    from a different episode
    """
    filename = os.path.join(root, 'rope_neg_pairs.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)

    idxs = np.argwhere(resets).reshape(-1)
    idxs = np.concatenate(([-1], idxs), axis=0)
    episodes = []
    for i in range(len(idxs) - 1):
        episodes.append((idxs[i] + 1, idxs[i + 1] + 1)) # (inclusive, exclusive)

    neg_image_pairs = []
    pbar = tqdm(total=len(imgs))
    for i, img in enumerate(imgs):
        ep_idx = np.random.randint(0, len(episodes))
        while episodes[ep_idx][0] <= i < episodes[ep_idx][1]:
            ep_idx = np.random.randint(0, len(episodes))

        next_img = np.random.randint(episodes[ep_idx][0], episodes[ep_idx][1])
        next_img = imgs[next_img]
        neg_image_pairs.append(((img[0], np.array(0.0, dtype='float32')),
                                (next_img[0], np.array(0.0, dtype='float32'))))
        pbar.update(1)
    pbar.close()

    with open(filename, 'wb') as f:
        pkl.dump(neg_image_pairs, f)
    return neg_image_pairs


class ImagePairs(data.Dataset):
    """
    A copy of ImageFolder from torchvision. Output image pairs that are k steps apart.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        n_frames_apart (int): The number of frames between the image pairs. Fixed for now.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        img_pairs (list): List of pairs of (image path, class_index) tuples
    """

    url = 'https://drive.google.com/uc?export=download&confirm=ypZ7&id=10xovkLQ09BDvhtpD_nqXWFX-rlNzMVl9'

    def __init__(self, root, include_actions=False, transform=None, target_transform=None,
                 loader=default_loader, n_frames_apart=1, download=False, include_neg=False,
                 thanard_dset=False):
        self.root = root
        self.include_actions = include_actions
        self.thanard_dset = thanard_dset
        if download:
            self.download()

        classes, class_to_idx = DatasetFolder._find_classes(root)
        imgs, actions = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        resets = 1. - actions[:, -1]
        assert len(imgs) == len(resets)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        img_pairs = make_pair(imgs, resets, n_frames_apart, self._get_image, self.root)
        self.img_pairs = img_pairs

        if include_neg:
            neg_img_pairs = make_negative_pairs(imgs, resets, root)
            self.img_pairs += neg_img_pairs

        if self.include_actions:
            if self.thanard_dset:
                self.mean = np.array([121.65736939, 109.50327158,   2.77160466,   0.13424053, 0.87449964])
                self.std = np.array([39.65629748, 26.78163011,  1.78058705,  0.15868182,  0.3312854])
            else:
                self.mean = np.array([0.5, 0.5, 0., 0.])
                self.std = np.array([0.5, 0.5, np.sqrt(2), np.sqrt(2)])


    def _get_image(self, path):
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _check_exists(self):
        return os.path.exists(self.processed_folder)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # for url in self.urls:
        # filename = self.url.rpartition('/')[2]
        filename = "rope"
        file_path = os.path.join(self.raw_folder, filename)
        # import ipdb;ipdb.set_trace()
        download_url(self.url, root=self.raw_folder, filename=filename, md5=None)
        self.extract_gzip(gzip_path=file_path, remove_finished=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        output = []
        for path, target in self.img_pairs[index]:
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.include_actions:
                dir_name = dirname(path)
                actions = np.load(join(dir_name, 'actions.npy'))
                i = int(basename(path).split('.')[0].split('_')[1])
                a = (actions[i] - self.mean) / self.std
                output.append((img, target, torch.FloatTensor(a)))
            else:
                output.append((img, target))
        return output

    def __len__(self):
        return len(self.img_pairs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class NCEVineDataset(data.Dataset):
    def __init__(self, root, n_neg, transform=None, loader=default_loader):
        self.root = root
        with open(join(root, 'pos_neg_pairs.pkl'), 'rb') as f:
            data = pkl.load(f)
        self.nst = data['neg_samples_t']
        self.nstraj = data['neg_samples_traj']
        self.pos_pairs = data['pos_pairs']
        self.image_paths = data['all_images']

        self.images = h5py.File(join(root, 'images.hdf5'), 'r')['images'][:]
        self.img2idx = {self.image_paths[i]: i for i in range(len(self.image_paths))}

        self.transform = transform
        self.loader = loader
        self.n_neg = n_neg
        assert n_neg % 3 == 0

        self.mean = np.array([0.5, 0.5, 0., 0.])
        self.std = np.array([0.5, 0.5, np.sqrt(2), np.sqrt(2)])

    def _get_image(self, path, preloaded=True):
        img = self.images[self.img2idx[path]]
        img = img.astype('float32') / 255
        img = (img - 0.5) / 0.5
        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index):
        obs_file, obs_next_file, action_file = self.pos_pairs[index]
        obs, obs_next = self._get_image(obs_file), self._get_image(obs_next_file)
        actions = np.load(action_file)

        fsplit = obs_next_file.split('_')
        t = int(fsplit[-2])
        k = int(fsplit[-1].split('.')[0])
        action = actions[t-1, k]
        action = (action - self.mean) / self.std

        run = os.path.dirname(obs_file)
        n_per = self.n_neg // 3
        nst = self.nst[run][t-1]
        nstraj = self.nstraj[run][t-1]

        t_idxs = np.random.randint(0, len(nst), size=(n_per,))
        traj_idxs = np.random.randint(0, len(nstraj), size=(n_per,))
        other_idxs = np.random.randint(0, len(self.image_paths), size=(n_per,))

        t_images = [nst[idx] for idx in t_idxs]
        traj_images = [nstraj[idx] for idx in traj_idxs]
        other_images = [self.image_paths[idx] for idx in other_idxs]
        all_images = t_images + traj_images + other_images

        neg_images = torch.stack([self._get_image(img) for img in all_images], dim=0)

        return obs, obs_next, torch.FloatTensor(action), neg_images


class NCEDataset(data.Dataset):
    def __init__(self, root, n_neg, transform=None, loader=default_loader):
        self.root = root
        with open(join(root, 'pos_neg_pairs.pkl'), 'rb') as f:
            data = pkl.load(f)
        self.nst = data['neg_samples_t']
        self.nstraj = data['neg_samples_traj']
        self.pos_pairs = data['pos_pairs']
        self.image_paths = data['all_images']

        self.images = h5py.File(join(root, 'images.hdf5'), 'r')['images'][:]
        self.img2idx = {self.image_paths[i]: i for i in range(len(self.image_paths))}

        self.transform = transform
        self.loader = loader
        self.n_neg = n_neg

        self.mean = np.array([0.5, 0.5, 0., 0.])
        self.std = np.array([0.5, 0.5, np.sqrt(2), np.sqrt(2)])

    def _get_image(self, path, preloaded=True):
        img = self.images[self.img2idx[path]]
        img = img.astype('float32') / 255
        img = (img - 0.5) / 0.5
        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index):
        obs_file, obs_next_file, action_file = self.pos_pairs[index]
        obs, obs_next = self._get_image(obs_file), self._get_image(obs_next_file)
        actions = np.load(action_file)

        fsplit = obs_next_file.split('_')
        t = int(fsplit[-2])
        k = int(fsplit[-1].split('.')[0])
        action = actions[t-1, k]
        action = (action - self.mean) / self.std

        other_idxs = np.random.randint(0, len(self.image_paths), size=(self.n_neg,))
        other_images = [self.image_paths[idx] for idx in other_idxs]

        neg_images = torch.stack([self._get_image(img) for img in other_images], dim=0)

        return obs, obs_next, torch.FloatTensor(action), neg_images
