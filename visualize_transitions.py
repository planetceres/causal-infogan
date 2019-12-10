import sys
import numpy as np
import cv2
import glob
from os.path import join
import random

n = 32
path = sys.argv[1]
images = glob.glob(join(path, 'img_*_*.png'))
images = sorted(images)
images.remove(join(path, 'img_00_000.png'))
chosen = [random.choice(images) for _ in range(n)]

actions = np.load(join(path, 'actions.npy'))
to_disp = []

for img_path in chosen:
    path_split = img_path.split('_')
    t = int(path_split[-2])
    k = int(path_split[-1].split('.')[0])

    act = actions[t - 1, k]
    next_img = cv2.imread(img_path)

    str_t, str_k = str(t - 1), str(k)
    cur_img_path = join(path, 'img_{}_{}.png'.format(str_t.zfill(2), '0'.zfill(3)))
    cur_img = cv2.imread(cur_img_path)

    loc = act[:2] * 63
    act = act[2:]
    act[1] = -act[1]
    act = act[[1, 0]]
    act *= 0.1 * 64

    startr, startc = loc
    endr, endc = loc + act
    startr, startc, endr, endc = int(startr), int(startc), int(endr), int(endc)
    cv2.arrowedLine(cur_img, (startc, startr), (endc, endr),  (255, 0, 0), 1)

    to_disp.extend((cur_img, next_img))

h, w = 8, 8
padding = 1
full_img = np.zeros((h * 64 + padding * (h - 1), w * 64 + padding * (w - 1), 3), dtype='uint8')

idxs = []
for r in range(h):
    startr = 64 * r + padding * r
    for c in range(w):
        startc = 64 * c + padding * c
        full_img[startr:startr + 64, startc:startc + 64] = to_disp[r * w + c]

cv2.imwrite('visualize_transitions.png', full_img)
