import sys
import numpy as np
import cv2
import glob
from os.path import join

path = sys.argv[1]
images = glob.glob(join(path, 'img_*_000.png'))
images = sorted(images)
images = [cv2.imread(img) for img in images]

actions = np.load(join(path, 'actions.npy'))[:, 0]

for act, img in zip(actions, images):
    loc = act[:2] * 63
    act = act[2:]
    act[1] = -act[1]
    act = act[[1, 0]]
    act *= 0.1 * 64

    startr, startc = loc
    endr, endc = loc + act
    startr, startc, endr, endc = int(startr), int(startc), int(endr), int(endc)
    cv2.arrowedLine(img, (startc, startr), (endc, endr),  (255, 0, 0), 1)

h, w = 1, 6
padding = 1
full_img = np.zeros((h * 64 + padding * (h - 1), w * 64 + padding * (w - 1), 3), dtype='uint8')

idxs = []
for r in range(h):
    startr = 64 * r + padding * r
    for c in range(w):
        startc = 64 * c + padding * c
        full_img[startr:startr + 64, startc:startc + 64] = images[r * w + c]

cv2.imwrite('visualize_traj.png', full_img)
