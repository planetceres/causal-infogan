import sys
import numpy as np
import cv2
import glob
from os.path import join

path = sys.argv[1]
images = glob.glob(join(path, 'img_*_000.png'))
images = sorted(images)
images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in images]

actions = np.load(join(path, 'actions.npy'))[:, 0]

for act, img in zip(actions, images):
    loc = act[:2] * 63
    act = act[2:]
    act[1] = -act[1]
    act = act[[1, 0]]
    act *= 0.1 * 64

    startr, startc = loc
    endr, endc = loc + act
    cv2.arrowedLine(img, (startc, startr), (endc, endr), (0, 0, 0), 3)
