import os
import cv2
import numpy as np

base_dir = './data/RaFD'
exps = ['angry', 'happy', 'sad', 'surprised']
splits = ['train', 'val']
a = np.random.randint(0, 256, (128, 128, 3))

for split in splits:
    split_dir = os.path.join(base_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    for exp in exps:
        exp_dir = os.path.join(split_dir, exp)
        os.makedirs(exp_dir, exist_ok=True)
        for i in range(32):
            cv2.imwrite(os.path.join(exp_dir, str(i)+'.png'), a)