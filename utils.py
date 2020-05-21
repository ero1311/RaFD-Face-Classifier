import matplotlib.pyplot as plt
import numpy as np
import cv2


def imsave(inp, name, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    fig = plt.figure()
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig(name)
    plt.close(fig)