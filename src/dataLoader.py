from utils import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from imageio import imread
import random

class myDataset(Dataset):
    def __init__(self, args, phase):

        self.img = os.path.join("./dataset", phase, "img_color")
        self.depth = os.path.join("./dataset", phase, "depth")

        self.img_list = os.listdir(self.img)
        self.depth_list = os.listdir(self.depth)

        self.img_list.sort(key=lambda x: int(x.split('.')[0]))
        self.depth_list.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):

        im_name = self.img_list[idx]
        img_path = os.path.join(self.img, im_name)
        depth_path = os.path.join(self.depth, im_name)

        img = imread(img_path)
        if len(img.shape) < 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = img[..., 1, np.newaxis]
        im = im2float(img, dtype=np.float32)  # convert to double, max 1
        #sRGB to linear
        low_val = im <= 0.04045
        im[low_val] = 25 / 323 * im[low_val]
        im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                       / 211) ** (12 / 5)
        amp = np.sqrt(im)  # to amplitude
        amp = np.transpose(amp, axes=(2, 0, 1))
        amp = resize_keep_aspect(amp, [1072, 1072])
        amp = np.reshape(amp, (1, 1072, 1072))

        depth = imread(depth_path)
        if len(depth.shape) < 3:
            depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        depth = depth[..., 1, np.newaxis]
        depth = im2float(depth, dtype=np.float32)
        depth = np.transpose(depth, axes=(2, 0, 1))
        depth = resize_keep_aspect(depth, [1072, 1072])
        depth = np.reshape(depth, (1, 1072, 1072))
        depth = 1 - depth

        list_k = list(range(3))
        list_ikk = random.sample(list_k, 1)
        ikk = list_ikk[0]

        if ikk == 0:
            mask = np.logical_and(depth >= 0, depth <= 0.33)
        elif ikk == 1:
            mask = np.logical_and(depth > 0.33, depth <= 0.67)
        else:
            mask = np.logical_and(depth > 0.67, depth <= 1)

        return amp, depth, mask, ikk


def data_loader(args):

    train_images = myDataset(args, "train")
    train_loader = DataLoader(
        train_images, batch_size=args.size_of_miniBatches, shuffle=True)

    return train_loader



