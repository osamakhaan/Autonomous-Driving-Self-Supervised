'''
This file contains the data loading code for pretraining with the jigsaw task.
Sample usage: python train_jigsaw.py --bsz 4 --epochs 5 --lr 0.001 --data ../../data --perms 1000
'''

# inspired from: https://github.com/bbrattoli/JigsawPuzzlePytorch

import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
]


class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, classes):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data
            classes: number of permutations
        """

        self.image_folder = image_folder
        self.scene_index = scene_index

        self.__image_transformer = transforms.Compose([
            transforms.CenterCrop(255)])

        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor()
        ])

        self.permutations = self.__retrieve_permutations(classes)

    def __retrieve_permutations(self, classes):
        try:
            all_perm = np.load('permutations_jigsaw_%d.npy' % (classes))
        except:
            raise RuntimeError(f"Permutations file not found! Create permutations file using select_permutations.py for {classes} classes!")
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
        sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
        image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

        image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name)

        image = Image.open(image_path)

        image, index = self.__image_transformer(image), index % NUM_IMAGE_PER_SAMPLE

        s = float(image.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = image.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches independently to avoid low level features shortcut
            m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order)


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


if __name__ == "__main__":
    unlabeled_scene_index = np.arange(106)
    image_folder = '../../data'
    unlabeled_trainset = JigsawDataset(image_folder=image_folder, scene_index=unlabeled_scene_index, classes=20)
    data, order = unlabeled_trainset[0]
    print(data.shape)
    print(order)
