'''
This file contains the data loading code for the stereo pretext task.
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

import matplotlib.pyplot as plt

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


class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, classes, visualize=False):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data
            classes: number of permutations
            visualize: used for visualizing the encodings of the stereo pretrained network
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.visualize = visualize

        self.__image_transformer = transforms.Compose([
            transforms.CenterCrop(150),
            transforms.Lambda(rgb_jittering),
            transforms.ToTensor()
        ])

        self.permutations = self.__retrieve_permutations(classes)

    def __retrieve_permutations(self, classes):
        try:
            all_perm = np.load('permutations_stereo_%d.npy' % (classes))
        except:
            raise RuntimeError(f"Permutations file not found! Create permutations file using select_permutations.py for {classes} classes!")
        # from range [1,6] to [0,5]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):

        if self.visualize:
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name)

            image = Image.open(image_path)

            return self.__image_transformer(image), index % NUM_IMAGE_PER_SAMPLE

        else:

            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                image = self.__image_transformer(image)
                # Normalize the images independently to avoid low level features shortcut
                m, s = image.view(3, -1).mean(dim=1).numpy(), image.view(3, -1).std(dim=1).numpy()
                s[s == 0] = 1
                norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
                image = norm(image)
                images.append(image)

            order = np.random.randint(len(self.permutations))
            data = [images[self.permutations[order][t]] for t in range(6)]
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
    unlabeled_trainset = StereoDataset(image_folder=image_folder, scene_index=unlabeled_scene_index, classes=20)
    data, order = unlabeled_trainset[0]
    print(data.shape)
    print(order)
    plt.figure()
    plt.imshow(data[0].numpy().transpose(1,2,0))
    plt.savefig('test.jpg')
    print(data[0].min())
    print(data[0].max())