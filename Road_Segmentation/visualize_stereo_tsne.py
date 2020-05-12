from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable

import os, time
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

from models_refactored import StereoModelVisualization
from stereo_loader import StereoDataset

def visualize(model, validation_loader, device, save_dir, checkpoint_path):
    '''
    visualizes encodings of the stereo pretrain network to determine if different cameras view are grouped separately in the latent space
    :param model: model instance
    :param validation_loader: dataset loader
    :param device: cpu or cuda
    :param save_dir: path to save visualizations
    :param checkpoint_path: weights to be loaded
    '''
    model.to(device)


    if os.path.isfile(checkpoint_path):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(checkpoint_path)
        pretrained_dict = {k: v for k, v in list(pretrained_dict.items()) if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise RuntimeError(f"No checkpoint file found at {checkpoint_path}!")


    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    codes = []
    labels = []

    model.eval()
    with torch.no_grad():

        for batch_idx, (images, indices) in enumerate(validation_loader):
            images = images.to(device)
            indices = indices.to(device)

            outputs = model(images)
            codes.append(outputs.detach())
            labels.append(indices.detach())

        codes = torch.stack(codes, dim=0).squeeze(dim=1).cpu().numpy()
        labels = torch.stack(labels, dim=0).squeeze().cpu().numpy()
        tsne = TSNE().fit_transform(codes)

        tx, ty = tsne[:, 0], tsne[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        # plt.style.use('dark_background')

        for i in range(6):
            sample = labels == i
            plt.scatter(tx[sample], ty[sample], label=i)

        plt.legend(loc=4)
        plt.gca().invert_yaxis()
        plt.title("T-SNE Visualization of Encoded Features after Stereo Pretrain")
        plt.savefig(os.path.join(save_dir, "tsne_vis.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        help="data directory", required=True)
    parser.add_argument("-c", "--checkpoint", type=str,
                        help="checkpoint to load weights", required=True)
    parser.add_argument("--perms", type=int,
                        help="number of permutations", default=700)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unlabeled_val_scene_index = np.arange(80, 106)
    unlabeled_valset = StereoDataset(image_folder=args.data, scene_index=unlabeled_val_scene_index, classes=args.perms, visualize = True)


    bsz = 1
    val_loader = torch.utils.data.DataLoader(dataset=unlabeled_valset,
                                             batch_size=bsz,
                                             shuffle=False,
                                             num_workers=2)

    model = StereoModelVisualization()
    save_dir = "Stereo_TSNE_Visualizations"

    visualize(model, val_loader, device, save_dir, args.checkpoint)


if __name__ == "__main__":
    main()


