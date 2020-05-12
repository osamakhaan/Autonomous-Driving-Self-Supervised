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
import matplotlib.pyplot as plt
import argparse

from data_helper import LabeledDataset, UnlabeledDataset
from helper import collate_fn, draw_box
from models_refactored import RoadSegmentationModel
from utils import compute_ts_road_map


def visualize(model, data_loader, device, save_dir, checkpoint_path, max_save_good = 30, max_save_bad = 30):
    '''
    saves good and bad visualization results
    :param model: model instance
    :param data_loader: dataset loader
    :param device: cpu or cuda
    :param save_dir: path to save visualizations
    :param checkpoint_path: weights to be loaded
    :param max_save_good: maximum number of + results to save
    :param max_save_bad: maximum number of - results to save
    :return:
    '''

    pos_images_saved = 0
    neg_images_saved = 0

    model.to(device)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    good_results_dir = os.path.join(save_dir, "good_results_gt_0.85_ts")
    bad_results_dir = os.path.join(save_dir, "bad_results_lt_0.60_ts")

    if not os.path.isdir(good_results_dir):
        os.mkdir(good_results_dir)

    if not os.path.isdir(bad_results_dir):
        os.mkdir(bad_results_dir)

    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}.")
    else:
        raise RuntimeError(f"No checkpoint file found at {checkpoint_path}!")

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _, target) in enumerate(data_loader):
            data = torch.stack(data).to(device)
            target = torch.stack(target).float().to(device)

            logits, probs = model(data)

            predicted_road_map = probs > 0.5

            for idx in range(target.shape[0]):
                ts_road_map = compute_ts_road_map(predicted_road_map[idx].float(), target[idx])
                if ts_road_map >= 0.85 and pos_images_saved < max_save_good:
                    fig = plt.figure()

                    a = fig.add_subplot(1, 2, 1)
                    predicted_road_map = predicted_road_map.squeeze().detach().cpu().numpy()
                    plt.imshow(predicted_road_map, cmap='gray')
                    plt.axis('off')
                    a.set_title('Predicted Mask')

                    b = fig.add_subplot(1, 2, 2)
                    target = target.squeeze().detach().cpu().numpy()
                    plt.imshow(target, cmap='gray')
                    plt.axis('off')
                    b.set_title('Ground Truth')
                    plt.savefig(os.path.join(good_results_dir, f"{batch_idx}_{idx}_{ts_road_map}.png"))
                    pos_images_saved += 1
                    plt.close()

                elif ts_road_map <= 0.60 and neg_images_saved < max_save_bad:
                    fig = plt.figure()

                    a = fig.add_subplot(1, 2, 1)
                    predicted_road_map = predicted_road_map.squeeze().detach().cpu().numpy()
                    plt.imshow(predicted_road_map, cmap='gray')
                    plt.axis('off')
                    a.set_title('Predicted Mask')

                    b = fig.add_subplot(1, 2, 2)
                    target = target.squeeze().detach().cpu().numpy()
                    plt.imshow(target, cmap='gray')
                    plt.axis('off')
                    b.set_title('Ground Truth')
                    plt.savefig(os.path.join(bad_results_dir, f"{batch_idx}_{idx}_{ts_road_map}.png"))
                    neg_images_saved += 1
                    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int,
                        help="batch size", required=True)
    parser.add_argument("--data", type=str,
                        help="data directory", required=True)
    parser.add_argument("-c", "--checkpoint", type=str,
                        help="checkpoint to load weights", required=True)
    args = parser.parse_args()


    annotation_csv = os.path.join(args.data, "annotation.csv")
    save_dir = "visualizations"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validation_scene_index = np.arange(106, 112)

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5563, 0.6024, 0.6325), (0.3195, 0.3271, 0.3282))
    ])

    labeled_validation_set = LabeledDataset(image_folder=args.data,
                                            annotation_file=annotation_csv,
                                            scene_index=validation_scene_index,
                                            transform=validation_transform,
                                            extra_info=False
                                            )
    validation_loader = torch.utils.data.DataLoader(labeled_validation_set, batch_size=args.bsz, shuffle=False,
                                                    num_workers=2, collate_fn=collate_fn)

    model = RoadSegmentationModel()
    visualize(model, validation_loader, device, save_dir, args.checkpoint)

if __name__ == "__main__":
    main()
