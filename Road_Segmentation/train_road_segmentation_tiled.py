'''
This file trains a road segmentation model with tiled image (512 * 918 * 3) as input.
'''


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
import argparse

from data_helper import LabeledDataset, UnlabeledDataset
from helper import collate_fn, draw_box
from models_refactored import RoadSegmentationModelTiledImage
from utils import compute_dataset_statistics, weights_init, compute_ts_road_map


def train(model, train_loader, validation_loader, num_epochs, criterion, optimizer, scheduler, device, save_dir):
    '''trains and evaluates model on each epoch'''
    '''
    :param model: model instance
    :param train_loader: loader for train dataset
    :param validation_loader: loader for validation dataset
    :param num_epochs: number of epochs to train for
    :param criterion: criterion to use for loss
    :param optimizer: optimizer e.g. SGD
    :param scheduler: scheduler e.g. ReduceLROnPlateau
    :param device: cpu or cuda
    :param save_dir: directory to save weights
    '''
    model.to(device)
    prev_train_loss = float('inf')
    prev_test_loss = float('inf')
    prev_test_threat_score = float('-inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_start = time.time()

        for batch_idx, (data, _, target) in enumerate(train_loader):
            images = []
            for img in data:
                tiled_img = torchvision.utils.make_grid(img, nrow=3, padding=0)
                images.append(tiled_img)

            images = torch.stack(images).to(device)
            data = images
            target = torch.stack(target).float().to(device)

            optimizer.zero_grad()
            logits, probs = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # detaches loss from computational graph

        train_time = time.time() - train_start

        if train_loss < prev_train_loss:
            prev_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "train_best_weights.pth"))

        print("Epoch #{}\tTrain Loss: {:.8f}\t Time: {:2f}s".format(epoch + 1, train_loss, train_time))

        # Check performance on the validation set
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_start = time.time()
            total_ts_road_map = 0  # threat score
            total = 0

            for batch_idx, (data, _, target) in enumerate(validation_loader):
                images = []
                for img in data:
                    tiled_img = torchvision.utils.make_grid(img, nrow=3, padding=0)
                    images.append(tiled_img)

                images = torch.stack(images).to(device)
                data = images
                target = torch.stack(target).float().to(device)

                logits, probs = model(data)
                loss = criterion(logits, target)
                test_loss += loss.item()  # detaches loss from computational graph

                predicted_road_map = probs > 0.5

                for idx in range(target.shape[0]):
                    total += 1
                    ts_road_map = compute_ts_road_map(predicted_road_map[idx].float(), target[idx])
                    total_ts_road_map += ts_road_map

            test_time = time.time() - test_start

            test_loss = test_loss / total
            test_threat_score = total_ts_road_map / total

            if test_threat_score > prev_test_threat_score:
                prev_test_threat_score = test_threat_score
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "test_best_weights_" + str(test_threat_score.item()) + ".pth"))

            if test_loss < prev_test_loss:
                prev_test_loss = test_loss
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "test_best_weights.pth"))

            print(f"Epoch: {epoch + 1}, Test Loss: {test_loss}, Threat Score: {test_threat_score}")
            scheduler.step(test_loss)


def main():
    '''
    trains a road segmentation model with tiled image as input
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int,
                        help="batch size", required=True)
    parser.add_argument("--epochs", type=int,
                        help="number of epochs", required=True)
    parser.add_argument("--lr", type=float,
                        help="learning rate", required=True)
    parser.add_argument("--data", type=str,
                        help="data directory", required=True)
    args = parser.parse_args()

    annotation_csv = os.path.join(args.data, "annotation.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validation_scene_index = np.arange(106, 112)
    train_scene_index = np.arange(112, 134)

    # Uncomment the code below for computing dataset statistics. Statistics have been used in transforms.Normalize below.
    # # compute mean and std for normalization (UnlabeledDataset used to get 'image' instead of 'sample')
    # total_scene_index = np.arange(106, 134)
    # dataset = UnlabeledDataset(image_folder, scene_index=total_scene_index, first_dim="image", transform=transforms.ToTensor())
    # mean, std = compute_dataset_statistics(dataset)

    train_transform = transforms.Compose([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5563, 0.6024, 0.6325), (0.3195, 0.3271, 0.3282))
    ])

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5563, 0.6024, 0.6325), (0.3195, 0.3271, 0.3282))
    ])

    labeled_train_set = LabeledDataset(image_folder=args.data,
                                       annotation_file=annotation_csv,
                                       scene_index=train_scene_index,
                                       transform=train_transform,
                                       extra_info=False
                                       )
    train_loader = torch.utils.data.DataLoader(labeled_train_set, batch_size=args.bsz, shuffle=True, num_workers=2,
                                               collate_fn=collate_fn)

    labeled_validation_set = LabeledDataset(image_folder=args.data,
                                            annotation_file=annotation_csv,
                                            scene_index=validation_scene_index,
                                            transform=validation_transform,
                                            extra_info=False
                                            )
    validation_loader = torch.utils.data.DataLoader(labeled_validation_set, batch_size=args.bsz, shuffle=False,
                                                    num_workers=2, collate_fn=collate_fn)


    print(f"Learning Rate = {args.lr}")
    print(f"Device = {device}")

    model = RoadSegmentationModelTiledImage()
    model.apply(weights_init)

    criterion = torch.nn.BCEWithLogitsLoss()

    save_dir = os.path.join("road_segmentation_tiled_image_" + str(args.lr))

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, "train_best_weights.pth")
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, min_lr=1e-5)
    train(model, train_loader, validation_loader, args.epochs, criterion, optimizer, scheduler, device, save_dir)

if __name__ == "__main__":
    main()