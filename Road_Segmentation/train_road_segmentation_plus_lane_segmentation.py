'''
This file trains a network which predicts both road and lane masks simultaneously.
Usage: python train_road_segmentation_plus_lane_segmentation.py --bsz 2 --epochs 5 --lr 0.001 --data ../../data
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
from models_refactored import RoadSegmentationPlusLaneSegmentationModel
from utils import compute_dataset_statistics, weights_init, compute_ts_road_map

def train(model, train_loader, validation_loader, num_epochs, criterion_road, criterion_lane, optimizer, scheduler, device, save_dir):
    '''trains and evaluates model on each epoch'''
    '''
    :param model: model instance
    :param train_loader: loader for train dataset
    :param validation_loader: loader for validation dataset
    :param num_epochs: number of epochs to train for
    :param criterion_road: criterion to use for road loss
    :param criterion_lane: criterion to use for lane loss
    :param optimizer: optimizer e.g. SGD
    :param scheduler: scheduler e.g. ReduceLROnPlateau
    :param device: cpu or cuda
    :param save_dir: directory to save weights
    '''
    model.to(device)
    prev_train_loss = float('inf')
    prev_test_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_start = time.time()

        for batch_idx, (data, _, target, extra) in enumerate(train_loader):
            data = torch.stack(data).to(device)
            target = torch.stack(target).float().to(device)

            assert len(extra) == data.shape[0]
            lane_images = []
            for i in range(data.shape[0]):
                lane_images.append(extra[i]['lane_image'])
            lane_images = torch.stack(lane_images).float().to(device)

            optimizer.zero_grad()
            logits, probs, logits_lane, probs_lane = model(data)
            road_loss = criterion_road(logits, target)
            lane_loss = criterion_lane(logits_lane, lane_images)
            loss = (0.8 * road_loss) + (0.2 * lane_loss)
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
            total_ts_lane_map = 0  # threat score
            total = 0

            for batch_idx, (data, _, target, extra) in enumerate(validation_loader):
                data = torch.stack(data).to(device)
                target = torch.stack(target).float().to(device)

                assert len(extra) == data.shape[0]
                lane_images = []
                for i in range(data.shape[0]):
                    lane_images.append(extra[i]['lane_image'])
                lane_images = torch.stack(lane_images).float().to(device)

                logits, probs, logits_lane, probs_lane = model(data)
                road_loss = criterion_road(logits, target)
                lane_loss = criterion_lane(logits_lane, lane_images)
                loss = (0.8 * road_loss) + (0.8 * lane_loss)

                test_loss += loss.item()  # detaches loss from computational graph

                predicted_road_map = probs > 0.5
                predicted_lane_map = probs_lane > 0.5

                for idx in range(target.shape[0]):
                    total += 1
                    ts_road_map = compute_ts_road_map(predicted_road_map[idx].float(), target[idx])
                    total_ts_road_map += ts_road_map
                    ts_lane_map = compute_ts_road_map(predicted_lane_map[idx].float(), lane_images[idx])
                    total_ts_lane_map += ts_lane_map

            test_time = time.time() - test_start

            test_loss = test_loss / total
            road_ts = total_ts_road_map / total
            lane_ts = total_ts_lane_map / total

            if test_loss < prev_test_loss:
                prev_test_loss = test_loss
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "test_best_weights_" + str(road_ts.item()) + ".pth"))

            print(
                f"Epoch: {epoch + 1}, Test Loss: {test_loss}, Road Threat Score: {road_ts}, Lane Threat Score: {lane_ts}")
            scheduler.step(test_loss)


def main():
    '''
    trains model from scratch or with pretrained weights from jigsaw or stereo
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
    parser.add_argument("-p", "--pretrain", type=str,
                        help="pretrain type --> stereo OR jigsaw")
    parser.add_argument("-c", "--pretrain_checkpoint", type=str,
                        help="checkpoint to load weights for stereo or jigsaw")
    args = parser.parse_args()

    if args.pretrain is not None:
        assert args.pretrain in ["jigsaw", "stereo"], "pretrain should be --> jigsaw or stereo"
        assert args.pretrain_checkpoint is not None, "Enter checkpoint path for pretrained jigsaw or stereo weights"

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
                                       extra_info=True
                                       )
    train_loader = torch.utils.data.DataLoader(labeled_train_set, batch_size=args.bsz, shuffle=True, num_workers=2,
                                               collate_fn=collate_fn)

    labeled_validation_set = LabeledDataset(image_folder=args.data,
                                            annotation_file=annotation_csv,
                                            scene_index=validation_scene_index,
                                            transform=validation_transform,
                                            extra_info=True
                                            )
    validation_loader = torch.utils.data.DataLoader(labeled_validation_set, batch_size=args.bsz, shuffle=False,
                                                    num_workers=2, collate_fn=collate_fn)


    print(f"Learning Rate = {args.lr}")
    print(f"Pretrain = {args.pretrain}")
    print(f"Device = {device}")
    model = RoadSegmentationPlusLaneSegmentationModel()
    model.apply(weights_init)

    # weight calculated across all labeled dataset (pos_weight = number of 0s/number of 1s)
    criterion_road = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.35]).to(device))
    criterion_lane = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([25.5]).to(device))

    if args.pretrain == "jigsaw":
        save_dir = os.path.join("road_seg_plus_lane_seg_jigsaw_" + str(args.lr))
    elif args.pretrain == "stereo":
        save_dir = os.path.join("road_seg_plus_lane_seg_stereo_" + str(args.lr))
    else:
        save_dir = os.path.join("road_seg_plus_lane_seg_no_pretrain_" + str(args.lr))

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if args.pretrain == "jigsaw" or args.pretrain == "stereo":
        if os.path.isfile(args.pretrain_checkpoint):
            model_dict = model.state_dict()
            pretrained_dict = torch.load(args.pretrain_checkpoint)
            filtered_dict = {}
            for key, val in pretrained_dict.items():
                if 'encoder1' in key:
                    # change encoder1.encoder to encoder.encoder
                    split_keys = key.split('.', 1)
                    filtered_dict['encoder.' + split_keys[1]] = val
            pretrained_dict = {k: v for k, v in list(filtered_dict.items()) if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded weights from {args.pretrain_checkpoint}")

            optimizer = torch.optim.Adam([
                {'params': model.encoder.parameters(), 'lr': 1e-5},
                {'params': model.decoder.parameters(), 'lr': args.lr, 'weight_decay': 5e-4}
            ])
            print(f"Training {args.pretrain} with different lrs!")

        else:
            raise RuntimeError(f"Checkpoints file {args.pretrain_checkpoint} not found!")

    else:
        print("Training from scratch!")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, min_lr=1e-5)
    train(model, train_loader, validation_loader, args.epochs, criterion_road, criterion_lane, optimizer, scheduler, device, save_dir)

if __name__ == "__main__":
    main()