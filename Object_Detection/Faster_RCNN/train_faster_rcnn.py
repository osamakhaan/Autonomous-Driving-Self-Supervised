# Inspired from Torchvision Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import argparse

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms

from train_helpers import train_one_epoch, train_one_epoch_modified_loss
from evaluate_faster_rcnn import evaluate
import utils
import transforms as T
from data_helper import LabeledDataset
from helper import compute_ats_bounding_boxes, compute_ts_road_map, collate_fn


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int,
                        help="batch size", required=True)
    parser.add_argument("--epochs", type=int,
                        help="number of epochs", required=True)
    parser.add_argument("--lr", type=float,
                        help="learning rate", required=True)
    parser.add_argument("--data", type=str,
                        help="data directory", required=True)
    parser.add_argument("--loss", type=str,
                        help="normal or modified loss", default="normal")
    args = parser.parse_args()

    assert args.loss in ["normal", "modified"], "can only take values --> normal OR modified"


    annotation_csv = os.path.join(args.data, "annotation.csv")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validation_scene_index = np.arange(106, 112)
    train_scene_index = np.arange(112, 134)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    validation_transform = transforms.Compose([
        transforms.ToTensor()
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

    # num classes = 10 since Faster RCNN expects class 0 to be background
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=10)
    model.train()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    prev_threat_score = float('-inf')
    save_dir = "faster_rcnn"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, "train_best_weights.pth")
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}.")

    for epoch in range(1, args.epochs+1):
        if args.loss == "normal":
            train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)
        elif args.loss == "modified":
            train_one_epoch_modified_loss(model, optimizer, train_loader, device, epoch, print_freq=100)

        torch.save(model.state_dict(), os.path.join(save_dir, "train_best_weights.pth"))

        # evaluate on the test dataset
        threat_score = evaluate(model, validation_loader, device, epoch, prev_threat_score, save_dir)
        if threat_score > prev_threat_score:
            prev_threat_score = threat_score
        # update the learning rate
        lr_scheduler.step()


if __name__ == "__main__":
    train()














