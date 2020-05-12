'''
This file is used to pretrain with the jigsaw pretext task or stereo pretext task.
Usage: python train_pretext_tasks.py --bsz 4 --epochs 5 --lr 0.001 --data ../../data --perms 10 -p jigsaw
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

from models_refactored import JigsawModel, StereoModel
from jigsaw_loader import JigsawDataset
from stereo_loader import StereoDataset

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

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_start = time.time()

        for batch_idx, (patches, labels) in enumerate(train_loader):
            patches = patches.to(device)
            labels = labels.to(device)
            train_total += labels.shape[0]

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            pred = outputs.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # detaches loss from computational graph

        train_time = time.time() - train_start
        train_acc = (train_correct / train_total) * 100

        if train_loss < prev_train_loss:
            prev_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "train_best_weights.pth"))

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.6}, Train Accuracy: {train_acc:.3}, Train Time: {train_time:.3} s")

        # Check performance on the validation set
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_correct = 0
            test_total = 0
            test_start = time.time()

            for batch_idx, (patches, labels) in enumerate(validation_loader):
                patches = patches.to(device)
                labels = labels.to(device)
                test_total += labels.shape[0]

                outputs = model(patches)
                loss = criterion(outputs, labels)
                pred = outputs.data.max(1, keepdim=True)[1]
                test_correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

                test_loss += loss.item()  # detaches loss from computational graph

            test_time = time.time() - test_start
            test_acc = (test_correct / test_total) * 100

            if test_loss < prev_test_loss:
                prev_test_loss = test_loss
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "test_best_weights_" + str(test_acc) + ".pth"))

            print(f"Epoch: {epoch+1}, Test Loss: {test_loss:.6}, Test Accuracy: {test_acc:.3}, Test Time: {test_time:.3} s")
            scheduler.step(test_loss)

def main():
    '''
    trains road segmentation model encoder with jigsaw pretext task for some number of permutations
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
    parser.add_argument("--perms", type=int,
                        help="number of permutations", required=True)
    parser.add_argument("-p", "--pretrain", type=str,
                        help="pretrain type --> stereo OR jigsaw")
    args = parser.parse_args()

    assert args.pretrain in ["jigsaw", "stereo"], "pretrain should be --> jigsaw or stereo"

    print(f"Pretrain = {args.pretrain}, Permutations = {args.perms}, Batch Size = {args.bsz}, Num Epochs = {args.epochs}, Learning Rate = {args.lr}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unlabeled_train_scene_index = np.arange(80)
    unlabeled_val_scene_index = np.arange(80, 106)

    if args.pretrain == "jigsaw":
        unlabeled_trainset = JigsawDataset(image_folder=args.data, scene_index=unlabeled_train_scene_index, classes=args.perms)
        unlabeled_valset = JigsawDataset(image_folder=args.data, scene_index=unlabeled_val_scene_index, classes=args.perms)
        model = JigsawModel(classes=args.perms)
        save_dir = os.path.join(f"jigsaw_{args.perms}_" + str(args.lr))
    elif args.pretrain == "stereo":
        unlabeled_trainset = StereoDataset(image_folder=args.data, scene_index=unlabeled_train_scene_index, classes=args.perms)
        unlabeled_valset = StereoDataset(image_folder=args.data, scene_index=unlabeled_val_scene_index, classes=args.perms)
        model = StereoModel(classes=args.perms)
        save_dir = os.path.join(f"stereo_{args.perms}_" + str(args.lr))

    train_loader = torch.utils.data.DataLoader(dataset=unlabeled_trainset,
                                               batch_size=args.bsz,
                                               shuffle=True,
                                               num_workers=2)

    val_loader = torch.utils.data.DataLoader(dataset=unlabeled_valset,
                                             batch_size=args.bsz,
                                             shuffle=False,
                                             num_workers=2)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-5)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, "train_best_weights.pth")
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}.")

    print("Starting Training!")
    train(model, train_loader, val_loader, args.epochs, criterion, optimizer, scheduler, device, save_dir)

if __name__ == "__main__":
    main()
