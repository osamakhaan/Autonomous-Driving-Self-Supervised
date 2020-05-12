from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

import os, time
import random
import numpy as np
import pandas as pd
import argparse

from data_helper import LabeledDataset
from helper import collate_fn, draw_box
from models_refactored import RoadLaneCorrespondenceModel, RoadLaneCorrespondenceModel2
from utils import weights_init

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
    prev_test_accuracy = float('-inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_start = time.time()

        for batch_idx, (sample, target, road_image, extra) in enumerate(train_loader):
            road_images = torch.stack(road_image).float().to(device)
            lane_images = []
            for i in range(road_images.shape[0]):
                lane_images.append(extra[i]['lane_image'])
            lane_images = torch.stack(lane_images).float().to(device)
            # create 2 positive examples (assuming bsz = 2)
            x = []
            y = []
            for i in range(road_images.shape[0]):
                x.append(torch.stack([road_images[i], lane_images[i]]))
                y.append(torch.Tensor([1]))        
            # create 2 negative examples
            for i in range(road_images.shape[0]):
                x.append(torch.stack([road_images[i], lane_images[1-i]]))
                y.append(torch.Tensor([0]))

            x = torch.stack(x).float().to(device)
            y = torch.stack(y).float().to(device)        
            
            optimizer.zero_grad()
            logits, probs = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
                
            train_loss += loss.item() # detaches loss from computational graph

        train_time = time.time() - train_start
        
        if train_loss < prev_train_loss:
            prev_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "train_best_weights.pth"))
            

        print("Epoch #{}\tTrain Loss: {:.8f}\t Time: {:2f}s".format(epoch+1, train_loss, train_time))

        
        # Check performance on the validation set
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_start = time.time()
            total_ts_road_map = 0 # threat score
            total = 0
            correct = 0

            for batch_idx, (sample, target, road_image, extra) in enumerate(validation_loader):
                road_images = torch.stack(road_image).float().to(device)
                lane_images = []
                for i in range(road_images.shape[0]):
                    lane_images.append(extra[i]['lane_image'])
                lane_images = torch.stack(lane_images).float().to(device)
                # create 2 positive examples (assuming bsz = 2)
                x = []
                y = []
                for i in range(road_images.shape[0]):
                    x.append(torch.stack([road_images[i], lane_images[i]]))
                    y.append(torch.Tensor([1]))        
                # create 2 negative examples
                for i in range(road_images.shape[0]):
                    x.append(torch.stack([road_images[i], lane_images[1-i]]))
                    y.append(torch.Tensor([0]))

                x = torch.stack(x).float().to(device)
                y = torch.stack(y).float().to(device)  
                total += x.shape[0]

                logits, probs = model(x)
                loss = criterion(logits, y)
                test_loss += loss.item() # detaches loss from computational graph
                
                y_hat = (probs > 0.5).float()
                correct += y_hat.eq(y.view_as(y_hat)).cpu().sum().item()

            test_time = time.time() - test_start
            
            test_loss = test_loss/total
            accuracy = 100. * correct / total

            if accuracy > prev_test_accuracy:
                prev_test_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(save_dir, "test_best_weights_" + str(accuracy) + ".pth"))

            print(f"Epoch: {epoch+1}, Test Loss: {test_loss}, Accuracy: {accuracy}")
            scheduler.step(test_loss)
            
            
def main():
    '''trains the load and lane correspondence model where the network predicts 1 if both masks correspond and 0 otherwise'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,
                        help="number of epochs", required=True)
    parser.add_argument("--lr", type=float,
                        help="learning rate", required=True)
    parser.add_argument("--data", type=str,
                        help="data directory", required=True)
    args = parser.parse_args()

    annotation_csv = os.path.join(args.data, "annotation.csv")
    # The positive and negative examples creation code below assumes bsz = 2
    bsz = 2

    validation_scene_index = np.arange(106, 112)
    train_scene_index = np.arange(112, 134)

    train_transform = transforms.Compose([
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
    train_loader = torch.utils.data.DataLoader(labeled_train_set, batch_size=bsz, shuffle=True, num_workers=2, collate_fn=collate_fn)


    labeled_validation_set = LabeledDataset(image_folder=args.data,
                                      annotation_file=annotation_csv,
                                      scene_index=validation_scene_index,
                                      transform=validation_transform,
                                      extra_info=True
                                     )
    validation_loader = torch.utils.data.DataLoader(labeled_validation_set, batch_size=bsz, shuffle=False, num_workers=2, collate_fn=collate_fn)
    

    print(f"Learning Rate = {args.lr}")
    model = RoadLaneCorrespondenceModel2()
    model.apply(weights_init)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-5)


    save_dir = os.path.join("road_lane_correspondence_" + str(args.lr))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, "train_best_weights.pth")
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}.")

    train(model, train_loader, validation_loader, args.epochs, criterion, optimizer, scheduler, device, save_dir)

if __name__ == "__main__":
    main()