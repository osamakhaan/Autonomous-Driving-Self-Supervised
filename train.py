# TODO Check if pos_weight in criterion is decreasing performance

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import os, time
import random
import numpy as np
import pandas as pd

from code.data_helper import LabeledDataset
from code.helper import collate_fn, draw_box
from model import RoadSegmentationModel


def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()
    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)  

def train(model, train_loader, validation_loader, num_epochs, criterion, optimizer, scheduler, device, save_dir):
    model.to(device)
    prev_train_loss = float('inf')
    prev_test_threat_score = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_start = time.time()

        for batch_idx, (data, _, target) in enumerate(train_loader):
            data = torch.stack(data).to(device)
            target = torch.stack(target).float().to(device)
            
            optimizer.zero_grad()
            logits, probs = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
                
            train_loss += loss.item() # detaches loss from computational graph

        train_time = time.time() - train_start
        
        if train_loss < prev_train_loss:
            prev_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "train_best_weights.pth"))
            

        print("Epoch #{}\tTrain Loss: {:.8f}\t Time: {:2f}s".format(epoch+1, train_loss, train_time))
        
        
        # Check performance on the validation set
        with torch.no_grad():
            test_loss = 0
            test_start = time.time()
            total_ts_road_map = 0 # threat score
            total = 0

            for batch_idx, (data, _, target) in enumerate(validation_loader):
                data = torch.stack(data).to(device)
                target = torch.stack(target).float().to(device)

                logits, probs = model(data)
                loss = criterion(logits, target)
                test_loss += loss.item() # detaches loss from computational graph
                
                predicted_road_map = probs > 0.5
                
                for idx in range(target.shape[0]):
                    total += 1
                    ts_road_map = compute_ts_road_map(predicted_road_map[idx].float(), target[idx])
                    total_ts_road_map += ts_road_map


            test_time = time.time() - test_start
            
            test_loss = test_loss/total
            test_threat_score = total_ts_road_map/total

            if test_threat_score < prev_test_threat_score:
                prev_test_threat_score = test_threat_score
                torch.save(model.state_dict(), os.path.join(save_dir, "test_best_weights.pth"))

            print(f"Epoch: {epoch+1}, Test Loss: {test_loss}, Threat Score: {test_threat_score}")
            scheduler.step(test_threat_score)
            
            
if __name__ == "__main__":
    image_folder = 'data'
    annotation_csv = 'data/annotation.csv'
    train_scene_index = np.arange(106, 128)
    validation_scene_index = np.arange(128, 134)
    transform = torchvision.transforms.ToTensor()
    bsz = 4

    labeled_train_set = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=train_scene_index,
                                      transform=transform,
                                      extra_info=False
                                     )
    train_loader = torch.utils.data.DataLoader(labeled_train_set, batch_size=bsz, shuffle=True, num_workers=2, collate_fn=collate_fn)


    labeled_validation_set = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=validation_scene_index,
                                      transform=transform,
                                      extra_info=False
                                     )
    validation_loader = torch.utils.data.DataLoader(labeled_validation_set, batch_size=bsz, shuffle=True, num_workers=2, collate_fn=collate_fn)
        
    model = RoadSegmentationModel(input_channels=3, pretrain=False, mode="concatenate")
    num_epochs = 100
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.35]).to(device)) # weight calculated across all labeled dataset (pos_weight = number of 0s/number of 1s)
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
#     optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
     
    save_dir = "road_segmentation_weights_concatenate_model"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    checkpoint_path = os.path.join(save_dir,"train_best_weights.pth")    
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}.")    
        
    train(model, train_loader, validation_loader, num_epochs, criterion, optimizer, scheduler, device, save_dir)