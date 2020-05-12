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

import matplotlib
import matplotlib.pyplot as plt

from code.data_helper import LabeledDataset
from code.helper import collate_fn, draw_box
from model import RoadSegmentationModel


def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()
    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)


def evaluate(model, data_loader, criterion, device, save_dir, checkpoint_path, thresholds=[0.5], show_images=False):
    
    model.to(device)
 
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}.") 
    else:
        raise ValueError(f"No checkpoint file found at {checkpoint_path}!")
    
    model.eval()
    test_loss = 0
    total_ts_road_map = [0]*len(thresholds)
    total = 0

    for batch_idx, (data, _, target) in enumerate(data_loader):
        total += 1 # batch size = 1
        data = torch.stack(data).to(device)
        target = torch.stack(target).float().to(device)

        logits, probs = model(data)
        loss = criterion(logits, target).item()
        test_loss += loss
        
        # create binary road image
        for idx, threshold in enumerate(thresholds):
            predicted_road_map = probs > threshold
            ts_road_map = compute_ts_road_map(predicted_road_map.float(), target)
            total_ts_road_map[idx] += ts_road_map
        
            if show_images:
                print(f"Threshold = {threshold}")
                fig = plt.figure()

                a = fig.add_subplot(1,2,1)
                predicted_road_map = predicted_road_map.squeeze().detach().cpu().numpy()
                plt.imshow(predicted_road_map, cmap='gray')
                a.set_title('Predicted Mask')

                a = fig.add_subplot(1,2,2)
                target = target.squeeze().detach().cpu().numpy()
                plt.imshow(target, cmap='gray')
                a.set_title('Ground Truth')


    test_loss = test_loss/total
    threat_score = total_ts_road_map
    for i in range(len(total_ts_road_map)):
        threat_score[i]/=total
    
    return test_loss, threat_score



if __name__ == "__main__":
    image_folder = 'data'
    annotation_csv = 'data/annotation.csv'
    train_scene_index = np.arange(106, 128)
    validation_scene_index = np.arange(128, 134)
    transform = torchvision.transforms.ToTensor()
    bsz = 1
    
    labeled_validation_set = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=validation_scene_index,
                                      transform=transform,
                                      extra_info=False
                                     )
    validation_loader = torch.utils.data.DataLoader(labeled_validation_set, batch_size=bsz, shuffle=True, num_workers=2, collate_fn=collate_fn)
    save_dir = "predictions_road_segmentation_concatenate"
     
    weights_dirs = ["weights_road_segmentation_concatenate_no_pretrain_4"]
    modes = ["concatenate"]
    
    for weights_dir, mode_val in zip(weights_dirs, modes):
        checkpoint_path = os.path.join(weights_dir,"train_best_weights.pth")
        model = RoadSegmentationModel(pretrain=False, mode=mode_val)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.35]).to(device)) # weight calculated across all labeled dataset (pos_weight = number of 0s/number of 1s)
          

        test_loss, threat_score = evaluate(model, validation_loader, criterion, device, save_dir, checkpoint_path)
        print(f"Model: {mode_val}, Test Loss: {test_loss}, Threat Score: {threat_score}")
    


