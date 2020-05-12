import torch
from shapely.geometry import Polygon
import os
import numpy as np
import torch
from PIL import Image
import pandas as pd

import utils
import transforms as T
from data_helper import LabeledDataset
from helper import compute_ats_bounding_boxes, compute_ts_road_map, collate_fn
from train_helpers import transform_for_training

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, prev_threat_score, save_dir):
    '''
    evaluates performance of Faster RCNN model
    :param model: model instance
    :param data_loader: test data loader
    :param device: cpu or cuda
    :param epoch: epoch number for logging
    :param prev_threat_score: best threat score seen so far
    :param save_dir: directory to save weights
    :return:
    '''
    model.eval()
    model.to(device)
    total_ats_bounding_boxes = 0
    total = 0

    for images, targets, road_image in data_loader:
        images, _, __ = transform_for_training(images, targets, road_image, device)
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        outputs = model(images)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        for idx, output in enumerate(outputs):
            total += 1
            predicted_bounding_boxes = output['boxes']
            recovered_boxes = []
            for bbox in predicted_bounding_boxes:
                recovered_boxes.append(torch.Tensor([[bbox[0], bbox[2], bbox[0], bbox[2]],
                                                   [bbox[1], bbox[1], bbox[3], bbox[3]]]))

            if len(recovered_boxes) != 0:
                recovered_boxes = torch.stack(recovered_boxes)
                predicted_bounding_boxes = recovered_boxes
                predicted_bounding_boxes = (predicted_bounding_boxes - 400) / 10
                ats_bounding_boxes = compute_ats_bounding_boxes(predicted_bounding_boxes, targets[idx]['bounding_box'])
                total_ats_bounding_boxes += ats_bounding_boxes

    threat_score = total_ats_bounding_boxes/total
    print(f"Epoch: {epoch}, Threat Score: {threat_score}")

    if threat_score > prev_threat_score:
        torch.save(model.state_dict(),
                   os.path.join(save_dir, "test_best_weights_" + str(threat_score) + ".pth"))

    return threat_score