# Some functions inspired from https://github.com/pytorch/vision/tree/master/references/detection

'''
This file contains helper functions for training.
'''

import math
import sys
import time
import torch
import torchvision
import torch.nn.functional as F
import utils

def transform_for_training(sample, targets, road_image, device):
    '''
    :param sample: input images from data loader
    :param targets: bounding boxes and class labels from data loader
    :param road_image: road image from data loader
    :param device: cpu or cuda
    :return: images, modified_targets in the format required by Faster RCNN

    '''
    images = []

    # tile the 6 images into a single image
    for img in sample:
        tiled_img = torchvision.utils.make_grid(img, nrow=3, padding=0)
        images.append(tiled_img.float().to(device))

    modified_targets = []

    for target in targets:
        mod_trgt = {}
        mod_trgt['boxes'] = target['bounding_box']
        mod_trgt['labels'] = target['category']
        mod_trgt['labels'][mod_trgt['labels'] == 0] = 9  # convert 0 to 9 since 0 is background class in Faster RCNN
        mod_trgt['boxes'] = (mod_trgt['boxes'] * 10) + 400 # convert from meters to pixels

        # select min and max x and y coordinates since Faster RCNN uses these to predict bounding boxes
        min_x = mod_trgt['boxes'][:, 0].min(dim=1)[0]
        min_y = mod_trgt['boxes'][:, 1].min(dim=1)[0]
        max_x = mod_trgt['boxes'][:, 0].max(dim=1)[0]
        max_y = mod_trgt['boxes'][:, 1].max(dim=1)[0]

        transformed_boxes = torch.cat((min_x.unsqueeze(1), min_y.unsqueeze(1), max_x.unsqueeze(1),
                                       max_y.unsqueeze(1)), dim=1)

        mod_trgt['boxes'] = transformed_boxes

        # convert to device
        mod_trgt['labels'] = mod_trgt['labels'].to(device)
        mod_trgt['boxes'] = mod_trgt['boxes'].float().to(device)

        modified_targets.append(mod_trgt)

    return images, modified_targets, road_image

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    '''
    helper function to train model for one epoch
    :param model: model instance
    :param optimizer: optimizer e.g. SGD
    :param data_loader: train dataloader
    :param device: cpu or cuda
    :param epoch: epoch number for logging
    :param print_freq: frequency with which to print losses
    '''
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets_actual, road_image in metric_logger.log_every(data_loader, print_freq, header):
        images, targets, _ = transform_for_training(images, targets_actual, road_image, device)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def train_one_epoch_modified_loss(model, optimizer, data_loader, device, epoch, print_freq):
    '''
    helper function to train model for one epoch with modified loss (higher weight for box regression loss)
    :param model: model instance
    :param optimizer: optimizer e.g. SGD
    :param data_loader: train dataloader
    :param device: cpu or cuda
    :param epoch: epoch number for logging
    :param print_freq: frequency with which to print losses
    '''
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets_actual, road_image in metric_logger.log_every(data_loader, print_freq, header):

        images, targets, _ = transform_for_training(images, targets_actual, road_image, device)
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # increase the penalty for loss_box_regression
        losses_reduced += (5*loss_dict_reduced['loss_box_reg'])

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses_reduced.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger
