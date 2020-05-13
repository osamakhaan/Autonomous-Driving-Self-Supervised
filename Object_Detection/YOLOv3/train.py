from __future__ import division

from model import *
from utils.logger import *
from utils.utils import *
from utils.parse_config import *
from utils.data_helper import UnlabeledDataset, LabeledDataset
from utils.helper import collate_fn, draw_box, compute_ats_bounding_boxes
import torch.nn.functional as F
import torchvision
from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/modified_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--image_folder", type=str, default="./data", help="path to image folder")
    parser.add_argument("--annotation_csv", type=str, default="./data/annotation.csv", help="path to annotation csv")
    parser.add_argument("--data_config", type=str, default="config/car_scenes.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--yolov3_weights", default=False, help="initialize from yolov3 weights")
    parser.add_argument("--own_trained_weights", default=False, help="initialize from trained weights")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=288, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--nms", type=float, default=0.8, help="Non-Maximum suppression threshold")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    
    # If specified we start from checkpoint 
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    if opt.yolov3_weights:
        path = "weights/yolov3.weights"
        model.load_darknet_weights(path)
        print(f"Loaded pretrained yolov3 weights from {path}")

    
    if opt.own_trained_weights:
        trained_weights = "checkpoints/yolov3_ckpt_0.pth"
        model.load_state_dict(torch.load(trained_weights))
        print(f"Loaded pretrained own weights from {trained_weights}")

    
    '''We define which scenes to use for training 
    and which scenes to use for validation'''
    
    # train_scene_index = np.arange(106, 128)
    # validation_scene_index = np.arange(128, 134)

    # TODO for testing
    train_scene_index = np.arange(106, 107)
    validation_scene_index = np.arange(128, 129)

    from torchvision.transforms import *

    # TODO transform
    transform = Compose([Resize((288,288)),ToTensor()])

    transform = torchvision.transforms.ToTensor()
    labeled_trainset = LabeledDataset(image_folder=opt.image_folder,
                                  annotation_file=opt.annotation_csv,
                                  scene_index=train_scene_index,
                                  transform=transform,
                                  extra_info=False
    )
    labeled_validation_set = LabeledDataset(image_folder=opt.image_folder,
                                    annotation_file=opt.annotation_csv,
                                    scene_index=validation_scene_index,
                                    transform=transform,
                                    extra_info=False
)
    
    trainloader = torch.utils.data.DataLoader(
        labeled_trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    validationloader = torch.utils.data.DataLoader(
        labeled_validation_set,
        # TODO
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    prev_threat_score = float('-inf')
    prev_train_loss = float('inf')
    prev_test_threat_score = float('-inf')

    for epoch in range(opt.epochs):
        model.train()
        total_samples =0
        train_loss = 0
        start_time = time.time()
        for batch_i, (sample, one_batch_targets, _ ) in enumerate(trainloader):
            
            labels =  preprocess_targets(one_batch_targets,img_width = 800, img_height = 800).float()
            imgs = transform_images(sample).float()

            images = Variable(imgs.to(device))

            targets = Variable(labels.to(device), requires_grad=False)


            batches_done = len(trainloader) * epoch + batch_i
            
            loss, outputs = model(images, targets)

            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # We accumulates the gradients before taking the step 
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()  # detach the loss
            # calculating the estimate of time left 
            epoch_batches_left = len(trainloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))

            if train_loss < prev_train_loss:
                prev_train_loss = train_loss
                torch.save(model.state_dict(), "checkpoints/train_best_weights.pth")

            print(f"[Epoch: {epoch}/{opt.epochs}, Batch: {batch_i}/{len(trainloader)}], Train loss: {train_loss}")

        print(80*"-")
        print(f"Epoch: {epoch+1}, Total Train loss: {train_loss}")

        if epoch % opt.evaluation_interval == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                test_start_time = time.time()
                total_examples = 0
                epoch_test_threat_score =0
                for batch_i, (sample, one_batch_targets, _ ) in enumerate(validationloader):
                    
                    labels =  preprocess_targets(one_batch_targets,img_width = 800, img_height = 800).float()
                    imgs = transform_images(sample).float()

                    # TODO take 6 imagess
                    images = Variable(imgs.to(device))

                    targets = Variable(labels.to(device), requires_grad=False)

                    loss, outputs = model(images, targets)

                    test_loss += loss.item()  # detach the loss

                    outputs = model(images)
                    outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=opt.nms)
                    predicted_boxes = get_prediction_boxes(outputs)
                    for idx in range(len(predicted_boxes)):
                        threat_score =0
                        total_examples+=1
                        threat_score = compute_ats_bounding_boxes(predicted_boxes[idx],one_batch_targets[idx]['bounding_box'])
                        threat_score = threat_score.item()
                        epoch_test_threat_score += threat_score

                        print(f"[Epoch: {epoch}/{opt.epochs}, Batch: {batch_i}/{len(validationloader)}], Sample Idx: {idx}, Test Threat Score: {threat_score}")
            
            
            test_time = time.time() - test_start_time
            epoch_test_threat_score /= total_examples
            test_loss /= total_examples
            my_tensorboard_epoch_log = [("epoch_threat_score", epoch_threat_score)]
            logger.list_of_scalars_summary(my_tensorboard_epoch_log, epoch)
            
            if epoch_test_threat_score > prev_test_threat_score:
                torch.save(model.state_dict(),"checkpoints/test_weights_" + str(epoch_test_threat_score)+"_ep_"+str(epoch) + ".pth")

            print(f"Epoch: {epoch+1}, Test Threat Score: {epoch_test_threat_score}, Test Loss: {test_loss}")
            print(80*"-")
        
        
        
        

