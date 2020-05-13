from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import os, time
import random
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

from data_helper import LabeledDataset
from helper import collate_fn, draw_box


class FCN8s(nn.Module):


    def __init__(self, n_class=1):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)


        self.decoder = nn.Sequential(OrderedDict([
          ('dec1_upsample', nn.Upsample(size=(400,400))),

          ('dec2_conv', nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)),
          ('dec2_bn', nn.BatchNorm2d(6)),
          ('dec2_relu', nn.ReLU()),

          ('dec3_conv', nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)),
          ('dec3_bn', nn.BatchNorm2d(6)),
          ('dec3_relu', nn.ReLU()),

          ('dec4_upsample', nn.Upsample(scale_factor=2)),

          ('dec5_conv', nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)),
          ('dec5_bn', nn.BatchNorm2d(6)),
          ('dec5_relu', nn.ReLU()),

          ('dec6_conv', nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)),
          ('dec6_bn', nn.BatchNorm2d(6)),
          ('dec6_relu', nn.ReLU()),

          ('dec7_conv', nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, padding=1))
        ]))


    def forward(self, imgs):
        img1, img2, img3, img4, img5, img6 = \
        imgs[:,0,:,:,:], imgs[:,1,:,:,:], imgs[:,2,:,:,:], imgs[:,3,:,:,:], imgs[:,4,:,:,:], imgs[:,5,:,:,:]

        encodings = []
        for x in (img1, img2, img3, img4, img5, img6):
            h = x
            h = self.relu1_1(self.conv1_1(h))
            h = self.relu1_2(self.conv1_2(h))
            h = self.pool1(h)

            h = self.relu2_1(self.conv2_1(h))
            h = self.relu2_2(self.conv2_2(h))
            h = self.pool2(h)

            h = self.relu3_1(self.conv3_1(h))
            h = self.relu3_2(self.conv3_2(h))
            h = self.relu3_3(self.conv3_3(h))
            h = self.pool3(h)
            pool3 = h  # 1/8

            h = self.relu4_1(self.conv4_1(h))
            h = self.relu4_2(self.conv4_2(h))
            h = self.relu4_3(self.conv4_3(h))
            h = self.pool4(h)
            pool4 = h  # 1/16

            h = self.relu5_1(self.conv5_1(h))
            h = self.relu5_2(self.conv5_2(h))
            h = self.relu5_3(self.conv5_3(h))
            h = self.pool5(h)

            h = self.relu6(self.fc6(h))
            h = self.drop6(h)

            h = self.relu7(self.fc7(h))
            h = self.drop7(h)

            h = self.score_fr(h)
            h = self.upscore2(h)
            upscore2 = h  # 1/16

            h = self.score_pool4(pool4)
            h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            score_pool4c = h  # 1/16

            h = upscore2 + score_pool4c  # 1/16
            h = self.upscore_pool4(h)
            upscore_pool4 = h  # 1/8

            h = self.score_pool3(pool3)
            h = h[:, :,
                  9:9 + upscore_pool4.size()[2],
                  9:9 + upscore_pool4.size()[3]]
            score_pool3c = h  # 1/8

            h = upscore_pool4 + score_pool3c  # 1/8

            h = self.upscore8(h)
            h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
            encodings.append(h)


        feature_map = torch.cat(encodings, dim=1)

        logits = self.decoder(feature_map)
        logits = torch.squeeze(logits, 1)
        probs = torch.sigmoid(logits)

        return logits, probs

def compute_ts_road_map(road_map1, road_map2):
    tp = (road_map1 * road_map2).sum()
    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)


def train(model, train_loader, validation_loader, num_epochs, criterion, optimizer, scheduler, device, save_dir):
    model.to(device)
    prev_train_loss = float('inf')
    prev_test_threat_score = float('-inf')

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
                data = torch.stack(data).to(device)
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

            print(f"Epoch: {epoch + 1}, Test Loss: {test_loss}, Threat Score: {test_threat_score}")
            scheduler.step(test_loss)


if __name__ == "__main__":
    image_folder = '../../data'
    annotation_csv = '../../data/annotation.csv'
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
    train_loader = torch.utils.data.DataLoader(labeled_train_set, batch_size=bsz, shuffle=True, num_workers=2,
                                               collate_fn=collate_fn)

    labeled_validation_set = LabeledDataset(image_folder=image_folder,
                                            annotation_file=annotation_csv,
                                            scene_index=validation_scene_index,
                                            transform=transform,
                                            extra_info=False
                                            )
    validation_loader = torch.utils.data.DataLoader(labeled_validation_set, batch_size=bsz, shuffle=False,
                                                    num_workers=2, collate_fn=collate_fn)

    learning_rates = [0.0001, 0.00001, 0.001, 0.01, 0.1]
    for learning_rate in learning_rates:
        print(f"Learning rate = {learning_rate}")
        model = FCN8s()
        num_epochs = 40

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-5)

        save_dir = os.path.join("road_segmentation_fcn8s_" + str(learning_rate))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        checkpoint_path = os.path.join(save_dir, "train_best_weights.pth")
        if os.path.isfile(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded weights from {checkpoint_path}.")

        print("Starting training!")

        train(model, train_loader, validation_loader, num_epochs, criterion, optimizer, scheduler, device, save_dir)


