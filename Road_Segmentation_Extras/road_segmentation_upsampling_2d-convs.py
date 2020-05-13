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


class RoadSegmentationModel(nn.Module):
    def __init__(self, input_channels=3, mode="mean"):
        '''pretrain is just used for comparison, not for final reporting'''
        super(RoadSegmentationModel, self).__init__()

        self.input_channels = input_channels
        self.mode = mode

        if self.mode == "mean" or self.mode == "attention":
            self.decoder_input_channels = 512
        elif self.mode == "concatenate":
            self.decoder_input_channels = 512 * 6
        else:
            raise ValueError('Invalid Mode. Please select attention, mean or concatenate.')

        self.encoder = nn.Sequential(OrderedDict([
            ('enc1_conv', nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, padding=1)),
            ('enc1_bn', nn.BatchNorm2d(64)),
            ('enc1_relu', nn.ReLU()),

            ('enc2_conv', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)),
            ('enc2_bn', nn.BatchNorm2d(64)),
            ('enc2_relu', nn.ReLU()),
            ('enc2_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('enc3_conv', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)),
            ('enc3_bn', nn.BatchNorm2d(128)),
            ('enc3_relu', nn.ReLU()),

            ('enc4_conv', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)),
            ('enc4_bn', nn.BatchNorm2d(128)),
            ('enc4_relu', nn.ReLU()),
            ('enc4_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('enc5_conv', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('enc5_bn', nn.BatchNorm2d(256)),
            ('enc5_relu', nn.ReLU()),

            ('enc6_conv', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('enc6_bn', nn.BatchNorm2d(256)),
            ('enc6_relu', nn.ReLU()),

            ('enc7_conv', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('enc7_bn', nn.BatchNorm2d(256)),
            ('enc7_relu', nn.ReLU()),
            ('enc7_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('enc8_conv', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ('enc8_bn', nn.BatchNorm2d(512)),
            ('enc8_relu', nn.ReLU()),

            ('enc9_conv', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('enc9_bn', nn.BatchNorm2d(512)),
            ('enc9_relu', nn.ReLU()),

            ('enc10_conv', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('enc10_bn', nn.BatchNorm2d(512)),
            ('enc10_relu', nn.ReLU()),
            ('enc10_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('enc11_conv', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('enc11_bn', nn.BatchNorm2d(512)),
            ('enc11_relu', nn.ReLU()),

            ('enc12_conv', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('enc12_bn', nn.BatchNorm2d(512)),
            ('enc12_relu', nn.ReLU()),

            ('enc13_conv', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('enc13_bn', nn.BatchNorm2d(512)),
            ('enc13_relu', nn.ReLU()),
            ('enc13_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('dec1_upsample', nn.Upsample(scale_factor=2)),

            ('dec2_conv', nn.Conv2d(in_channels=self.decoder_input_channels, out_channels=512,
                                    kernel_size=(5, 7), padding=0)),
            ('dec2_bn', nn.BatchNorm2d(512)),
            ('dec2_relu', nn.ReLU()),

            ('dec3_upsample', nn.Upsample(scale_factor=2)),

            ('dec4_conv', nn.Conv2d(in_channels=512, out_channels=512,
                                    kernel_size=3, padding=2)),
            ('dec4_bn', nn.BatchNorm2d(512)),
            ('dec4_relu', nn.ReLU()),

            ('dec5_upsample', nn.Upsample(scale_factor=2)),

            ('dec6_conv', nn.Conv2d(in_channels=512, out_channels=512,
                                    kernel_size=3, padding=0)),
            ('dec6_bn', nn.BatchNorm2d(512)),
            ('dec6_relu', nn.ReLU()),

            ('dec7_conv', nn.Conv2d(in_channels=512, out_channels=256,
                                    kernel_size=3, padding=1)),
            ('dec7_bn', nn.BatchNorm2d(256)),
            ('dec7_relu', nn.ReLU()),

            ('dec8_upsample', nn.Upsample(scale_factor=2)),

            ('dec9_conv', nn.Conv2d(in_channels=256, out_channels=256,
                                    kernel_size=3, padding=1)),
            ('dec9_bn', nn.BatchNorm2d(256)),
            ('dec9_relu', nn.ReLU()),

            ('dec10_conv', nn.Conv2d(in_channels=256, out_channels=128,
                                     kernel_size=3, padding=1)),
            ('dec10_bn', nn.BatchNorm2d(128)),
            ('dec10_relu', nn.ReLU()),

            ('dec11_upsample', nn.Upsample(scale_factor=2)),

            ('dec12_conv', nn.Conv2d(in_channels=128, out_channels=128,
                                     kernel_size=3, padding=1)),
            ('dec12_bn', nn.BatchNorm2d(128)),
            ('dec12_relu', nn.ReLU()),

            ('dec13_conv', nn.Conv2d(in_channels=128, out_channels=64,
                                     kernel_size=3, padding=1)),
            ('dec13_bn', nn.BatchNorm2d(64)),
            ('dec13_relu', nn.ReLU()),

            ('dec14_upsample', nn.Upsample(scale_factor=2)),

            ('dec15_conv', nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=3, padding=1)),
            ('dec15_bn', nn.BatchNorm2d(64)),
            ('dec15_relu', nn.ReLU()),

            ('dec16_conv', nn.Conv2d(in_channels=64, out_channels=32,
                                     kernel_size=3, padding=1)),
            ('dec16_bn', nn.BatchNorm2d(32)),
            ('dec16_relu', nn.ReLU()),

            ('dec17_upsample', nn.Upsample(scale_factor=2)),

            ('dec18_conv', nn.Conv2d(in_channels=32, out_channels=32,
                                     kernel_size=3, padding=1)),
            ('dec18_bn', nn.BatchNorm2d(32)),
            ('dec18_relu', nn.ReLU()),

            ('dec19_conv', nn.Conv2d(in_channels=32, out_channels=1,
                                     kernel_size=3, padding=1)),
        ]))

        # Note sigmoid is applied directly in the loss function

    def forward(self, imgs):
        img1, img2, img3, img4, img5, img6 = \
            imgs[:, 0, :, :, :], imgs[:, 1, :, :, :], imgs[:, 2, :, :, :], imgs[:, 3, :, :, :], imgs[:, 4, :, :,
                                                                                                :], imgs[:, 5, :, :, :]

        encoded_img1 = self.encoder(img1)
        encoded_img2 = self.encoder(img2)
        encoded_img3 = self.encoder(img3)
        encoded_img4 = self.encoder(img4)
        encoded_img5 = self.encoder(img5)
        encoded_img6 = self.encoder(img6)

        if self.mode == "concatenate":
            feature_map = torch.cat((encoded_img1, encoded_img2, encoded_img3,
                                     encoded_img4, encoded_img5, encoded_img6), dim=1)

        elif self.mode == "mean":
            feature_map = (encoded_img1 + encoded_img2 + encoded_img3 + encoded_img4 + encoded_img5 + encoded_img6) / 6

        elif self.mode == "attention":
            weight1 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight2 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight3 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight4 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight5 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight6 = Variable(torch.tensor(1.0), requires_grad=True).to(device)

            feature_map = ((weight1 * encoded_img1) + (weight2 * encoded_img2) + (weight3 * encoded_img3)
                           + (weight4 * encoded_img4) + (weight5 * encoded_img5) + (weight6 * encoded_img6)) / 6

        logits = self.decoder(feature_map)
        logits = torch.squeeze(logits, 1)
        probs = F.sigmoid(logits)

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

    learning_rates = [0.001, 0.0001, 0.01, 0.1]
    for learning_rate in learning_rates:
        print(f"Learning Rate = {learning_rate}")
        model = RoadSegmentationModel()
        num_epochs = 30

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-5)

        save_dir = os.path.join("road_segnet_6_images_combined_" + str(learning_rate))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        checkpoint_path = os.path.join(save_dir, "train_best_weights.pth")
        if os.path.isfile(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded weights from {checkpoint_path}.")

        print("Starting training!")

        train(model, train_loader, validation_loader, num_epochs, criterion, optimizer, scheduler, device, save_dir)