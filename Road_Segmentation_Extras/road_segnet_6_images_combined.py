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


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_channels = input_channels

        self.vgg16 = models.vgg16(pretrained=False)


        # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=self.input_channels,
                                                          out_channels=64,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(64)
                                                ])
        self.encoder_conv_01 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=64,
                                                          out_channels=64,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(64)
                                                ])
        self.encoder_conv_10 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=64,
                                                          out_channels=128,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(128)
                                                ])
        self.encoder_conv_11 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=128,
                                                          out_channels=128,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(128)
                                                ])
        self.encoder_conv_20 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=128,
                                                          out_channels=256,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(256)
                                                ])
        self.encoder_conv_21 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=256,
                                                          out_channels=256,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(256)
                                                ])
        self.encoder_conv_22 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=256,
                                                          out_channels=256,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(256)
                                                ])
        self.encoder_conv_30 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=256,
                                                          out_channels=512,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(512)
                                                ])
        self.encoder_conv_31 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=512,
                                                          out_channels=512,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(512)
                                                ])
        self.encoder_conv_32 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=512,
                                                          out_channels=512,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(512)
                                                ])
        self.encoder_conv_40 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=512,
                                                          out_channels=512,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(512)
                                                ])
        self.encoder_conv_41 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=512,
                                                          out_channels=512,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(512)
                                                ])
        self.encoder_conv_42 = nn.Sequential(*[
                                                nn.Conv2d(in_channels=512,
                                                          out_channels=512,
                                                          kernel_size=3,
                                                          padding=1),
                                                nn.BatchNorm2d(512)
                                                ])

        # self.init_vgg_weigts()

        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=512,
                                                                   out_channels=512,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(512)
                                               ])
        self.decoder_convtr_41 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=512,
                                                                   out_channels=512,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(512)
                                               ])
        self.decoder_convtr_40 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=512,
                                                                   out_channels=512,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(512)
                                               ])
        self.decoder_convtr_32 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=512,
                                                                   out_channels=512,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(512)
                                               ])
        self.decoder_convtr_31 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=512,
                                                                   out_channels=512,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(512)
                                               ])
        self.decoder_convtr_30 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=512,
                                                                   out_channels=256,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(256)
                                               ])
        self.decoder_convtr_22 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=256,
                                                                   out_channels=256,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(256)
                                               ])
        self.decoder_convtr_21 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=256,
                                                                   out_channels=256,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(256)
                                               ])
        self.decoder_convtr_20 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=256,
                                                                   out_channels=128,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(128)
                                               ])
        self.decoder_convtr_11 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=128,
                                                                   out_channels=128,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(128)
                                               ])
        self.decoder_convtr_10 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=128,
                                                                   out_channels=64,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(64)
                                               ])
        self.decoder_convtr_01 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=64,
                                                                   out_channels=64,
                                                                   kernel_size=3,
                                                                   padding=1),
                                                nn.BatchNorm2d(64)
                                               ])
        self.decoder_convtr_00 = nn.Sequential(*[
                                                nn.ConvTranspose2d(in_channels=64,
                                                                   out_channels=self.output_channels,
                                                                   kernel_size=3,
                                                                   padding=1)
                                               ])

        self.decoder = nn.Sequential(OrderedDict([
            ('dec1_upsample', nn.Upsample(size=(400, 400))),

            ('dec2_conv', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)),
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
        """
        Forward pass `input_img` through the network
        """

        # Encoder Stage - 1
        img1, img2, img3, img4, img5, img6 = imgs[:, 0, :, :, :], imgs[:, 1, :, :, :], imgs[:, 2, :, :, :],\
                                             imgs[:, 3, :, :, :], imgs[:, 4, :, :,:], imgs[:, 5, :, :, :]

        encodings = []

        for input_img in (img1, img2, img3, img4, img5, img6):
            dim_0 = input_img.size()
            x_00 = F.relu(self.encoder_conv_00(input_img))
            x_01 = F.relu(self.encoder_conv_01(x_00))
            x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)

            # Encoder Stage - 2
            dim_1 = x_0.size()
            x_10 = F.relu(self.encoder_conv_10(x_0))
            x_11 = F.relu(self.encoder_conv_11(x_10))
            x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)

            # Encoder Stage - 3
            dim_2 = x_1.size()
            x_20 = F.relu(self.encoder_conv_20(x_1))
            x_21 = F.relu(self.encoder_conv_21(x_20))
            x_22 = F.relu(self.encoder_conv_22(x_21))
            x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)

            # Encoder Stage - 4
            dim_3 = x_2.size()
            x_30 = F.relu(self.encoder_conv_30(x_2))
            x_31 = F.relu(self.encoder_conv_31(x_30))
            x_32 = F.relu(self.encoder_conv_32(x_31))
            x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)

            # Encoder Stage - 5
            dim_4 = x_3.size()
            x_40 = F.relu(self.encoder_conv_40(x_3))
            x_41 = F.relu(self.encoder_conv_41(x_40))
            x_42 = F.relu(self.encoder_conv_42(x_41))
            x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)

            # Decoder

            dim_d = x_4.size()

            # Decoder Stage - 5
            x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
            x_42d = F.relu(self.decoder_convtr_42(x_4d))
            x_41d = F.relu(self.decoder_convtr_41(x_42d))
            x_40d = F.relu(self.decoder_convtr_40(x_41d))
            dim_4d = x_40d.size()

            # Decoder Stage - 4
            x_3d = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
            x_32d = F.relu(self.decoder_convtr_32(x_3d))
            x_31d = F.relu(self.decoder_convtr_31(x_32d))
            x_30d = F.relu(self.decoder_convtr_30(x_31d))
            dim_3d = x_30d.size()

            # Decoder Stage - 3
            x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
            x_22d = F.relu(self.decoder_convtr_22(x_2d))
            x_21d = F.relu(self.decoder_convtr_21(x_22d))
            x_20d = F.relu(self.decoder_convtr_20(x_21d))
            dim_2d = x_20d.size()

            # Decoder Stage - 2
            x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
            x_11d = F.relu(self.decoder_convtr_11(x_1d))
            x_10d = F.relu(self.decoder_convtr_10(x_11d))
            dim_1d = x_10d.size()

            # Decoder Stage - 1
            x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
            x_01d = F.relu(self.decoder_convtr_01(x_0d))
            x_00d = F.relu(self.decoder_convtr_00(x_01d))
            dim_0d = x_00d.size()
            encodings.append(x_00d)

        feature_map = torch.cat(encodings, dim=1)




        logits = self.decoder(x_00d)
        logits = torch.squeeze(logits, 1)
        # x_softmax = F.softmax(x_00d, dim=1
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

    learning_rates = [0.001, 0.0001, 0.01, 0.1]
    for learning_rate in learning_rates:
        print(f"Learning Rate = {learning_rate}")
        model = SegNet(3,1)
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


