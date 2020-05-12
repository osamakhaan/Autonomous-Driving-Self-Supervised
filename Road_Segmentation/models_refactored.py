'''
This file contains all the models used for Road Segmentation.
'''

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

from data_helper import LabeledDataset
from helper import collate_fn, draw_box

class RoadSegmentationEncoder(nn.Module):
    def __init__(self, input_channels, mode="concatenate"):
        super(RoadSegmentationEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.mode = mode
            

        self.encoder = nn.Sequential(OrderedDict([
          ('enc1_conv', nn.Conv2d(in_channels=self.input_channels,out_channels=64,kernel_size=3,padding=1)),
          ('enc1_bn', nn.BatchNorm2d(64)),
          ('enc1_relu', nn.ReLU()),
            
          ('enc2_conv', nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)),
          ('enc2_bn', nn.BatchNorm2d(64)),
          ('enc2_relu', nn.ReLU()),
          ('enc2_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
            
          ('enc3_conv', nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)),
          ('enc3_bn', nn.BatchNorm2d(128)),
          ('enc3_relu', nn.ReLU()),
          # ('enc3_dropout', nn.Dropout2d(p=0.2)),
            
            
          ('enc4_conv', nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)),
          ('enc4_bn', nn.BatchNorm2d(128)),
          ('enc4_relu', nn.ReLU()),  
          ('enc4_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            
            
          ('enc5_conv', nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)),
          ('enc5_bn', nn.BatchNorm2d(256)),
          ('enc5_relu', nn.ReLU()),
          # ('enc5_dropout', nn.Dropout2d(p=0.2)),
            
            
          ('enc6_conv', nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)),
          ('enc6_bn', nn.BatchNorm2d(256)),
          ('enc6_relu', nn.ReLU()), 
            
          ('enc7_conv', nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)),
          ('enc7_bn', nn.BatchNorm2d(256)),
          ('enc7_relu', nn.ReLU()), 
          ('enc7_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
            
          ('enc8_conv', nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)),
          ('enc8_bn', nn.BatchNorm2d(512)),
          ('enc8_relu', nn.ReLU()),
          # ('enc8_dropout', nn.Dropout2d(p=0.2)),
            
            
          ('enc9_conv', nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)),
          ('enc9_bn', nn.BatchNorm2d(512)),
          ('enc9_relu', nn.ReLU()),
            
          ('enc10_conv', nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)),
          ('enc10_bn', nn.BatchNorm2d(512)),
          ('enc10_relu', nn.ReLU()), 
          ('enc10_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            
          ('enc11_conv', nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)),
          ('enc11_bn', nn.BatchNorm2d(512)),
          ('enc11_relu', nn.ReLU()),
          # ('enc11_dropout', nn.Dropout2d(p=0.2)),
            
            
          ('enc12_conv', nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)),
          ('enc12_bn', nn.BatchNorm2d(512)),
          ('enc12_relu', nn.ReLU()), 
            
          ('enc13_conv', nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)),
          ('enc13_bn', nn.BatchNorm2d(512)),
          ('enc13_relu', nn.ReLU()), 
          ('enc13_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        
 

    def forward(self, imgs):
        img1, img2, img3, img4, img5, img6 = \
        imgs[:,0,:,:,:], imgs[:,1,:,:,:], imgs[:,2,:,:,:], imgs[:,3,:,:,:], imgs[:,4,:,:,:], imgs[:,5,:,:,:]
           
        encoded_img1 = self.encoder(img1)
        encoded_img2 = self.encoder(img2)
        encoded_img3 = self.encoder(img3)
        encoded_img4 = self.encoder(img4)
        encoded_img5 = self.encoder(img5)
        encoded_img6 = self.encoder(img6)
        
        
        if self.mode == "concatenate":
            feature_map = torch.cat((encoded_img1,encoded_img2,encoded_img3,
                                   encoded_img4,encoded_img5,encoded_img6), dim=1)
        
        elif self.mode == "mean":
            feature_map = (encoded_img1+encoded_img2+encoded_img3+encoded_img4+encoded_img5+encoded_img6)/6
        
        elif self.mode == "attention":
            weight1 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight2 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight3 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight4 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight5 = Variable(torch.tensor(1.0), requires_grad=True).to(device)
            weight6 = Variable(torch.tensor(1.0), requires_grad=True).to(device)

            feature_map = ((weight1*encoded_img1)+(weight2*encoded_img2)+(weight3*encoded_img3)
                          +(weight4*encoded_img4)+(weight5*encoded_img5)+(weight6*encoded_img6))/6
            
        return feature_map

class RoadSegmentationDecoder(nn.Module):
    def __init__(self, input_channels):
        super(RoadSegmentationDecoder, self).__init__()
        
        self.input_channels = input_channels


        self.decoder = nn.Sequential(OrderedDict([
          ('dec1_conv_tr', nn.ConvTranspose2d(in_channels=self.input_channels, out_channels=512,stride=3,kernel_size=(5,3),
                                           padding=1,output_padding=(1,0))),
          ('dec1_bn', nn.BatchNorm2d(512)),
          ('dec1_relu', nn.ReLU()),

            
          ('dec2_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)),
          ('dec2_bn', nn.BatchNorm2d(512)),
          ('dec2_relu', nn.ReLU()),
            
            
          ('dec3_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)),
          ('dec3_bn', nn.BatchNorm2d(512)),
          ('dec3_relu', nn.ReLU()),
          # ('dec3_dropout', nn.Dropout2d(p=0.2)),
           
           
          ('dec4_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=512,stride=2,kernel_size=3,
                                           padding=1,output_padding=1)),
          ('dec4_bn', nn.BatchNorm2d(512)),
          ('dec4_relu', nn.ReLU()),
            
            
            
          ('dec5_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)),
          ('dec5_bn', nn.BatchNorm2d(512)),
          ('dec5_relu', nn.ReLU()),
          # ('dec5_dropout', nn.Dropout2d(p=0.2)),
            
            
          ('dec6_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3, padding=1)),
          ('dec6_bn', nn.BatchNorm2d(256)),
          ('dec6_relu', nn.ReLU()),
           
        
          ('dec7_conv_tr', nn.ConvTranspose2d(in_channels=256,out_channels=256,stride=2,kernel_size=3,
                                           padding=1,output_padding=1)),
          ('dec7_bn', nn.BatchNorm2d(256)),
          ('dec7_relu', nn.ReLU()),
            
            
            
          ('dec8_conv_tr', nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3, padding=1)),
          ('dec8_bn', nn.BatchNorm2d(256)),
          ('dec8_relu', nn.ReLU()),
          # ('dec8_dropout', nn.Dropout2d(p=0.2)),
            
            
          ('dec9_conv_tr', nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3, padding=1)),
          ('dec9_bn', nn.BatchNorm2d(128)),
          ('dec9_relu', nn.ReLU()),

           
           
          ('dec10_conv_tr', nn.ConvTranspose2d(in_channels=128,out_channels=128,stride=2,kernel_size=3,
                                           padding=1,output_padding=1)),
          ('dec10_bn', nn.BatchNorm2d(128)),
          ('dec10_relu', nn.ReLU()),
          # ('dec10_dropout', nn.Dropout2d(p=0.2)),
            
            
            
          ('dec11_conv_tr', nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3, padding=1)),
          ('dec11_bn', nn.BatchNorm2d(64)),
          ('dec11_relu', nn.ReLU()),
           
           
          ('dec12_conv_tr', nn.ConvTranspose2d(in_channels=64,out_channels=64,stride=2,kernel_size=3,
                                           padding=1,output_padding=1)),
          ('dec12_bn', nn.BatchNorm2d(64)),
          ('dec12_relu', nn.ReLU()),
            
            
            
          ('dec13_conv_tr', nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3, padding=1)),
          ('dec13_bn', nn.BatchNorm2d(32)),
          ('dec13_relu', nn.ReLU()),
          # ('dec13_dropout', nn.Dropout2d(p=0.2)),
           
           
          ('dec14_conv_tr', nn.ConvTranspose2d(in_channels=32,out_channels=32,stride=2,kernel_size=3,
                                           padding=1,output_padding=1)),
          ('dec14_bn', nn.BatchNorm2d(32)),
          ('dec14_relu', nn.ReLU()),
            
            
            
          ('dec15_conv_tr', nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=3, padding=1))
        ]))

            # Note sigmoid is applied directly in the loss function

    def forward(self, feature_map):
        
        logits = self.decoder(feature_map)
        logits = torch.squeeze(logits, 1)
        probs = torch.sigmoid(logits)

        return logits, probs
        
        
class RoadSegmentationModel(nn.Module):
    '''
    6 encoders (shared weights) and 1 decoder architecture for road segmentation
    '''
    def __init__(self, encoder_input_channels=3, mode="concatenate"):
        super(RoadSegmentationModel, self).__init__()
        
        self.mode = mode
        if self.mode == "mean" or self.mode == "attention":
            self.decoder_input_channels = 512
        elif self.mode == "concatenate":
            self.decoder_input_channels = 512*6
        else:
            raise ValueError('Invalid Mode. Please select attention, mean or concatenate.')
        
        self.encoder = RoadSegmentationEncoder(encoder_input_channels, mode)
        self.decoder = RoadSegmentationDecoder(self.decoder_input_channels)
        
        
    def forward(self, imgs):
        
        feature_map = self.encoder(imgs)
        logits, probs = self.decoder(feature_map)
        
        return logits, probs


class RoadSegmentationPlusLaneSegmentationModel(nn.Module):
    '''
    6 encoders (shared weights) and 2 decoders (shared weights) model for predicting road segmentation + lane segmentation
    '''
    def __init__(self, encoder_input_channels=3, mode="concatenate"):
        super(RoadSegmentationPlusLaneSegmentationModel, self).__init__()

        self.mode = mode
        if self.mode == "mean" or self.mode == "attention":
            self.decoder_input_channels = 512
        elif self.mode == "concatenate":
            self.decoder_input_channels = 512 * 6
        else:
            raise ValueError('Invalid Mode. Please select attention, mean or concatenate.')

        self.encoder = RoadSegmentationEncoder(encoder_input_channels, mode)
        self.decoder = RoadSegmentationDecoder(self.decoder_input_channels)


    def forward(self, imgs):

        feature_map = self.encoder(imgs)
        logits, probs = self.decoder(feature_map)
        logits_lane, probs_lane = self.decoder(feature_map)

        return logits, probs, logits_lane, probs_lane
    
    
    
class RoadLaneCorrespondenceEncoder(RoadSegmentationEncoder):
    def forward(self, mask):
        feature_map = self.encoder(mask)
        return feature_map
    

class RoadLaneCorrespondenceDecoder(nn.Module):
    def __init__(self):
        super(RoadLaneCorrespondenceDecoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1024,out_channels=128,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64*9*9, 2048)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(2048, 1024)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU()
        
        self.fc6 = nn.Linear(512, 1)
        
    def forward(self, concat_feature_map):
        
        res = self.relu1(self.bn1(self.conv1(concat_feature_map)))
        res = self.relu2(self.bn2(self.conv2(res)))
        res = res.view(res.shape[0], -1)
        res = self.relu3(self.fc3(res))
        res = self.relu4(self.fc4(res))
        res = self.relu5(self.fc5(res))
        logit = self.fc6(res)
        prob = torch.sigmoid(logit)
        
        return logit, prob
    

class RoadLaneCorrespondenceModel(nn.Module):
    '''
    Given a lane mask and road mask, checks if the 2 correspond.
    '''
    def __init__(self, encoder_input_channels=1, mode="concatenate"):
        super(RoadLaneCorrespondenceModel, self).__init__()
        ''''''

        self.encoder = RoadLaneCorrespondenceEncoder(encoder_input_channels)
        self.decoder = RoadLaneCorrespondenceDecoder()
        
        
    def forward(self, masks):
        masks = F.interpolate(masks, size=[300,300]) # reduce the masks size to speed up computation
        road_mask, lane_mask = masks[:,0,:,:].unsqueeze(1), masks[:,1,:,:].unsqueeze(1)
        road_encoding = self.encoder(road_mask)
        lane_encoding = self.encoder(lane_mask)

        concat_feature_map = torch.cat((road_encoding, lane_encoding), dim=1)
        logit, prob = self.decoder(concat_feature_map)

        return logit, prob



class RoadLaneCorrespondenceModel2(nn.Module):
    '''
    Given a lane mask and road mask, checks if the 2 correspond.
    '''
    def __init__(self, classes=1):
        super(RoadLaneCorrespondenceModel2, self).__init__()

        self.encoder1 = JigsawEncoder(input_channels=1)

        self.encoder2 = nn.Sequential(OrderedDict([
            ('2enc1_fc', nn.Linear(512 * 25 * 25, 1024)),
            ('2enc1_relu', nn.ReLU()),
            ('2enc1_dropout', nn.Dropout(p=0.5)),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('dec1_fc', nn.Linear(1024 * 2, 2048)),
            ('dec1_relu', nn.ReLU()),
            ('dec1_dropout', nn.Dropout(p=0.5)),

            ('dec2_fc', nn.Linear(2048, classes)),
        ]))


    def forward(self, masks):
        road_mask, lane_mask = masks[:,0,:,:].unsqueeze(1), masks[:,1,:,:].unsqueeze(1)
        road_tmp = self.encoder1(road_mask)
        lane_tmp = self.encoder1(lane_mask)

        road_encoding = self.encoder2(road_tmp.view(road_tmp.shape[0],-1))
        lane_encoding = self.encoder2(lane_tmp.view(lane_tmp.shape[0],-1))

        concat_feature_map = torch.cat((road_encoding, lane_encoding), dim=1)
        logit = self.decoder(concat_feature_map)
        prob = torch.sigmoid(logit)

        return logit, prob



class RoadSegmentationModelFCNPyTorch(nn.Module):
    '''
    road segmentation model with PyTorch's implementation of FCN as backbone
    '''
    def __init__(self):
        super(RoadSegmentationModelFCNPyTorch, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('fcn_resnet_50', torchvision.models.segmentation.fcn_resnet50(num_classes=1))
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('dec1_upsample', nn.Upsample(size=(400, 400))),

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
            imgs[:, 0, :, :, :], imgs[:, 1, :, :, :], imgs[:, 2, :, :, :], imgs[:, 3, :, :, :], imgs[:, 4, :, :,
                                                                                                :], imgs[:, 5, :, :, :]

        encoded_img1 = F.relu(self.encoder(img1)['out'])
        encoded_img2 = F.relu(self.encoder(img2)['out'])
        encoded_img3 = F.relu(self.encoder(img3)['out'])
        encoded_img4 = F.relu(self.encoder(img4)['out'])
        encoded_img5 = F.relu(self.encoder(img5)['out'])
        encoded_img6 = F.relu(self.encoder(img6)['out'])

        feature_map = torch.cat((encoded_img1, encoded_img2, encoded_img3,
                                 encoded_img4, encoded_img5, encoded_img6), dim=1)

        logits = self.decoder(feature_map)
        logits = torch.squeeze(logits, 1)
        probs = torch.sigmoid(logits)

        return logits, probs


class RoadSegmentationModelTiledImage(nn.Module):
    '''
    road segmentation model with the input as a tiled image of the 6 images
    '''
    def __init__(self):
        super(RoadSegmentationModelTiledImage, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('fcn_resnet_50', torchvision.models.segmentation.fcn_resnet50(num_classes=1))
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('dec1_conv', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)),
            ('dec1_bn', nn.BatchNorm2d(6)),
            ('dec1_relu', nn.ReLU()),

            ('dec2_upsample', nn.Upsample(size=(800, 800))),

            ('dec3_conv', nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)),
            ('dec3_bn', nn.BatchNorm2d(12)),
            ('dec3_relu', nn.ReLU()),

            ('dec4_conv', nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)),
            ('dec4_bn', nn.BatchNorm2d(12)),
            ('dec4_relu', nn.ReLU()),

            ('dec5_conv', nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, padding=1)),
            ('dec5_bn', nn.BatchNorm2d(6)),
            ('dec5_relu', nn.ReLU()),

            ('dec6_conv', nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)),
            ('dec6_bn', nn.BatchNorm2d(6)),
            ('dec6_relu', nn.ReLU()),

            ('dec7_conv', nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, padding=1))
        ]))

    def forward(self, img):
        encoded_img = F.relu(self.encoder(img)['out'])


        logits = self.decoder(encoded_img)
        logits = torch.squeeze(logits, 1)
        probs = torch.sigmoid(logits)

        return logits, probs


class JigsawEncoder(RoadSegmentationEncoder):
    def forward(self, patches):
        feature_map = self.encoder(patches)
        return feature_map



class JigsawModel(nn.Module):
    '''
    model for pretraining with Jigsaw Task. uses same encoder as road segmentation model
    '''
    def __init__(self, classes):
        super(JigsawModel, self).__init__()

        self.encoder1 = JigsawEncoder(input_channels=3)

        self.encoder2 = nn.Sequential(OrderedDict([
            ('2enc1_fc', nn.Linear(512 * 2 * 2, 1024)),
            ('2enc1_relu', nn.ReLU()),
            ('2enc1_dropout', nn.Dropout(p=0.5)),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('dec1_fc', nn.Linear(1024 * 9, 2048)),
            ('dec1_relu', nn.ReLU()),
            ('dec1_dropout', nn.Dropout(p=0.5)),

            ('dec2_fc', nn.Linear(2048, classes)),
        ]))


    # borrowed from: https://github.com/bbrattoli/JigsawPuzzlePytorch
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(9):
            z = self.encoder1(x[i])
            z = self.encoder2(z.view(z.shape[0], -1))
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.decoder(x.view(B, -1))
        x = F.log_softmax(x, dim=1)

        return x


class StereoModel(nn.Module):
    '''
    model for pretraining with stereo task. uses same encoder as road segmentation model
    '''
    def __init__(self, classes):
        super(StereoModel, self).__init__()

        self.encoder1 = JigsawEncoder(input_channels=3)

        self.encoder2 = nn.Sequential(OrderedDict([
            ('2enc1_fc', nn.Linear(512 * 4 * 4, 1024)),
            ('2enc1_relu', nn.ReLU()),
            ('2enc1_dropout', nn.Dropout(p=0.5)),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('dec1_fc', nn.Linear(1024 * 6, 2048)),
            ('dec1_relu', nn.ReLU()),
            ('dec1_dropout', nn.Dropout(p=0.5)),

            ('dec2_fc', nn.Linear(2048, classes)),
        ]))


    # borrowed from: https://github.com/bbrattoli/JigsawPuzzlePytorch
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(6):
            z = self.encoder1(x[i])
            z = self.encoder2(z.view(z.shape[0], -1))
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.decoder(x.view(B, -1))
        x = F.log_softmax(x, dim=1)

        return x


class StereoModelVisualization(nn.Module):
    '''model used to visualize the encodings of the stereo pretrain task'''
    def __init__(self):
        super(StereoModelVisualization, self).__init__()

        self.encoder1 = JigsawEncoder(input_channels=3)

        self.encoder2 = nn.Sequential(OrderedDict([
            ('2enc1_fc', nn.Linear(512 * 4 * 4, 1024)),
            ('2enc1_relu', nn.ReLU()),
            ('2enc1_dropout', nn.Dropout(p=0.5)),
        ]))

    def forward(self, x):
        codes = self.encoder1(x)
        codes = self.encoder2(codes.view(codes.shape[0], -1))

        return codes
        
        
if __name__ == "__main__":
    model = RoadSegmentationModel()
    print(model)
    for i, j in model.named_parameters():
        print(i,j.shape)
    model = RoadLaneCorrespondenceModel()
    print(model)
    model = JigsawModel(classes=300)
    tmp = torch.rand(1,9,3,75,75)
    res = model(tmp)
    print(res.shape)
    model = StereoModel(classes=300)
    tmp = torch.rand(1,6,3,150,150)
    res = model(tmp)
    print(res.shape)
    model = RoadSegmentationModelTiledImage()
    tmp = torch.rand(1,3,768,612)
    res = model(tmp)
    print(res[0].shape)
    model = StereoModelVisualization()
    tmp = torch.rand(1,3,150,150)
    res = model(tmp)
    print(res.shape)
    