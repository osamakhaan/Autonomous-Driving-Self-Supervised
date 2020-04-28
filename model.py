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

class RoadSegmentationModel(nn.Module):
    def __init__(self, input_channels=3, pretrain=True, mode="mean"):
        '''pretrain is just used for comparison, not for final reporting'''
        super(RoadSegmentationModel, self).__init__()
        
        self.input_channels = input_channels
        self.pretrain = pretrain
        self.mode = mode
        
        if self.pretrain:
            self.vgg16 = models.vgg16(pretrained=True)
            
        if self.mode == "mean" or self.mode == "attention":
            self.decoder_input_channels = 512
        elif self.mode == "concatenate":
            self.decoder_input_channels = 512*6
        else:
            raise ValueError('Invalid Mode. Please select attention, mean or concatenate.')
        
        
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
            
            
          ('enc4_conv', nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)),
          ('enc4_bn', nn.BatchNorm2d(128)),
          ('enc4_relu', nn.ReLU()),  
          ('enc4_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
            
            
            
          ('enc5_conv', nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)),
          ('enc5_bn', nn.BatchNorm2d(256)),
          ('enc5_relu', nn.ReLU()), 
            
            
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
            
            
          ('enc12_conv', nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)),
          ('enc12_bn', nn.BatchNorm2d(512)),
          ('enc12_relu', nn.ReLU()), 
            
          ('enc13_conv', nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)),
          ('enc13_bn', nn.BatchNorm2d(512)),
          ('enc13_relu', nn.ReLU()), 
          ('enc13_maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        
        
        if self.pretrain:
            self.init_vgg_weigts()
            print("Loaded VGG16 weights into encoder layers.")
        
        
        self.decoder = nn.Sequential(OrderedDict([
          ('dec1_conv_tr', nn.ConvTranspose2d(in_channels=self.decoder_input_channels,out_channels=512,stride=3,kernel_size=(5,3),
                                           padding=1,output_padding=(1,0))),
          ('dec1_bn', nn.BatchNorm2d(512)),
          ('dec1_relu', nn.ReLU()),

            
          ('dec2_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)),
          ('dec2_bn', nn.BatchNorm2d(512)),
          ('dec2_relu', nn.ReLU()),
            
            
          ('dec3_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)),
          ('dec3_bn', nn.BatchNorm2d(512)),
          ('dec3_relu', nn.ReLU()),
           
           
          ('dec4_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=512,stride=2,kernel_size=3,
                                           padding=1,output_padding=1)),
          ('dec4_bn', nn.BatchNorm2d(512)),
          ('dec4_relu', nn.ReLU()),
            
            
            
          ('dec5_conv_tr', nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3, padding=1)),
          ('dec5_bn', nn.BatchNorm2d(512)),
          ('dec5_relu', nn.ReLU()),
            
            
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
            
            
          ('dec9_conv_tr', nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3, padding=1)),
          ('dec9_bn', nn.BatchNorm2d(128)),
          ('dec9_relu', nn.ReLU()),
           
           
          ('dec10_conv_tr', nn.ConvTranspose2d(in_channels=128,out_channels=128,stride=2,kernel_size=3,
                                           padding=1,output_padding=1)),
          ('dec10_bn', nn.BatchNorm2d(128)),
          ('dec10_relu', nn.ReLU()),
            
            
            
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
           
           
          ('dec14_conv_tr', nn.ConvTranspose2d(in_channels=32,out_channels=32,stride=2,kernel_size=3,
                                           padding=1,output_padding=1)),
          ('dec14_bn', nn.BatchNorm2d(32)),
          ('dec14_relu', nn.ReLU()),
            
            
            
          ('dec15_conv_tr', nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=3, padding=1))
        ]))

            # Note sigmoid is applied directly in the loss function

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
            
        logits = self.decoder(feature_map)
        logits = torch.squeeze(logits, 1)
        probs = F.sigmoid(logits)

        return logits, probs


    def init_vgg_weigts(self):
        
        assert self.encoder.enc1_conv.weight.size() == self.vgg16.features[0].weight.size()
        self.encoder.enc1_conv.weight.data = self.vgg16.features[0].weight.data
        assert self.encoder.enc1_conv.bias.size() == self.vgg16.features[0].bias.size()
        self.encoder.enc1_conv.bias.data = self.vgg16.features[0].bias.data

        assert self.encoder.enc2_conv.weight.size() == self.vgg16.features[2].weight.size()
        self.encoder.enc2_conv.weight.data = self.vgg16.features[2].weight.data
        assert self.encoder.enc2_conv.bias.size() == self.vgg16.features[2].bias.size()
        self.encoder.enc2_conv.bias.data = self.vgg16.features[2].bias.data

        assert self.encoder.enc3_conv.weight.size() == self.vgg16.features[5].weight.size()
        self.encoder.enc3_conv.weight.data = self.vgg16.features[5].weight.data
        assert self.encoder.enc3_conv.bias.size() == self.vgg16.features[5].bias.size()
        self.encoder.enc3_conv.bias.data = self.vgg16.features[5].bias.data

        assert self.encoder.enc4_conv.weight.size() == self.vgg16.features[7].weight.size()
        self.encoder.enc4_conv.weight.data = self.vgg16.features[7].weight.data
        assert self.encoder.enc4_conv.bias.size() == self.vgg16.features[7].bias.size()
        self.encoder.enc4_conv.bias.data = self.vgg16.features[7].bias.data

        assert self.encoder.enc5_conv.weight.size() == self.vgg16.features[10].weight.size()
        self.encoder.enc5_conv.weight.data = self.vgg16.features[10].weight.data
        assert self.encoder.enc5_conv.bias.size() == self.vgg16.features[10].bias.size()
        self.encoder.enc5_conv.bias.data = self.vgg16.features[10].bias.data

        assert self.encoder.enc6_conv.weight.size() == self.vgg16.features[12].weight.size()
        self.encoder.enc6_conv.weight.data = self.vgg16.features[12].weight.data
        assert self.encoder.enc6_conv.bias.size() == self.vgg16.features[12].bias.size()
        self.encoder.enc6_conv.bias.data = self.vgg16.features[12].bias.data

        assert self.encoder.enc7_conv.weight.size() == self.vgg16.features[14].weight.size()
        self.encoder.enc7_conv.weight.data = self.vgg16.features[14].weight.data
        assert self.encoder.enc7_conv.bias.size() == self.vgg16.features[14].bias.size()
        self.encoder.enc7_conv.bias.data = self.vgg16.features[14].bias.data

        assert self.encoder.enc8_conv.weight.size() == self.vgg16.features[17].weight.size()
        self.encoder.enc8_conv.weight.data = self.vgg16.features[17].weight.data
        assert self.encoder.enc8_conv.bias.size() == self.vgg16.features[17].bias.size()
        self.encoder.enc8_conv.bias.data = self.vgg16.features[17].bias.data

        assert self.encoder.enc9_conv.weight.size() == self.vgg16.features[19].weight.size()
        self.encoder.enc9_conv.weight.data = self.vgg16.features[19].weight.data
        assert self.encoder.enc9_conv.bias.size() == self.vgg16.features[19].bias.size()
        self.encoder.enc9_conv.bias.data = self.vgg16.features[19].bias.data

        assert self.encoder.enc10_conv.weight.size() == self.vgg16.features[21].weight.size()
        self.encoder.enc10_conv.weight.data = self.vgg16.features[21].weight.data
        assert self.encoder.enc10_conv.bias.size() == self.vgg16.features[21].bias.size()
        self.encoder.enc10_conv.bias.data = self.vgg16.features[21].bias.data

        assert self.encoder.enc11_conv.weight.size() == self.vgg16.features[24].weight.size()
        self.encoder.enc11_conv.weight.data = self.vgg16.features[24].weight.data
        assert self.encoder.enc11_conv.bias.size() == self.vgg16.features[24].bias.size()
        self.encoder.enc11_conv.bias.data = self.vgg16.features[24].bias.data

        assert self.encoder.enc12_conv.weight.size() == self.vgg16.features[26].weight.size()
        self.encoder.enc12_conv.weight.data = self.vgg16.features[26].weight.data
        assert self.encoder.enc12_conv.bias.size() == self.vgg16.features[26].bias.size()
        self.encoder.enc12_conv.bias.data = self.vgg16.features[26].bias.data

        assert self.encoder.enc13_conv.weight.size() == self.vgg16.features[28].weight.size()
        self.encoder.enc13_conv.weight.data = self.vgg16.features[28].weight.data
        assert self.encoder.enc13_conv.bias.size() == self.vgg16.features[28].bias.size()
        self.encoder.enc13_conv.bias.data = self.vgg16.features[28].bias.data
        
        
        
        
if __name__ == "__main__":
    model = RoadSegmentationModel(input_channels=3, pretrain=False, mode="concatenate")
    print(model)
    