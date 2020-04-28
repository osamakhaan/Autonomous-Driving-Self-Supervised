"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
from model import RoadSegmentationModel
import os

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Three Musketeers'
    round_number = 1
    team_member = ['Muhammad Osama Khan', 'Muhammad Muneeb Afzal', 'Divya Juneja']
    contact_email = 'mok232@nyu.edu'

    def __init__(self, model_file='train_best_weights.pth'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.model = RoadSegmentationModel(pretrain=False, mode="concatenate")
        
        if os.path.isfile(model_file):
            self.model.load_state_dict(torch.load(model_file))
            print(f"Loaded weights from {model_file}.") 
        else:
            raise ValueError(f"No checkpoint file found at {model_file}!")
        
        self.model.cuda()
            
    def get_bounding_boxes(self, samples): # NOT IMPLEMENTED
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return (torch.rand(1, 15, 2, 4) * 10).cuda().double()

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        logits, probs = self.model(samples)
        predicted_road_map = (probs > 0.5).float().cuda()
        
        return predicted_road_map
        #return torch.rand(1, 800, 800) > 0.5
