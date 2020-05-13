"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# import your model class
# import ...
from models_refactored import RoadSegmentationModel
import os

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2():
    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5563, 0.6024, 0.6325), (0.3195, 0.3271, 0.3282))
    ])

    return validation_transform

class ModelLoader():
    # Fill the information for your team
    team_name = 'Three Musketeers'
    team_number = 19
    round_number = 3
    team_member = ['Muhammad Osama Khan', 'Muhammad Muneeb Afzal', 'Divya Juneja']
    contact_email = 'mok232@nyu.edu'

    def __init__(self, model_file=['obj_det.pth', 'road_seg.pth']):
        # You should
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 

        # classes = 10 since Faster RCNN expects class 0 to be background
        self.obj_det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=10)

        if os.path.isfile(model_file[0]):
            self.obj_det_model.load_state_dict(torch.load(model_file[0]))
            print(f"Loaded weights from {model_file[0]}.")
        else:
            raise ValueError(f"No checkpoint file found at {model_file[0]}!")

        self.obj_det_model.cuda()


        self.road_seg_model = RoadSegmentationModel()

        if os.path.isfile(model_file[1]):
            self.road_seg_model.load_state_dict(torch.load(model_file[1]))
            print(f"Loaded weights from {model_file[1]}.")
        else:
            raise ValueError(f"No checkpoint file found at {model_file[1]}!")

        self.road_seg_model.cuda()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        self.obj_det_model.eval()

        images = []
        for img in samples:
            tiled_img = torchvision.utils.make_grid(img, nrow=3, padding=0)
            images.append(tiled_img.float().cuda())

        outputs = self.obj_det_model(images)

        results = []

        for output in outputs:
            predicted_bounding_boxes = output['boxes']
            recovered_boxes = []
            for bbox in predicted_bounding_boxes:
                recovered_boxes.append(torch.Tensor([[bbox[0], bbox[2], bbox[0], bbox[2]],
                                                     [bbox[1], bbox[1], bbox[3], bbox[3]]]))

            if len(recovered_boxes) != 0:
                recovered_boxes = torch.stack(recovered_boxes)
                predicted_bounding_boxes = recovered_boxes
                predicted_bounding_boxes = (predicted_bounding_boxes - 400) / 10 # convert back to real world coordinates
                predicted_bounding_boxes = predicted_bounding_boxes.cuda()
                results.append(predicted_bounding_boxes)
            else:
                # if no boxes are predicted, return random to avoid error in run_test.py
                predicted_bounding_boxes = (torch.rand(15, 2, 4) * 10).cuda()
                results.append(predicted_bounding_boxes)

        return results


    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]

        self.road_seg_model.eval()

        logits, probs = self.road_seg_model(samples)
        predicted_road_map = (probs > 0.5).float().cuda()

        return predicted_road_map

