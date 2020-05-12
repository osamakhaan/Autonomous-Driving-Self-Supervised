import torch
import torch.nn as nn


def compute_dataset_statistics(dataset):
    '''
    :param dataset: dataset to compute the statistics fors
    :return: mean and std for each channel. this is later used for normalization.
    '''
    means = []
    stds = []
    for img, _ in dataset:
        m = torch.mean(img, dim=[1, 2])
        s = torch.std(img, dim=[1, 2])
        means.append(m)
        stds.append(s)

    mean = torch.mean(torch.stack(means), dim=0)
    std = torch.mean(torch.stack(stds), dim=0)

    return mean, std

def weights_init(m):
    '''
    initializes Conv2d and ConvTranspose2d according to xavier initialization
    '''
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def compute_ts_road_map(road_map1, road_map2):
    '''
    :param road_map1: predicted road map
    :param road_map2: ground truth road map
    :return: threat score
    '''
    tp = (road_map1 * road_map2).sum()
    return tp * 1.0 / (road_map1.sum() + road_map2.sum() - tp)