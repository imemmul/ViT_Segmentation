import torch
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv import Config
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import matplotlib.image as pli
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from mmseg.apis import train_segmentor
import numpy as np
from dataset_parser import EddyDatasetREGISTER
print(device)
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmcv.runner import build_runner, build_optimizer
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.models.decode_heads.atm_head import ATMHead
from mmseg.models.losses.atm_loss import ATMLoss
from mmseg.datasets import build_dataloader, build_dataset
import torch.nn as nn
# OKAY SO BATCH SIZE SHOULD BE "1". 
try:
    ATMHead() # initiliaze and add register to mmseg module
except:
    pass
try:
    ATMLoss()
except:
    pass
try:
    EddyDatasetREGISTER()
except:
    pass
    

def dataloader_handler(cfg):
    datasets = build_dataset(cfg.data.train)
    return datasets


def train_model(model, cfg, dataset):
    train_segmentor(model=model, cfg=cfg, dataset=dataset, validate=True, distributed=False)


def build_model(cfg):
    model = build_segmentor(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    return model

def get_len_parameters(model):
    len = 0

    for param in model.parameters():
        len += 1

    return len


def check_frozen_layers(model):
    for param in model.parameters():
        print(param.requires_grad)


def defreeze_layers(model, num_layers):
    """
    this function gives a non freezed layers model.
    """
    count = 0
    size = get_len_parameters(model)
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.parameters():
        count += 1
        if count >= (size - num_layers):
            param.requires_grad = True
    return model

def print_TODO():
    """
    To Learn where [6, 640, 640, 4] where 4 comes from ????? ?
    try img_scale = None
    IN segmentors/base.py check forward function takes 1, 3, 640, 640
    """

if __name__ == "__main__":
    seg_Vit_L_cfg = "./configs/SegViT_L_EddyData.py"
    cp = '/home/emir/dev/segmentation_eddies/downloads/checkpoints/download'
    cfg = Config.fromfile(seg_Vit_L_cfg)
    model = init_segmentor(config=cfg, checkpoint=cp, device=device) # checkpoint loaded.
    #print(model)
    print(model.decode_head.class_embed)
    model.decode_head.class_embed.out_features = 2
    print(model.decode_head.class_embed.out_features)
    model = defreeze_layers(model=model, num_layers=20)
    #check_frozen_layers(model)
    #print(model)
    #print(type(model))
    datasets = build_dataset(cfg.data.train)
    #torch.cuda.synchronize()
    #print(cfg.data.train)
    #print(datasets[0]['img'].size()) # images comes in correct shape
    #print(datasets[0]['gt_semantic_seg'].size())
    train_model(cfg=cfg, model=model, dataset=datasets)