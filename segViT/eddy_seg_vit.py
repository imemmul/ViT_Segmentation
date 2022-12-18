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
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        drop_last=True)
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
        '   test_dataloader'
        ]
    })
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    data_loaders = build_dataloader(datasets, **train_loader_cfg)
    return data_loaders, datasets





def train_model(model, cfg, dataset):
    train_segmentor(model=model, cfg=cfg, dataset=dataset, validate=True)


def build_model(cfg):
    model = build_segmentor(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    return model



if __name__ == "__main__":
    seg_Vit_L_cfg = "./configs/SegViT_L_EddyData.py"
    cfg = Config.fromfile(seg_Vit_L_cfg)
    model = build_model(cfg)
    dataloaders, datasets = dataloader_handler(cfg)
    train_model(cfg=cfg, model=model, dataset=datasets)