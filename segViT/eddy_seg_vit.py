import torch
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv import Config
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import matplotlib.image as pli
import warnings
from mmcv.utils import build_from_cfg
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
from dataset_parser import EddyDatasetREGISTER, EddyDataset, create_dataloaders
print(device)
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmcv.runner import build_runner, build_optimizer, HOOKS
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.models.decode_heads.atm_head import ATMHead
from mmseg.models.losses.atm_loss import ATMLoss
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger, build_dp, find_latest_checkpoint
from mmseg.core import DistEvalHook, EvalHook, build_optimizer

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
    
BATCH_SIZE = 1


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
    +To Learn where [6, 640, 640, 4] where 4 comes from ????? ?
    +try img_scale = None
    +IN segmentors/base.py check forward function takes 1, 3, 640, 640
    +should convert all rgb images to grayscale
    -try to load dataset without converting to png.
    """


def set_batch_size(cfg, batch_size):
    BATCH_SIZE = batch_size
    cfg.data.samples_per_gpu = batch_size
    return cfg

def train_model(model, cfg, dataloaders, validate:True):
    distributed = False
    logger = get_root_logger(cfg.log_level)
    # The default loader config
    train_dataloader = dataloaders[0]
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # building model (return nn.module)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    #build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    
    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
        
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=None))
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    runner.timestamp = None
    
    if validate:
        # The specific dataloader settings
        val_dataloader = dataloaders[1]
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run([train_dataloader], cfg.workflow)



if __name__ == "__main__":
    seg_Vit_L_cfg = "./configs/SegViT_L_EddyData.py"
    cp = '/home/emir/dev/segmentation_eddies/downloads/checkpoints/download'
    cfg = Config.fromfile(seg_Vit_L_cfg)
    cfg = set_batch_size(cfg, 8)
    model = init_segmentor(config=cfg, checkpoint=cp, device=device) # checkpoint loaded.
    #print(model)
    #Changing output channels in order to fit to EddyData
    print(model.decode_head.class_embed)
    model.decode_head.class_embed.out_features = 2
    print(model.decode_head.class_embed.out_features)
    print(model)
    #model = defreeze_layers(model=model, num_layers=20)
    #check_frozen_layers(model)
    #print(model)
    img_dir = "/home/emir/dev/segmentation_eddies/downloads/data4test/data/"
    annot_dir = "/home/emir/dev/segmentation_eddies/downloads/data4test/label_grayscale/"
    train_data = EddyDataset(input_image_dir=img_dir, mask_image_dir=annot_dir, split=0.85, train_bool=True)
    valid_data = EddyDataset(input_image_dir=img_dir, mask_image_dir=annot_dir, split=0.85, train_bool=False)
    train_dataloader, valid_dataloader, class_names = create_dataloaders(train_data=train_data, test_data=valid_data, is_data_exist=True, batch_size=BATCH_SIZE)
    dataloaders = [train_dataloader, valid_dataloader]
    #train_model(model=model, cfg=cfg, dataloaders=dataloaders, validate=True)
    # print(cfg.data.train)
    datasets = build_dataset(cfg.data.train)
    #train_segmentor(model, cfg=cfg, dataset=datasets)
