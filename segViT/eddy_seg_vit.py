import torch
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv import Config
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    +try to load dataset without converting to png.
    -change somethings in architecture
    -maybe we do not have to scale img try with it.
    """


def set_batch_size(cfg, batch_size):
    BATCH_SIZE = batch_size
    cfg.data.samples_per_gpu = batch_size
    return cfg

def train_model(model, cfg, dataloaders, validate:True):
    """
    this method aims to train model 
    """
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

import mmcv
from mmcv.runner import load_checkpoint
def load_model(config, checkpoint, device, CLASSES, PALETTE):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # print(checkpoint)
        model.CLASSES = CLASSES
        model.PALETTE = PALETTE
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def predict_random_img(model, data_dir, label_dir):
    dirs = os.listdir(data_dir)
    output_dir = "/home/emir/Desktop/dev/img_output/"
    rand_indx = np.random.randint(len(dirs))
    img = data_dir + dirs[rand_indx]
    label = label_dir + dirs[rand_indx][:-3]+"png"
    label_img = mpimg.imread(label)
    print(f"Label Dir {label}")
    print(f"img dir {img}")
    result = inference_segmentor(model=model, imgs=img)
    fig = plt.figure(figsize=(10, 6))
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(result[0])
    plt.title("Pred")
    fig.add_subplot(rows, columns, 2)
    plt.title("Label")
    plt.imshow(label_img)
    

from mmseg.apis.test import single_gpu_test
if __name__ == "__main__":
    seg_Vit_L_cfg = "/home/emir/Desktop/dev/myResearch/src/ViT_Segmentation/segViT/configs/SegViT_L_EddyData.py"
    cp = "/home/emir/Desktop/dev/myResearch/src/ViT_Segmentation/segViT/output/latest.pth"
    valid_dir = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/valid_data_mat/"
    valid_label = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/valid_label/"
    out_dir = "/home/emir/Desktop/dev/img_output/"
    cfg = Config.fromfile(seg_Vit_L_cfg)
    cfg = set_batch_size(cfg, 1) # batch size of 1
    model = init_segmentor(cfg, device=device)
    # model.CLASSES = EddyDatasetREGISTER.CLASSES
    # model.PALETTE = EddyDatasetREGISTER.PALETTE
    # predict_random_img(model=model, data_dir=valid_dir, label_dir=valid_label)
    # datasets = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(dataset=datasets, samples_per_gpu=1, workers_per_gpu=2)
    # eval = single_gpu_test(model=model, data_loader=data_loader,show=True, out_dir=out_dir)

    # print(eval)
    # print(model)
    datasets = build_dataset(cfg.data.train) # with customized pipeline registers we are able to train our model with eddy data
    train_segmentor(model, cfg=cfg, dataset=datasets, validate=True)
