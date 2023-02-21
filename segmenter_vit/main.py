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
import sys
sys.path.insert(1, '../segViT/')
from ViT_Segmentation.utils.dataset_parser import EddyDatasetREGISTER
# from eddy_seg_vit import predict_random_img
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

try:
    EddyDatasetREGISTER()
except:
    pass
import mmseg
if __name__ == '__main__':
    cfg_path = "/home/emir/Desktop/dev/myResearch/src/ViT_Segmentation/segmenter_vit/configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_eddy_data.py"
    # cp = "/home/emir/Desktop/dev/myResearch/checkpoints_segmenter/iter_160000.pth"
    cfg = Config.fromfile(cfg_path)
    model = build_segmentor(cfg.model)
    datasets = build_dataset(cfg.data.train)
    model.CLASSES = EddyDatasetREGISTER.CLASSES
    model.PALETTE = EddyDatasetREGISTER.PALETTE
    data_dir = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/train_data_mat/"
    label_dir = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/train_label/"
    # predict_random_img(model, data_dir, label_dir)
    print(model)
    print(mmseg.__version__)
    train_segmentor(model=model, dataset=datasets, cfg=cfg, validate=False)