from mmseg.apis.test import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint, wrap_fp16_model
import torch
import warnings
from mmseg.utils import build_dp
from mmseg.apis import inference_segmentor
import matplotlib.image as mpimg
import numpy as np
import os

def run_test(cfg, distributed, cp_path):
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('test_dataloader', {})
        }
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, cp_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        results = single_gpu_test(
            model,
            data_loader,)
            # args.show,
            # args.show_dir,
            # False,
            # args.opacity,
            # pre_eval=args.eval is not None and not eval_on_format_results,
            # format_only=args.format_only or eval_on_format_results,
            # format_args=eval_kwargs)
        metric = dataset.evaluate(results)
        print(f"results try {metric}")
        
import scipy.io as sio
from mmseg.apis.inference import show_result_pyplot
def predict_random_image(model, data_dir, label_dir, save_path):
    datas = os.listdir(data_dir)
    rand_idx = np.random.randint(len(datas))
    img_dir = datas[rand_idx]
    label_dir = label_dir + img_dir[:-3] + "png"
    print(f"label_dir {label_dir}")
    print(f"img_dir {img_dir}")
    data_im = data_dir+img_dir
    mat = sio.loadmat(data_im)
    matX = mat["vxSample"]
    matY = mat["vySample"]
    data_prev = np.stack((matX, matY, np.zeros(matX.shape)), -1)
    data_new = (data_prev - data_prev.min()) / (data_prev.max() - data_prev.min())
    label = mpimg.imread(label_dir)
    result = inference_segmentor(model=model, imgs=data_dir+img_dir)
    if save_path is not None:  
        mpimg.imsave(save_path+"predict.png", result[0])
        mpimg.imsave(save_path+"label.png", label)
        mpimg.imsave(save_path+"data.png", data_new)
        show_result_pyplot(model, img=data_prev, result=result, out_file=save_path+"result.png")
