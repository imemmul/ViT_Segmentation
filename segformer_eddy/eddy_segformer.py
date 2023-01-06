import torch
import mmseg
from mmcv import Config
import mmcv
from dataset_parser import EddyDatasetREGISTER
from mmseg.datasets import build_dataset
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot, train_segmentor
import torch.nn as nn
try :
    EddyDatasetREGISTER()
except :
    pass

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


def train_model(model, cfg, dataset):
    train_segmentor(model=model, cfg=cfg, dataset=dataset, validate=True, distributed=False)

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(mmseg.__version__)
    print(mmcv.__version__)
    checkpoint_path = "/home/emir/dev/segmentation_eddies/downloads/checkpoints/segformer/segformer_b5_ade20k/trained_models/segformer.b5.640x640.ade.160k.pth"
    config_path = "/home/emir/dev/segmentation_eddies/ViT_Segmentation/segformer_eddy/local_configs/segformer/B5/segformer.b5.640x640.eddy.160k.py"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    cfg = Config.fromfile(config_path)
    model = init_segmentor(config=cfg, checkpoint=checkpoint_path, device=device)
    datasets = build_dataset(cfg.data.train)
    print(model.decode_head.linear_pred)
    model.decode_head.linear_pred = nn.Conv2d(in_channels=768, out_channels=2, kernel_size=(1,1), stride=(1,1))
    print(model.decode_head.linear_pred)
    print(model)
    train_model(model, cfg, datasets)
    
    #run_training()