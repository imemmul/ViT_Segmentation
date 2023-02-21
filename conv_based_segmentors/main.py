import sys
sys.path.insert(0, '..')
from utils.dataset_parser import EddyDatasetREGISTER
from mmcv import Config
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor



try:
    EddyDatasetREGISTER()
except:
    pass



def main():
    cfg_path = "/home/emir/Desktop/dev/myResearch/src/ViT_Segmentation/conv_based_segmentors/configs/fcn_unet_s5/fcn_unet_s5-d16-eddy.py"
    cfg = Config.fromfile(cfg_path)
    model = build_segmentor(cfg.model)
    datasets = build_dataset(cfg.data.train)
    train_segmentor(model, dataset=datasets, cfg=cfg ,validate=False)



if __name__ == "__main__":
    main()
    