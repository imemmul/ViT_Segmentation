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
import matplotlib.image as mpimg
from mmseg.apis.test import single_gpu_test
from utils.utils import run_test, predict_random_image


try:
    EddyDatasetREGISTER()
except:
    pass



def main():
    #TO-DO
    # I am getting true predictions but while evaluating not
    
    
    
    cfg_path = "/home/emir/Desktop/dev/myResearch/src/ViT_Segmentation/conv_based_segmentors/configs/fcn_unet_s5/fcn_unet_s5-d16-eddy.py"
    cfg = Config.fromfile(cfg_path)
    cp_path = "/home/emir/Desktop/dev/myResearch/src/ViT_Segmentation/conv_based_segmentors/output/iter_160000.pth"
    # model = build_segmentor(cfg=cfg.model)
    # model.CLASSES = EddyDatasetREGISTER.CLASSES
    # model.PALETTE = EddyDatasetREGISTER.PALETTE
    model = init_segmentor(config=cfg, checkpoint=cp_path, device=device)
    datasets = build_dataset(cfg.data.train)
    # print(model)
    train_segmentor(model, dataset=datasets, cfg=cfg ,validate=False)
    save_path = "/home/emir/Desktop/dev/myResearch/src/ViT_Segmentation/conv_based_segmentors/output/"
    data_dir = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/train_data_aug_non/"
    label_dir = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/train_label_aug_non/"
    import time
    # for i in range(10):
        
    #     predict_random_image(model=model, label_dir=label_dir, data_dir=data_dir, save_path=save_path)
    #     time.sleep(1)
    # run_test(cfg=cfg, distributed=False, cp_path=cp_path)

if __name__ == "__main__":
    main()
    