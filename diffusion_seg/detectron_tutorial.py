import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.image as mpimg


def main():
    cfg = get_cfg()
    cfg_dir = "/home/emir/Desktop/dev/myResearch/src/DiffusionInst/configs"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(filename="/home/emir/Desktop/dev/myResearch/dataset/COCO/test2017/000000000063.jpg")
    outputs = predictor(im)
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print(type(out.get_image()[:, :, ::-1]))
    print(out.get_image()[:, :, ::-1].shape)
    mpimg.imsave("./predict.png", out.get_image()[:, :, ::-1])
if __name__ == "__main__":
    main()