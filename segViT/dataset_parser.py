import os
from xmlrpc.client import Boolean
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import matplotlib.image as mpimg

NUM_WORKERS = os.cpu_count()

class EddyDatasetTrain(Dataset):
    def __init__(self, input_image_dir, mask_image_dir, feature_extractor, split) -> None:
        self.input_image_dir = input_image_dir
        self.mask_image_dir = mask_image_dir
        self.feature_extractor = feature_extractor

        image_file_names = [f for f in os.listdir(self.input_image_dir)]
        mask_file_names = [f for f in os.listdir(self.mask_image_dir)]
        self.images = (sorted(image_file_names))[0:split]
        self.masks = (sorted(mask_file_names))[0:split]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = sio.loadmat(self.input_image_dir + self.images[index])
        img_x = img_file["vxSample"]
        img_y = img_file["vySample"]
        input_img = np.stack((img_x, img_y, np.zeros(img_x.shape)), -1)
        mask_img = label_img = mpimg.imread(self.mask_image_dir + self.masks[index])
        encoded_inputs = self.feature_extractor(input_img, mask_img, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        
        return encoded_inputs

class EddyDatasetValid(Dataset):
    def __init__(self, input_image_dir, mask_image_dir, feature_extractor, split) -> None:
        self.input_image_dir = input_image_dir
        self.mask_image_dir = mask_image_dir
        self.feature_extractor = feature_extractor

        image_file_names = [f for f in os.listdir(self.input_image_dir)]
        mask_file_names = [f for f in os.listdir(self.mask_image_dir)]
        self.images = (sorted(image_file_names))[split:]
        self.masks = (sorted(mask_file_names))[split:]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = sio.loadmat(self.input_image_dir + self.images[index])
        img_x = img_file["vxSample"]
        img_y = img_file["vySample"]
        input_img = np.stack((img_x, img_y, np.zeros(img_x.shape)), -1)
        mask_img = label_img = mpimg.imread(self.mask_image_dir + self.masks[index])
        encoded_inputs = self.feature_extractor(input_img, mask_img, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        
        return encoded_inputs