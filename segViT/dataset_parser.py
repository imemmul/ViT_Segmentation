import os
from xmlrpc.client import Boolean
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import matplotlib.image as mpimg

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str=None, 
    test_dir: str=None,
    test_data=None,
    train_data=None,
    is_data_exist: Boolean=False,
    transform=None, 
    batch_size: int=32, 
    num_workers: int=NUM_WORKERS):

    train_dataloader, test_dataloader = 0, 0
    train_data, test_data = train_data, test_data
    # Use ImageFolder to create dataset(s)
    if is_data_exist:
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Get class names
    class_names = None
    try:
        class_names = train_data.classes
    except:
        pass

    return train_dataloader, test_dataloader, class_names



class EddyDataset(Dataset):
    def __init__(self, input_image_dir, mask_image_dir, split, train_bool) -> None:
        self.input_image_dir = input_image_dir
        self.mask_image_dir = mask_image_dir
        self.split_len = int(len(os.listdir(input_image_dir)) * split)
        self.train_bool = train_bool
        image_file_names = [f for f in os.listdir(self.input_image_dir)]
        mask_file_names = [f for f in os.listdir(self.mask_image_dir)]
        if train_bool: #train
            self.images = (sorted(image_file_names))[0:self.split_len]
            self.masks = (sorted(mask_file_names))[0:self.split_len]
        else: # valid
            self.images = (sorted(image_file_names))[self.split_len:]
            self.masks = (sorted(mask_file_names))[self.split_len:]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_file = sio.loadmat(self.input_image_dir + self.images[index])
        img_x = img_file["vxSample"]
        img_y = img_file["vySample"]
        input_img = np.stack((img_x, img_y, np.zeros(img_x.shape)), -1)
        mask_img = mpimg.imread(self.mask_image_dir + self.masks[index])
        return input_img, mask_img

def convert_mat_to_img(dataset_dir, split, train_dir, valid_dir):
    all_dirs = os.listdir(dataset_dir)
    split_len = int(split*len(all_dirs))
    train_dirs = sorted(all_dirs)[0:split_len]
    valid_dirs = sorted(all_dirs)[split_len:]
    #converting starts
    for dir in train_dirs:
        img_dir = dataset_dir + dir
        mat = sio.loadmat(img_dir)
        matX = mat["vxSample"]
        matY = mat["vySample"]
        save_dir = train_dir + dir[:-3] + "png"
        print(f"converting {dir} to {dir[:-3]}png in {train_dir}")
        con_arr = (np.stack((matX, matY, np.zeros(matX.shape)), -1) * 255).astype(np.uint8)
        mpimg.imsave(save_dir, con_arr)
    for dir in valid_dirs:
        img_dir = dataset_dir + dir
        mat = sio.loadmat(img_dir)
        matX = mat["vxSample"]
        matY = mat["vySample"]
        print(f"converting {dir} to {dir[:-3]}png in {valid_dir}")
        con_arr = (np.stack((matX, matY, np.zeros(matX.shape)), -1) * 255).astype(np.uint8)
        save_dir = valid_dir + dir[:-3] + "png"
        mpimg.imsave(save_dir, con_arr)
    print("--Converting finished--")

if __name__ == "__main__":
    dataset_dir = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/data4test/data/"
    train_dir = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/data4test/train_data/"
    valid_dir = "/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/data4test/valid_data/"
    convert_mat_to_img(dataset_dir=dataset_dir, train_dir=train_dir, valid_dir=valid_dir, split=0.85)