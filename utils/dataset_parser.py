import os
from xmlrpc.client import Boolean
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import matplotlib.image as mpimg
import os.path as osp
from .CustomMatDataset import CustomMatDataset
from detectron2.data import MetadataCatalog, DatasetCatalog


import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset



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
            pin_memory=False
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
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




import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class EddyDatasetREGISTER(CustomMatDataset):
    """Eddy dataset.
    """
    CLASSES = ('eddy',)

    PALETTE = [[255,255,255]]

    def __init__(self, **kwargs):
        super(EddyDatasetREGISTER, self).__init__(
            img_suffix='.mat',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)
        # print(f"result files {result_files}")
        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        # print(f"result files {result_files}")
        return result_files




def convert_mat_to_img(dataset_dir, split, train_dir, valid_dir, label_dir, train_annot_dir, valid_annot_dir):
    all_dirs = os.listdir(dataset_dir)
    label_dirs = os.listdir(label_dir)
    split_len = int(split*len(all_dirs))
    train_dirs = sorted(all_dirs)[0:split_len]
    valid_dirs = sorted(all_dirs)[split_len:]
    train_annot = sorted(label_dirs)[0:split_len]
    valid_annot = sorted(label_dirs)[split_len:]
    #converting starts
    # for dir in train_dirs:
    #     img_dir = dataset_dir + dir
    #     mat = sio.loadmat(img_dir)
    #     matX = mat["vxSample"]
    #     matY = mat["vySample"]
    #     save_dir = train_dir + dir[:-3] + "png"
    #     print(f"converting {dir} to {dir[:-3]}png in {train_dir}")
    #     con_arr = (np.stack((matX, matY, np.zeros(matX.shape)), -1) * 255).astype(np.uint8)
    #     mpimg.imsave(save_dir, con_arr)
    # for dir in valid_dirs:
    #     img_dir = dataset_dir + dir
    #     mat = sio.loadmat(img_dir)
    #     matX = mat["vxSample"]
    #     matY = mat["vySample"]
    #     print(f"converting {dir} to {dir[:-3]}png in {valid_dir}")
    #     con_arr = (np.stack((matX, matY, np.zeros(matX.shape)), -1) * 255).astype(np.uint8)
    #     save_dir = valid_dir + dir[:-3] + "png"
    #     mpimg.imsave(save_dir, con_arr)
    # for dir in train_annot:
    #     img_dir = label_dir + dir
    #     img = mpimg.imread(img_dir)
    #     save_dir = train_annot_dir + dir
    #     print(f"converting {dir} to in {save_dir}")
    #     mpimg.imsave(save_dir, img)
    # for dir in valid_annot:
    #     img_dir = label_dir + dir
    #     img = mpimg.imread(img_dir)
    #     save_dir = valid_annot_dir + dir
    #     print(f"converting {dir} to in {save_dir}")
    #     mpimg.imsave(save_dir, img)
    # print("--Converting finished--")
    
def convert_rgb_annot2_gray_split(convert_dir, split, train_converted, valid_converted):
    dirs = os.listdir(convert_dir)
    split_len = int(split*len(dirs))
    train_dirs = sorted(dirs)[0:split_len]
    valid_dirs = sorted(dirs)[split_len:]
    for dir in train_dirs:
        ori_dir = train_converted + dir
        src = convert_dir + dir
        img = Image.open(src).convert('L')
        print(f"Converting {dir} to Grayscale")
        img.save(ori_dir)
        print(f"Saved {dir} to {ori_dir}")
    for dir in valid_dirs:
        ori_dir = valid_converted + dir
        src = convert_dir + dir
        img = Image.open(src).convert('L')
        print(f"Converting {dir} to Grayscale")
        img.save(ori_dir)
        print(f"Saved {dir} to {ori_dir}")

def split_mat_to_split(split, train_dir, valid_dir, ori_dir):
    """
    this method moves mat files into train and valid splits
    """
    dirs = os.listdir(ori_dir)
    split_len = int(split*len(dirs))
    train_dirs = sorted(dirs)[0:split_len]
    valid_dirs = sorted(dirs)[split_len:]
    import shutil
    for dir in train_dirs:
        src = ori_dir + dir
        dst = train_dir + dir
        shutil.move(src=src, dst=dst)
        print(f"{src} is moved into {dst}")
    for dir in valid_dirs:
        src = ori_dir + dir
        dst = valid_dir + dir
        shutil.move(src=src, dst=dst)
        print(f"{src} is moved into {dst}")
        
def max_min_values(img, obj):
    if isinstance(obj, staticVariable):
        if(obj.max < img.max()):
            obj.max = img.max()
        if(obj.min > img.min()):
            obj.min = img.min()

def convert_rgb_to_gray(ori_dir):
    dirs = os.listdir(ori_dir)
    for dir in dirs:
        img = Image.open(ori_dir+dir).convert('L')
        print(f"Converting {dir} to Grayscale")
        # print(img.shape)
        img.save(ori_dir+dir)
        print(f"Saved {dir} to {ori_dir+dir}")
        
import torch
def revert_gray_scale(ori_dir, save_dir):
    dirs = os.listdir(ori_dir)
    for dir in dirs:
        img = mpimg.imread(ori_dir+dir)
        img = torch.tensor(img)
        gt_cls = img.unique()
        gt_cls = gt_cls[gt_cls != 255]
        masks = []
        for cls in gt_cls:
            masks.append(img == cls)
        if len(gt_cls) == 0:
            masks.append(img == 255)

        masks = torch.stack(masks, dim=0)
        mpimg.imsave(save_dir+dir, masks[0], cmap='gray')
import time
import shutil
def remove_non_labeled_gts_data(data_dir, label_dir, data_dst, label_dst):
    count_blank = 0
    count = 0
    for dir in os.listdir(label_dir):
        img = mpimg.imread(label_dir+dir)
        if (img.max() == 0):
            print("Blank img")
        else:
            count += 1
            print("nonBlank")
            shutil.move(label_dir+dir, label_dst)
            shutil.move(data_dir+dir[:-3]+'mat', data_dst)
    print(count_blank)
    print(count)
    print(count + count_blank)


def load_mat_img(dir):
    img_file = sio.loadmat(dir)
    img_x = img_file["vxSample"]
    img_y = img_file["vySample"]
    
    img = np.stack((img_x, img_y, np.zeros(img_x.shape)), -1)
    #if self.to_float32:
    img = img.astype(np.float32)
    #print(f"new img {img}")
    # print(f"max value in img {img.max()}")
    # print(f"min value in img {img.min()}")
    return img


def convert_dataset_coco_format():
    pass


if __name__ == "__main__":

    
    
    split = 0.85
    
    