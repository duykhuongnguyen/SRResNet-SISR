### Module imports ###
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, CenterCrop, Resize, ToPILImage, ToTensor


### Global Variables ###


### Class declarations ###
class TrainImageDataset(Dataset):
    """ Train dataset class """
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        """
        Initialization for dataset

        Args:
            dataset_dir: dataset folder path
            crop_size: crop size of high resolution image
            upscale_factor: upscale for image from low resolution to high resolution
        """
        super(TrainImageDataset, self).__init__()
        # All files in folder
        self.files = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        crop_size = cal_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = hr_transform(crop_size)
        self.lr_transform = lr_transform(crop_size, upscale_factor)

    def __getitem__(self, idx):
        # Crop and transform image to torch tensor
        hr_img = self.hr_transform(Image.open(self.files[idx]))
        lr_img = self.lr_transform(hr_img)
        return lr_img, hr_img

    def __len__(self):
        return len(self.files)


class ValImageDataset(Dataset):
    """ Val dataset class """
    def __init__(self, dataset_dir, upscale_factor):
        """
        Initialization for dataset

        Args:
            dataset_dir: dataset folder path
            upscale_factor: upscale for image from low resolution to high resolution
        """
        super(ValImageDataset, self).__init__()
        self.upscale_factor = upscale_factor
        # All files in folder
        self.files = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, idx):
        # Crop and transform image to torch tensor
        hr_img = Image.open(self.files[idx])
        w, h = hr_img.size
        crop_size = cal_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_img = CenterCrop(crop_size)(hr_img)
        lr_img = lr_scale(hr_img)
        hr_restore_img = hr_scale(lr_img)
        return ToTensor()(lr_img), ToTensor()(hr_restore_img), ToTensor()(hr_img)

    def __len__(self):
        return len(self.files)


### Function declarations ###
def is_image_file(file_name):
    """
    Check if a file is a valid image file

    Args:
        file_name: path of a file

    Returns: True/False
    """
    return any(file_name.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def cal_valid_crop_size(crop_size, upscale_factor):
    """
    Calculate valid crop size for a crop size to be identified

    Args:
        crop_size: crop size for an image
        upscale_factor: upscale for image from low resolution to high resolution

    Returns: valid crop size
    """
    return crop_size - (crop_size % upscale_factor)


def hr_transform(crop_size):
    """ Transform for high resolution image """
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def lr_transform(crop_size, upscale_factor):
    """ Transform for low resolution image """
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor(),
    ])


def display_transform():
    """ Transform for display  """
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])
