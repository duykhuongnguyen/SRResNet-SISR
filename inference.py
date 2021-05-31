### Module imports ###
import os
from PIL import Image
import argparse

import torch
from torchvision.transforms import Compose, RandomCrop, CenterCrop, Resize, ToPILImage, ToTensor
import torchvision.utils as utils

from model import SRResNet
from data_utils import cal_valid_crop_size, display_transform_lr, display_transform_hr


### Global Variables ###


### Class declarations ###
class Inference:
    """ Module for inference """
    def __init__(self, model_path):
        """ Init and load model

        Args:
            model_path: pretrained model path
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = SRResNet(4, 5)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

    def infer(self, img_path, output_dir='examples', save=False):
        """ Infer from a single image with SRResNet or Nearest Neighbor

        Args:
            img_path: image file path
            mode: srresnet or nn
            output_dir: directory to save the image
            save: save image or not

        Returns:
            3 tensor: low resolution image, high resolution image, orignal image
        """
        # img_name = img_path.split('/')[-1].split('.')[0]

        # open image
        # hr_img = Image.open(img_path)
        hr_img = Image.fromarray(img_path, 'RGB')
        w, h = hr_img.size
        crop_size = cal_valid_crop_size(min(w, h), 4)
        lr_scale = Resize(crop_size // 4, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        # crop image to have valid size
        # hr_img = CenterCrop(crop_size)(hr_img)
        # lr_img = ToTensor()(lr_scale(hr_img)).unsqueeze(0)
        # hr_restore_img = hr_scale(lr_img)
        sr = self.model(ToTensor()(hr_img).unsqueeze(0))
        return ToPILImage()(sr.data.cpu().squeeze(0))

        if save:
            lr_display = display_transform_lr()(lr_img.data.cpu().squeeze(0))
            sr_display = display_transform_hr()(sr.data.cpu().squeeze(0))
            hr_display = display_transform_hr()(ToTensor()(hr_img).cpu().squeeze(0))
            utils.save_image(lr_display, os.path.join(output_dir, f'{img_name}_lr.png'))
            utils.save_image(hr_display, os.path.join(output_dir, f'{img_name}_hr.png'))
            utils.save_image(sr_display, os.path.join(output_dir, f'{img_name}_sr.png'))
        return lr_img.data.cpu().squeeze(0), sr.data.cpu().squeeze(0), ToTensor()(hr_img).cpu().squeeze(0)


### Function declarations ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrained_model/model_epoch100_0.32897382974624634.pt')
    parser.add_argument('--file_path', type=str, default='examples/input/0801.png')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='examples/output/')
    args = parser.parse_args()

    infer_module = Inference(args.model_path)
    infer_module.infer(args.file_path, output_dir=args.output_dir, save=args.save)
