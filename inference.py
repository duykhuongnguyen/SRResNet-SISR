### Module imports ###
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

    def __init__(self, model_path):
        self.model = SRResNet(4, 7)
        self.model.load_state_dict(torch.load(model_path))

    def infer(self, img_path):
        hr_img = Image.open(img_path)
        w, h = hr_img.size
        crop_size = cal_valid_crop_size(min(w, h), 4)
        lr_scale = Resize(crop_size // 4, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_img = CenterCrop(crop_size)(hr_img)
        lr_img = ToTensor()(lr_scale(hr_img)).unsqueeze(0)
        hr_restore_img = hr_scale(lr_img)
        hr = self.model(lr_img)
        lr_display = display_transform_lr()(lr_img.data.cpu().squeeze(0))
        hr_display = display_transform_hr()(hr.data.cpu().squeeze(0))
        utils.save_image(lr_display, 'examples/output/lr.png')
        utils.save_image(hr_display, 'examples/output/hr.png')


### Function declarations ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrained_model/pretrained_model_7_block/model_epoch63_0.35649678111076355.pt')
    parser.add_argument('--file_path', type=str, default='examples/input/0801.png')
    args = parser.parse_args()

    infer_module = Inference(args.model_path)
    infer_module.infer(args.file_path)
