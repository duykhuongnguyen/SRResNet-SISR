### Module imports ###
import os
import math
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torchvision.utils as utils
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_ssim

from model import SRResNet
from data_utils import TrainImageDataset, ValImageDataset, display_transform
import config as cf


### Global Variables ###


### Class declarations ###


### Function declarations ###
if __name__ == '__main__':
    # Data initialization
    train_dataset = TrainImageDataset(cf.TRAIN_HR, cf.CROP_SIZE, cf.UPSCALE_FACTOR)
    val_dataset = ValImageDataset(cf.VAL_HR, cf.UPSCALE_FACTOR)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)

    # Device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Model initialization
    model = SRResNet(cf.UPSCALE_FACTOR).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cf.LR)

    # Train
    best_val_loss = float("Inf")
    results = {'psnr': [], 'ssim': [], 'mse_loss': []}
    for epoch in range(1, cf.EPOCHS + 1):
        print(f'Training epoch {epoch}/{cf.EPOCHS}')
        running_results = {'batch_sizes': 0, 'mse_loss': 0}
        train_bar = tqdm(train_loader)
        model.train()
        for data, target in train_bar:
            data = data.to(device)
            batch_size = data.shape[0]
            running_results['batch_sizes'] += batch_size
            target = target.to(device)
            output = model(data)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss for current batch
            running_results['mse_loss'] += loss.item() * batch_size

            train_bar.set_description(desc='[%d/%d] MSE Loss: %.4f' % (
                epoch, cf.EPOCHS, running_results['mse_loss'] / running_results['batch_sizes']))

        # Evaluation
        model.eval()
        out_path = f'training_results/SRF_{cf.UPSCALE_FACTOR}'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.shape[0]
                val_results['batch_sizes'] += batch_size
                lr = val_lr.to(device)
                hr = val_hr.to(device)
                sr = model(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                val_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                val_results['ssims'] += batch_ssim * batch_size
                val_results['psnr'] = 10 * math.log10((target.max() ** 2) / (val_results['mse'] / val_results['batch_sizes']))
                val_results['ssim'] = val_results['ssims'] / val_results['batch_sizes']
                val_bar.set_description(
                desc='[Val converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f MSE: %.4f' % (
                    val_results['psnr'], val_results['ssim'], val_results['mse']))

                val_images.extend([display_transform()(hr.data.cpu().squeeze(0)), display_transform()(sr.data.cpu().squeeze(0))])
            if val_results['mse'] < best_val_loss:
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.shape[0] // 10)
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                idx = 1
                for img in val_save_bar:
                    img = utils.make_grid(img, nrow=2, padding=5)
                    utils.save_image(img, os.path.join(out_path, f'epoch_{epoch}_index_{idx}.png'), padding=5)
                    idx += 1

                best_val_loss = val_results['mse']
                torch.save(model.state_dict(), f'pretrained_model/model_epoch{epoch}_{val_results["mse"]}.pt')
                print('Best val mse. Saved model')

        results['mse_loss'].append(running_results['mse_loss'] / running_results['batch_sizes'])
        results['psnr'].append(val_results['psnr'])
        results['ssim'].append(val_results['ssim'])

    print('Finished training')

    df = pd.DataFrame(
        data={'mse_loss': results['mse_loss'], 'psnr': results['psnr'], 'ssim': results['ssim']},
        index=range(1, cf.EPOCHS + 1))
    df.to_csv('result.csv', index_label='Epoch')
