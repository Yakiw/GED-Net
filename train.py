import argparse
import logging
import logging.config
from utils.load_conf import ConfigLoader
from pathlib import Path

logger_path = Path("./configs/logger.yaml")
conf = ConfigLoader(logger_path)
_logger = logging.getLogger(__name__)

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
import torch.nn.functional as F

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split,Subset

from GED-Net import GEDNet
from network import*


from sklearn.model_selection import KFold
from torchvision import transforms

import torch.nn.init as init
from thop import profile

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


base = Path(os.environ['raw_data_base']) if 'raw_data_base' in os.environ.keys() else Path('./BUSI_malignant')
assert base is not None, "Please assign the raw_data_base(which store the training data) in system path "
dir_img = base / 'imgs'
dir_mask = base / 'masks/'
dir_checkpoint = './checkpoint/'

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

def train_net(net,
              device,
              model_name,
              epochs=5,
              batch_size=1,
              lr=0.0001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              num_folds=5,
              input_folds=[0,1,2,3,4]
              ):

    dataset = BasicDataset(str(dir_img.resolve()), str(dir_mask.resolve()), img_scale)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=41)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        if fold not in input_folds:
            # Skip folds not in input_folds
            continue
        
        # print(train_idx)
        # print(val_idx)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        n_val = len(val_dataset)
        n_train = len(train_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

        global_step = 0

        _logger.info(f'''Starting training:
            Fold:            {fold + 1}/{num_folds}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
            Images scaling:  {img_scale}
        ''')

        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000006)
        criterion = nn.CrossEntropyLoss() if net.num_classes > 1 else nn.BCEWithLogitsLoss()
        criterion2 = DiceLoss()
        avg_score = 0
        net.apply(weights_init)
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            total_batches = 0
            total_score = 0
            with tqdm(total=n_train, desc=f'Fold {fold + 1} / Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    imgs = batch['image']
                    true_masks = batch['mask']

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if net.num_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)

                    masks_pred = net(imgs)
                    loss = 0.5*criterion(masks_pred, true_masks) + 0.5*criterion2(masks_pred, true_masks)
                    epoch_loss += loss.item()
                    
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    pbar.update(imgs.shape[0])
                    global_step += 1
                    if global_step % max((n_train // (10 * batch_size)), 1) == 0:
                        val_score = eval_net(net, val_loader, device)
                        scheduler.step(val_score)
                        total_score += val_score
                        total_batches += 1
                        if net.num_classes > 1:
                            _logger.info('Validation cross entropy: {}'.format(val_score))
                        else:
                            _logger.info('Validation Dice Coeff: {}'.format(val_score))

	            # Calculate average score at the end of the epoch
            avg_score = total_score / total_batches
            print(f'Average score for epoch {epoch + 1}: {avg_score}')
            if epoch % 10 == 0:
                print(f'Saving checkpoint for epoch {epoch + 1} with average score: {avg_score}')
                if save_cp:
                    try:
                        os.mkdir(dir_checkpoint)
                        _logger.info('Created checkpoint directory')
                    except OSError:
                        _logger.error('Failed to create checkpoint directory!')
                torch.save(net.state_dict(), dir_checkpoint + f'Fold{fold + 1}_Epoch{epoch + 1}.pth')
                _logger.info(f'Checkpoint {fold + 1}/{epoch + 1} saved!')
                    
            if epoch == epochs - 1:
                output_dir = f'result_output_BUSI_malignant/{model_name}/fold_{fold + 1}/'
                os.makedirs(output_dir, exist_ok=True)
                with torch.no_grad():
                    net.eval()
                    for batch in val_loader:
                        imgs = batch['image'].to(device=device, dtype=torch.float32)
                        file_names = batch['file_name']
                        #print(file_names)

                        masks_pred = net(imgs)

                        for i in range(imgs.size(0)):
                            img = transforms.ToPILImage()(imgs[i].cpu())
                            mask_pred = transforms.ToPILImage()((torch.sigmoid(masks_pred[i]) > 0.5).float().cpu())

                            original_file_name = file_names[i]
                            #print(original_file_name)
                            img.save(os.path.join(output_dir,f'{original_file_name}_img.png'))
                            mask_pred.save(os.path.join(output_dir,f'{original_file_name}_mask.png'))

                    
        result_log = f'Model: {model_name}\n'
        result_log += f'Fold_{fold + 1}_Dice: {avg_score}\n'

             


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('--channels', type=int, default=3, help='image channels', dest='channels')
    parser.add_argument('--classes', type=int, default=1, help='mask nums', dest='classes')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.4375,
                        help='Downscaling factor of the images')
    parser.add_argument("--folds", nargs='+', type=int, default=[0], help="Specify the folds to train (e.g., --folds 0 1 2)")
    parser.add_argument('-nf',"--num_folds",  type=int, default=5, help="X fold cross-validation")
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device index (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    _logger.info(f'Using device {device}')
    
 
    model_names = ['GEDNet']
    for model_name in model_names:
        net = None
        if model_name == 'GEDNet':
            net = GEDNet(n_channels=args.channels, num_classes=args.classes)
        
        _logger.info(f'Network:\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{net.num_classes} output channels (classes)\n')
                     # f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            _logger.info(f'Model loaded from {args.load}')

        net.to(device=device)

        cudnn.benchmark = True
        
        try:
            train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device,
                      img_scale=args.scale,
                      val_percent=args.val / 100,
                      num_folds=args.num_folds,
                      input_folds=args.folds,
                      model_name=model_name)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            _logger.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
