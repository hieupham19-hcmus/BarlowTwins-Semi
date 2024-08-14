import argparse
import yaml
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from medpy.metric.binary import dc, jc

from Data_Processing.create_dataset import get_dataset, EndoscopyDataset
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Models.Transformer.SwinUnet import SwinUnet

torch.cuda.empty_cache()

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, output, target):
        # BCE Loss
        bce_loss = self.bce(output, target)
        
        # Dice Loss
        smooth = 1.
        output_flat = output.view(-1)
        target_flat = target.view(-1)
        intersection = (output_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (output_flat.sum() + target_flat.sum() + smooth)
        
        return bce_loss + dice_loss

def main(config):
    
    dataset = get_dataset(config, img_size=config.data.img_size, 
                                                supervised_ratio=config.data.supervised_ratio, 
                                                train_aug=config.data.train_aug,
                                                k=config.fold,
                                                lb_dataset=EndoscopyDataset)

    train_loader = data.DataLoader(dataset['lb_dataset'],
                                                batch_size=config.train.l_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    val_loader = data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    test_loader = data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    
    print(len(train_loader), len(dataset['lb_dataset']))

    model  = SwinUnet(img_size=config.data.img_size)
    
    # Log number of parameters
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params / 1e6:.2f}M total parameters')
    print(f'{total_trainable_params / 1e6:.2f}M total trainable parameters')

    model = model.cuda()
    
    criterion = DiceBCELoss().cuda()

    if config.test.only_test:
        test(config, model, config.test.test_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion)
        test(config, model, best_model_dir, test_loader, criterion)

def train_val(config, model, train_loader, val_loader, criterion):
    optimizer = optim.AdamW(model.parameters(), lr=config.train.optimizer.adamw.lr,
                            weight_decay=config.train.optimizer.adamw.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = config.train.num_epochs
    max_dice = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        start = time.time()

        model.train()
        dice_train_sum, loss_train_sum = 0, 0
        num_train = 0
        
        for idx, batch in enumerate(train_loader):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()

            optimizer.zero_grad()
            output = model(img)
            output = torch.sigmoid(output)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            loss_train_sum += loss.item() * img.size(0)
            num_train += img.size(0)
            
            with torch.no_grad():
                output = output.cpu().numpy() > 0.5
                label = label.cpu().numpy()
                dice_train = dc(output, label)
                dice_train_sum += dice_train * img.size(0)
            
            if config.debug: break

        avg_loss_train = loss_train_sum / num_train
        avg_dice_train = dice_train_sum / num_train

        # Validation phase
        model.eval()
        dice_val_sum, loss_val_sum = 0, 0
        num_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].cuda().float()
                label = batch['label'].cuda().float()
                
                output = model(img)
                output = torch.sigmoid(output)
                
                loss = criterion(output, label)
                loss_val_sum += loss.item() * img.size(0)
                
                output = output.cpu().numpy() > 0.5
                label = label.cpu().numpy()
                dice_val_sum += dc(output, label) * img.size(0)
                
                num_val += img.size(0)
                
                if config.debug: break

        avg_loss_val = loss_val_sum / num_val
        avg_dice_val = dice_val_sum / num_val

        scheduler.step(avg_loss_val)

        # Print the current learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs}, Current LR: {current_lr:.6f}")
        
        # Log results
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss_train:.4f}, Train Dice: {avg_dice_train:.4f}, Val Loss: {avg_loss_val:.4f}, Val Dice: {avg_dice_val:.4f}')

        # Save best model based on validation Dice score
        if avg_dice_val > max_dice:
            torch.save(model.state_dict(), best_model_dir)
            max_dice = avg_dice_val
            best_epoch = epoch + 1
            print(f'New best model at epoch {best_epoch} with Val Dice: {max_dice:.4f}')

        end = time.time()
        print(f'Epoch {epoch + 1} complete in {(end - start) // 60:.0f}m {(end - start) % 60:.0f}s')

        if config.debug: return
    
    print(f'Training complete. Best epoch: {best_epoch} with Val Dice: {max_dice:.4f}')


def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_sum, loss_test_sum = 0, 0
    num_test = 0
    
    with torch.no_grad():
        for batch in test_loader:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            output = model(img)
            output = torch.sigmoid(output)
            
            loss = criterion(output, label)
            loss_test_sum += loss.item() * img.size(0)
            
            output = output.cpu().numpy() > 0.5
            label = label.cpu().numpy()
            dice_test_sum += dc(output, label) * img.size(0)
            
            num_test += img.size(0)
    
    avg_loss_test = loss_test_sum / num_test
    avg_dice_test = dice_test_sum / num_test
    
    print(f'Test Results - Loss: {avg_loss_test:.4f}, Dice: {avg_dice_test:.4f}')

if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='isic2018')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['model_adapt']['adapt_method'] = args.adapt_method
    config['model_adapt']['num_domains'] = args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    # Print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}")
    
    store_config = config
    config = DotDict(config)
    
    # Logging directories
    exp_dir = f'{config.data.save_folder}/{args.exp}_{config["data"]["supervised_ratio"]}/fold{args.fold}'
    os.makedirs(exp_dir, exist_ok=True)
    best_model_dir = f'{exp_dir}/best.pth'
    
    if not config.debug:
        yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))
        
    main(config)
