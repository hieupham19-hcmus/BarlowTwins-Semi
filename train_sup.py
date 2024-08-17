'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 16 --adapt_method False --num_domains 1 --dataset PH2  --k_fold 4 > 4MedFormer_PH2.out 2>&1 &
'''
import argparse
from sqlite3 import adapt
import yaml
import os, time
from datetime import datetime
from tqdm import tqdm


import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics
import matplotlib.pyplot as plt
import numpy
import torchvision.utils as vutils

from Data_Processing.create_dataset import get_dataset, EndoscopyDataset
from Utils.losses import dice_loss
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Models.Transformer.SwinUnet import SwinUnet
from Models.Unet import UNet

torch.cuda.empty_cache()

def visualize_batch(batch, predictions=None):
    images = batch['image'].cpu()
    labels = batch['label'].cpu()
    grid_images = vutils.make_grid(images, nrow=2, normalize=True)
    grid_labels = vutils.make_grid(labels, nrow=2, normalize=True)
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Images")
    plt.imshow(grid_images.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.title("Labels")
    plt.imshow(grid_labels.permute(1, 2, 0))
    if predictions is not None:
        grid_predictions = vutils.make_grid(predictions.cpu(), nrow=8, normalize=True)
        plt.figure(figsize=(6, 8))
        plt.title("Predictions")
        plt.imshow(grid_predictions.permute(1, 2, 0))
    plt.show()

def main(config):
    
    dataset = get_dataset(config, img_size=config.data.img_size, 
                                                supervised_ratio=config.data.supervised_ratio, 
                                                train_aug=config.data.train_aug,
                                                k=config.fold,
                                                lb_dataset=EndoscopyDataset)

    train_loader = torch.utils.data.DataLoader(dataset['lb_dataset'],
                                                batch_size=config.train.l_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    print(len(train_loader), len(dataset['lb_dataset']))

    
    model = UNet(n_channels=3, n_classes=1)


    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    
    # from thop import profile
    # input = torch.randn(1,3,224,224)
    # flops, params = profile(model, (input,))
    # print(f"total flops : {flops/1e9} G")

    # test model
    # x = torch.randn(5,3,224,224)
    # y = model(x)
    # print(y.shape)

    model = model.cuda()
    
    criterion = [nn.BCELoss(), dice_loss]

    # only test
    if config.test.only_test == True:
        test(config, model, config.test.test_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion)
        test(config, model, best_model_dir, test_loader, criterion)



# =======================================================================================================

def train_val(config, model, train_loader, val_loader, criterion):
    # optimizer setup
    if config.train.optimizer.mode == 'adam':
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    epochs = config.train.num_epochs
    max_iou = 0
    max_dice = 0
    best_epoch = 0
    
    torch.save(model.state_dict(), best_model_dir)
    
    for epoch in range(epochs):
        start = time.time()
        
        # Training
        model.train()
        dice_train_sum = 0
        iou_train_sum = 0
        loss_train_sum = 0
        num_train = 0
        
        train_loader = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", unit="batch")
        for idx, batch in enumerate(train_loader):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]
            
            output = model(img)
            output = torch.sigmoid(output)
            
            # Calculate loss
            assert (output.shape == label.shape)
            dice_loss_value = criterion[1](output, label)
            bce_loss_value = criterion[0](output, label)
            loss = (dice_loss_value + bce_loss_value) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train_sum += loss.item() * batch_len
            
            # Calculate metrics
            with torch.no_grad():
                output = output.cpu().numpy() > 0.5
                label = label.cpu().numpy()
                dice_train = metrics.dc(output, label)
                iou_train = metrics.jc(output, label)
                dice_train_sum += dice_train * batch_len
                iou_train_sum += iou_train * batch_len
            
            num_train += batch_len
            
            # Update progress bar with current losses and learning rate
            train_loader.set_postfix({
                'Dice Loss': dice_loss_value.item(),
                'BCE Loss': bce_loss_value.item(),
                'Total Loss': loss.item(),
                'LR': optimizer.param_groups[0]['lr'],
                'GPU Memory': f'{torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB'
            })
            
            if config.debug:
                break
        
        # Print and log training results
        avg_loss_train = round(loss_train_sum / num_train, 5)
        avg_dice_train = round(dice_train_sum / num_train, 4)
        avg_iou_train = round(iou_train_sum / num_train, 4)
        file_log.write(f'Epoch {epoch}, Total train step {len(train_loader)} || AVG_loss: {avg_loss_train}, Avg Dice score: {avg_dice_train}, Avg IOU: {avg_iou_train}\n')
        file_log.flush()
        print(f'Epoch {epoch}, Total train step {len(train_loader)} || AVG_loss: {avg_loss_train}, Avg Dice score: {avg_dice_train}, Avg IOU: {avg_iou_train}')

        # Validation
        model.eval()
        dice_val_sum = 0
        iou_val_sum = 0
        loss_val_sum = 0
        num_val = 0
        
        val_loader = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_id, batch in enumerate(val_loader):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]
            
            with torch.no_grad():
                output = model(img)
                output = torch.sigmoid(output)
                
                assert (output.shape == label.shape)
                dice_loss_value = criterion[1](output, label)
                bce_loss_value = criterion[0](output, label)
                loss_val_sum += (dice_loss_value + bce_loss_value) * batch_len
                
                output = output.cpu().numpy() > 0.5
                label = label.cpu().numpy()
                dice_val_sum += metrics.dc(output, label) * batch_len
                iou_val_sum += metrics.jc(output, label) * batch_len

                num_val += batch_len
                if config.debug:
                    break

            # Update progress bar with current losses
            val_loader.set_postfix({
                'Dice Loss': dice_loss_value.item(),
                'BCE Loss': bce_loss_value.item(),
                'Total Loss': loss_val_sum / num_val,
                'GPU Memory': f'{torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB'
            })
        
        # Print and log validation results
        loss_val_epoch = loss_val_sum / num_val
        dice_val_epoch = dice_val_sum / num_val
        iou_val_epoch = iou_val_sum / num_val
        file_log.write(f'Epoch {epoch}, Validation || sum_loss: {round(loss_val_epoch.item(), 5)}, Dice score: {round(dice_val_epoch, 4)}, IOU: {round(iou_val_epoch, 4)}\n')
        file_log.flush()
        print(f'Epoch {epoch}, Validation || sum_loss: {round(loss_val_epoch.item(), 5)}, Dice score: {round(dice_val_epoch, 4)}, IOU: {round(iou_val_epoch, 4)}')

        # Scheduler step
        scheduler.step()

        # Save best model based on dice score
        if dice_val_epoch > max_dice:
            torch.save(model.state_dict(), best_model_dir)
            max_dice = dice_val_epoch
            best_epoch = epoch
            file_log.write(f'New best epoch {epoch}!===============================\n')
            file_log.flush()
            print(f'New best epoch {epoch}!===============================')
        
        end = time.time()
        time_elapsed = end - start
        file_log.write(f'Training and evaluating on epoch {epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        file_log.flush()
        print(f'Training and evaluating on epoch {epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if config.debug:
            return
    
    file_log.write(f'Complete training ---------------------------------------------------- \n The best epoch is {best_epoch}\n')
    file_log.flush()
    print(f'Complete training ---------------------------------------------------- \n The best epoch is {best_epoch}')

    return



# ========================================================================================================
def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_sum= 0
    iou_test_sum = 0
    loss_test_sum = 0
    num_test = 0
    for batch_id, batch in enumerate(test_loader):
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        batch_len = img.shape[0]
            
        with torch.no_grad():
                
            output = model(img)

            output = torch.sigmoid(output)

            # calculate loss
            assert (output.shape == label.shape)
            losses = []
            for function in criterion:
                losses.append(function(output, label))
            loss_test_sum += sum(losses)*batch_len

            # calculate metrics
            output = output.cpu().numpy() > 0.5
            label = label.cpu().numpy()
            dice_test_sum += metrics.dc(output, label)*batch_len
            iou_test_sum += metrics.jc(output, label)*batch_len

            num_test += batch_len
            # end one test batch
            if config.debug: break

    # logging results for one dataset
    loss_test_epoch, dice_test_epoch, iou_test_epoch = loss_test_sum/num_test, dice_test_sum/num_test, iou_test_sum/num_test


    # logging average and store results
    with open(test_results_dir, 'w') as f:
        f.write(f'loss: {loss_test_epoch.item()}, Dice_score {dice_test_epoch}, IOU: {iou_test_epoch}')

    # print
    file_log.write('========================================================================================\n')
    file_log.write('Test || Average loss: {}, Dice score: {}, IOU: {}\n'.
                        format(round(loss_test_epoch.item(),5), 
                        round(dice_test_epoch,4), round(iou_test_epoch,4)))
    file_log.flush()
    
    print('========================================================================================')
    print('Test || Average loss: {}, Dice score: {}, IOU: {}'.
            format(round(loss_test_epoch.item(),5), 
            round(dice_test_epoch,4), round(iou_test_epoch,4)))

    return




if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp', type=str,default='tmp')
    parser.add_argument('--config_yml', type=str,default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='isic2018')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['model_adapt']['adapt_method']=args.adapt_method
    config['model_adapt']['num_domains']=args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))
    
    store_config = config
    config = DotDict(config)
    
    # logging tensorbord, config, best model
    exp_dir = '{}/{}_{}/fold{}'.format(config.data.save_folder, args.exp, config['data']['supervised_ratio'], args.fold)
    os.makedirs(exp_dir, exist_ok=True)
    best_model_dir = '{}/best.pth'.format(exp_dir)
    test_results_dir = '{}/test_results.txt'.format(exp_dir)

    # store yml file
    if config.debug == False:
        yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
        
    file_log = open('{}/log.txt'.format(exp_dir), 'w')    
    main(config)