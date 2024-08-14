import argparse
import yaml
import os
import time
from datetime import datetime
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.optim as optim
import segmentation_models_pytorch as smp
import medpy.metric.binary as metrics

from Data_Processing.create_dataset import get_dataset, EndoscopyDataset
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Models.Transformer.SwinUnet import SwinUnet
from Models.CNN.ResNet import resnet50

torch.cuda.empty_cache()

def dice_loss(output, target):
    """Calculate Dice Loss."""
    return smp.losses.DiceLoss(mode='binary')(output, target)

def jaccard_loss(output, target):
    """Calculate Jaccard Loss."""
    return smp.losses.JaccardLoss(mode='binary')(output, target)

def threshold_predictions(output, threshold=0.5):
    """Threshold model predictions."""
    return (output > threshold).float()

def main(config):
    # Set up logging
    exp_dir = f'{config.data.save_folder}/{args.exp}_{config.data.supervised_ratio}/fold{args.fold}'
    os.makedirs(exp_dir, exist_ok=True)
    best_model_dir = f'{exp_dir}/best.pth'
    test_results_dir = f'{exp_dir}/test_results.txt'

    logging.basicConfig(filename=f'{exp_dir}/log.txt', level=logging.INFO)
    logger = logging.getLogger()

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
    
    # visual some samples from train_loader and val_loader
    
    def visualize_batch(batch, predictions=None):
        images = batch['image'].cpu()
        labels = batch['label'].cpu()
        grid_images = vutils.make_grid(images, nrow=8, normalize=True)
        grid_labels = vutils.make_grid(labels, nrow=8, normalize=True)
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
        
    def save_img_from_batch(batch, save_dir, normalize=True):
        images = batch['image'].cpu()
        labels = batch['label'].cpu()
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        for i, (image, label) in enumerate(zip(images, labels)):
            # Optionally normalize the image and label
            if normalize:
                image = (image - image.min()) / (image.max() - image.min())
                label = (label - label.min()) / (label.max() - label.min())

            # Convert image tensor to numpy array and handle permuting
            image_np = image.permute(1, 2, 0).numpy()
            
            # Check if the label is single-channel (grayscale) and handle it
            if label.shape[0] == 1:
                label_np = label.squeeze(0).numpy()  # Remove channel dimension for grayscale
                plt.imsave(os.path.join(save_dir, f'label_{i}.png'), label_np, cmap='gray')
            else:
                label_np = label.permute(1, 2, 0).numpy()
                plt.imsave(os.path.join(save_dir, f'label_{i}.png'), label_np)

            # Save the image
            plt.imsave(os.path.join(save_dir, f'image_{i}.png'), image_np)

        print(f"Images and labels saved in {save_dir}")
        

        
     
    model = SwinUnet(img_size=config.data.img_size)
    
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params/1e6}M total parameters')
    print(f'{total_trainable_params/1e6}M total trainable parameters')

    model = model.cuda()

    criterion = [dice_loss, jaccard_loss]

    # Only test
    if config.test.only_test:
        test(config, model, config.test.test_model_dir, test_loader, criterion, logger, test_results_dir)
    else:
        train_val(config, model, train_loader, val_loader, criterion, logger, best_model_dir)
        test(config, model, best_model_dir, test_loader, criterion, logger, test_results_dir)

def train_val(config, model, train_loader, val_loader, criterion, logger, best_model_dir):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-4,  # Lower learning rate
                            weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    epochs = config.train.num_epochs
    best_dice = -float('inf')
    best_epoch = 0

    dice_weight = 0.5
    jaccard_weight = 0.5

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss_sum = 0
        dice_train_sum = 0
        jaccard_train_sum = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch in train_loader:
                img = batch['image'].cuda().float()
                label = batch['label'].cuda().float()

                optimizer.zero_grad()

                output = torch.sigmoid(model(img))

                loss = dice_weight * criterion[0](output, label) + jaccard_weight * criterion[1](output, label)
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss_sum += loss.item()
                dice_train_sum += (1 - criterion[0](output, label)).item()
                jaccard_train_sum += (1 - criterion[1](output, label)).item()

                pbar.set_postfix({
                    'gpu_mem': f'{torch.cuda.memory_allocated() / 1e9:.2f} GB',
                    'lr': scheduler.get_last_lr()[0],
                    'train_loss': train_loss_sum / (pbar.n + 1)
                })
                pbar.update(1)

        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0
        dice_val_sum = 0
        jaccard_val_sum = 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].cuda().float()
                label = batch['label'].cuda().float()

                output = torch.sigmoid(model(img))

                val_loss_sum += (dice_weight * criterion[0](output, label) + jaccard_weight * criterion[1](output, label)).item()
                dice_val_sum += (1 - criterion[0](output, label)).item()
                jaccard_val_sum += (1 - criterion[1](output, label)).item()

        avg_val_dice = dice_val_sum / len(val_loader)
        avg_val_jaccard = jaccard_val_sum / len(val_loader)

        print(f"Valid Dice: {avg_val_dice:.4f} | Valid Jaccard: {avg_val_jaccard:.4f}")

        if avg_val_dice > best_dice:
            print(f"Valid Score Improved ({best_dice:.4f} ---> {avg_val_dice:.4f})")
            best_dice = avg_val_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_dir)
            print("Model Saved")

        print(f"Train : {train_loss_sum / len(train_loader):.4f} | Valid : {val_loss_sum / len(val_loader):.4f}")
        print(f"Epoch {epoch + 1}/{epochs} completed in {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.0f}s\n")

    logger.info(f'Complete training ---------------------------------------------------- \n The best epoch is {best_epoch + 1}')
    print(f'Complete training ---------------------------------------------------- \n The best epoch is {best_epoch + 1}')
    
def test(config, model, model_dir, test_loader, criterion, logger, test_results_dir):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_sum = 0
    jaccard_test_sum = 0
    loss_test_sum = 0
    num_test = 0

    for batch_id, batch in enumerate(test_loader):
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        batch_len = img.shape[0]

        with torch.no_grad():
            output = torch.sigmoid(model(img))

            # Calculate loss
            loss = criterion[0](output, label) + criterion[1](output, label)
            loss_test_sum += loss.item() * batch_len

            # Calculate metrics
            output = threshold_predictions(output)  # Apply thresholding
            label = label.cpu().numpy()
            dice_test_sum += metrics.dc(output.cpu().numpy(), label) * batch_len
            jaccard_test_sum += metrics.jc(output.cpu().numpy(), label) * batch_len

            num_test += batch_len
            if config.debug:
                break

    # Logging results for one dataset
    loss_test_epoch = loss_test_sum / num_test
    dice_test_epoch = dice_test_sum / num_test
    jaccard_test_epoch = jaccard_test_sum / num_test

    # Logging average and store results
    with open(test_results_dir, 'w') as f:
        f.write(f'loss: {loss_test_epoch:.5f}, Dice_score: {dice_test_epoch:.4f}, Jaccard: {jaccard_test_epoch:.4f}')

    # Print
    logger.info('========================================================================================')
    logger.info(f'Test || Average loss: {loss_test_epoch:.5f}, Dice score: {dice_test_epoch:.4f}, Jaccard: {jaccard_test_epoch:.4f}')
    print('========================================================================================')
    print(f'Test || Average loss: {loss_test_epoch:.5f}, Dice score: {dice_test_epoch:.4f}, Jaccard: {jaccard_test_epoch:.4f}')

if __name__ == '__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str, default='config/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='polypgen')
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

    exp_dir = f'{config.data.save_folder}/{args.exp}_{config.data.supervised_ratio}/fold{args.fold}'
    os.makedirs(exp_dir, exist_ok=True)
    best_model_dir = f'{exp_dir}/best.pth'
    print(best_model_dir)
    test_results_dir = f'{exp_dir}/test_results.txt'

    # Store yml file
    if not config.debug:
        yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))

    main(config)
