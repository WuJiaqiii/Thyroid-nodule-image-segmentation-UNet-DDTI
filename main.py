import os
import argparse

from models.model import UNet
from models.vnet import ImprovedVNet
from data.data_loader import MedicalDataset, create_dataloader

from utils.utils import create_logger, set_seed, Config
from utils.trainer import Trainer
from utils.transforms import *

def get_parser():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataset config
    parser.add_argument('--dataset_path', default='/root/Desktop/Thyroid-nodule-image-segmentation-UNet-DDTI/data/dataset', type=str)
    parser.add_argument('--dataset', default='DDTI', type=str)

    parser.add_argument('--checkpoint_path', default='/root/Desktop/Thyroid-nodule-image-segmentation-UNet-DDTI/model_last.pth', type=str)

    ## train config
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--early_stop_patience', default=20, type=int)
    parser.add_argument('--alpha', type=float, default=2)

    ## other config
    parser.add_argument('--use_data_parallel', type=bool, default=True, help="Whether to use DataParallel for multi-GPU training")
    parser.add_argument('--use_amp_autocast', type=bool, default=False)
    
    args = parser.parse_args()
    
    return args

def main(args):
    
    set_seed(seed=42)
    config = Config(args)
    logger = create_logger(os.path.join(config.log_dir, f"train_log.log"))

    train_transform = Compose([
        Flip(0.5),
        Rotate(0.5),
        AdjustBrightness(0.5),
        RandomCrop(0.5, 300, 300),
        Resize((512, 512)), 
        ToTensor()
    ])

    test_transform = Compose([
        Resize((512, 512)), 
        ToTensor()
    ])

    train_dataset = MedicalDataset(os.path.join(config.dataset_path, 'train'), os.path.join(config.dataset_path, 'train_mask'), train_transform)
    val_dataset = MedicalDataset(os.path.join(config.dataset_path, 'val'), os.path.join(config.dataset_path, 'val_mask'), test_transform)
    test_dataset = MedicalDataset(os.path.join(config.dataset_path, 'test'), os.path.join(config.dataset_path, 'test_mask'), test_transform)
    
    train_dataloader = create_dataloader(train_dataset, config, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, config, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, config, shuffle=True)

    
    # config.checkpoint_path = '/root/Desktop/Thyroid-nodule-image-segmentation-UNet-DDTI/experiments/20250516_161726/models/model_last.pth'
    # unet = UNet(in_channels=1, out_channels=1)
    # if config.checkpoint_path:
    #     unet.load_state_dict(torch.load(config.checkpoint_path, weights_only=True))

    model = ImprovedVNet()
    if os.path.isfile(config.checkpoint_path):
        model.load_state_dict(torch.load(config.checkpoint_path, weights_only=True))

    trainer = Trainer(config, (train_dataloader, val_dataloader, test_dataloader), logger, model)

    trainer.train()
    trainer.test()

if __name__ == "__main__":

    args = get_parser()
    main(args)