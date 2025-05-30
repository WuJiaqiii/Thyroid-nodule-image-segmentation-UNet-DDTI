import os
import argparse
from PIL import Image

from models.model import UNet
from models.vnet import ImprovedVNet
from models.mores import AttentionUNet, ResUNet, ASPPUNet, TransUNet, VNet2D
from data.data_loader import MedicalDataset, create_dataloader

from utils.utils import create_logger, set_seed, Config
from utils.trainer import Trainer
from utils.transforms import *

def get_parser():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataset config
    parser.add_argument('--dataset_path', default='data/dataset', type=str)
    parser.add_argument('--dataset', default='DDTI', type=str)

    parser.add_argument('--checkpoint_path', default='model_last.pth', type=str)

    ## data argument config
    parser.add_argument('--p_crop', default=0, type=float)
    parser.add_argument('--use_elastic', action='store_true')
    parser.add_argument('--use_speckle', action='store_true')
    parser.add_argument('--use_tgc',     action='store_true')
    parser.add_argument('--use_clahe',   action='store_true')

    ## model config
    parser.add_argument('--model_type', default='UNet', type=str)

    ## loss config
    parser.add_argument('--bce_ratio', type=float, default=1)
    parser.add_argument('--dice_ratio', type=float, default=1)
    parser.add_argument('--focal_ratio', type=float, default=1)
    parser.add_argument('--boundary_ratio', type=float, default=1)

    ## train config
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--early_stop_patience', default=500, type=int)
    parser.add_argument('--alpha', type=float, default=2)

    ## other config
    parser.add_argument('--use_data_parallel', type=bool, default=True, help="Whether to use DataParallel for multi-GPU training")
    parser.add_argument('--use_amp_autocast', type=bool, default=False)
    
    args = parser.parse_args()
    
    return args

def build_train_transform(cfg):
    # tfs = [RandomMaskCropOrFull(
    #     mask_crop_fn = MaskCrop(margin=0.1),
    #     p_crop       = cfg.p_crop 
    # )]
    tfs = []
    
    if cfg.use_elastic:
        tfs.append(ElasticDeform(p=0.7))
    tfs += [
        Flip(0.5),
        Rotate(0.5),
        AdjustBrightness(0.5)
    ]
    if cfg.use_speckle:
        tfs.append(SpeckleNoise(p=0.5))
    if cfg.use_tgc:
        tfs.append(TGCAugment(p=0.5))
    if cfg.use_clahe:
        tfs.append(CLAHE(p=0.3))

    tfs += [
        Resize((512,512)),
        ToTensor()
    ]
    return Compose(tfs)

def main(args):
    
    set_seed(seed=42)
    config = Config(args)
    logger = create_logger(os.path.join(config.log_dir, f"train_log.log"))

    train_transform = build_train_transform(config)
    test_transform = Compose([Resize((512, 512)), ToTensor()])

    train_dataset = MedicalDataset(os.path.join(config.dataset_path, 'train'), os.path.join(config.dataset_path, 'train_mask'), train_transform)
    val_dataset = MedicalDataset(os.path.join(config.dataset_path, 'val'), os.path.join(config.dataset_path, 'val_mask'), test_transform)
    test_dataset = MedicalDataset(os.path.join(config.dataset_path, 'test'), os.path.join(config.dataset_path, 'test_mask'), test_transform)
    
    train_dataloader = create_dataloader(train_dataset, config, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, config, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, config, shuffle=True)
    
    if config.model_type == 'UNet':
        model = UNet(in_channels=1, out_channels=1)
    elif config.model_type == 'VNet':
        model = VNet2D(in_channels=1, out_channels=1)
    elif config.model_type == 'ImprovedVNet':
        model = ImprovedVNet(in_channels=1, num_classes=1)
    elif config.model_type == 'TransUNet':
        model = TransUNet(in_channels=1, out_channels=1)
    elif config.model_type == 'ResUNet':
        model = ResUNet(in_channels=1, out_channels=1)
    elif config.model_type == 'ASPPUNet':
        model = ASPPUNet(in_channels=1, out_channels=1)
    elif config.model_type == 'AttentionUNet':
        model = AttentionUNet(in_channels=1, out_channels=1)
    else:
        logger.error('Inplemented model')
        raise(NotImplementedError())
    # if os.path.isfile(config.checkpoint_path):
    #     model.load_state_dict(torch.load(config.checkpoint_path, weights_only=True))

    trainer = Trainer(config, (train_dataloader, val_dataloader, test_dataloader), logger, model)

    trainer.train()
    trainer.test()

if __name__ == "__main__":

    args = get_parser()
    main(args)