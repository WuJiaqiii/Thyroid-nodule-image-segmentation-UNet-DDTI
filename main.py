import os
import argparse
from PIL import Image

# from models.model import UNet
# from models.vnet import ImprovedVNet
# from models.mores import AttentionUNet, ResUNet, ASPPUNet, TransUNet, VNet2D
from models.mod import *
from data.data_loader import MedicalDataset, create_dataloader

from utils.utils import create_logger, set_seed, Config
from utils.trainer import Trainer
from utils.transforms import *

import yaml

def get_parser():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataset config
    parser.add_argument('--dataset_path', default='data/dataset', type=str)
    parser.add_argument('--dataset', default='DDTI', type=str)

    parser.add_argument('--checkpoint_path', default='/root/Desktop/Thyroid-nodule-image-segmentation-UNet-DDTI/experiments/ResUNet_20250601_194117/models/ResUNet_best.pth', type=str)
    parser.add_argument('--config_path', default=None, type=str)

    ## data argument config
    parser.add_argument('--p_crop', default=0, type=float)
    parser.add_argument('--use_elastic', action='store_true')
    parser.add_argument('--use_speckle', action='store_true')
    parser.add_argument('--use_tgc',     action='store_true')
    parser.add_argument('--use_clahe',   action='store_true')
    
    parser.add_argument('--use_mixup',   action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--mixup_prob', type=float, default=0.3)

    ## model config
    parser.add_argument('--model_type', default='ResUNet', type=str)

    ## loss config
    parser.add_argument('--bce_ratio', type=float, default=1)
    parser.add_argument('--dice_ratio', type=float, default=0)
    parser.add_argument('--focal_ratio', type=float, default=1)
    parser.add_argument('--boundary_ratio', type=float, default=0)

    ## train config
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--early_stop_patience', default=50, type=int)
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
        tfs.append(ElasticDeform(p=0.25))
    tfs += [
        Flip(0.5),
        Rotate(0.5),
        AdjustBrightness(0.5)
    ]
    if cfg.use_speckle:
        tfs.append(SpeckleNoise(p=0.3))
    if cfg.use_tgc:
        tfs.append(TGCAugment(p=0.25))
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
    
    # if os.path.isfile(config.config_path):
    #     with open(config.config_path, "r", encoding="utf-8") as f:
    #         cfg = yaml.safe_load(f)
    #         model_cfg   = cfg["model"]
    #         model_type  = model_cfg["model_type"]   # 字符串 "VNet2D"
    #         model_kwargs = model_cfg["kwargs"]      # dict: {"in_channels":1, …}
    # else:
    #     logger.error(f'未找到配置文件：{config.config_path}')
    #     raise FileNotFoundError(f'未找到配置文件：{config.config_path}')
    
    config.model_type = "ResUNet"
    
    model = ResUNet()
    
    # if config.model_type == 'UNet':
    #     model = UNet(**model_kwargs)
    # elif config.model_type == 'VNet2D':
    #     model = VNet2D(**model_kwargs)
    # elif config.model_type == 'ImprovedVNet':
    #     model = ImprovedVNet(**model_kwargs)
    # elif config.model_type == 'TransUNet':
    #     model = TransUNet(**model_kwargs)
    # elif config.model_type == 'ResUNet':
    #     model = ResUNet(**model_kwargs)
    # elif config.model_type == 'ASPPUNet':
    #     model = ASPPUNet(**model_kwargs)
    # elif config.model_type == 'AttentionUNet':
    #     model = AttentionUNet(**model_kwargs)
    # else:
    #     logger.error('Inplemented model')
    #     raise(NotImplementedError())
    if os.path.isfile(config.checkpoint_path):
        model.load_state_dict(torch.load(config.checkpoint_path, weights_only=True))

    # --- 统计并打印可训练参数量 ------------------------
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_num = count_params(model)
    param_str = f"{param_num/1e6:.2f}M ({param_num:,})"
    logger.info(f"Model: {config.model_type} | Trainable params: {param_str}")
    print(f"[PARAMS] {config.model_type},{param_num}")   # 方便 bash 捕获
    # ---------------------------------------------------

    trainer = Trainer(config, (train_dataloader, val_dataloader, test_dataloader), logger, model)

    # trainer.train()
    trainer.test()

if __name__ == "__main__":

    args = get_parser()
    main(args)