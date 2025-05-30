import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from skimage import measure

from models.loss import DiceLoss, CompositeLoss, FocalTverskyLoss, BoundaryLoss
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.utils import AverageMeter, EarlyStopping, calculate_acc, calculate_iou, calculate_precision_recall_f1

class Trainer:
    def __init__(self, config, data_loader, logger, model):
        
        self.config = config
        self.logger = logger
        self.device = self.config.device

        self.train_loader, self.val_loader, self.test_loader = data_loader
        
        if torch.cuda.device_count() > 1 and self.config.use_data_parallel:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for training...")
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)

        self.scaler = GradScaler(enabled=self.config.use_amp_autocast and self.device == torch.device('cuda'), device="cuda")

        self.criterion_dice = DiceLoss()
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_focal = FocalTverskyLoss()
        self.criterion_boundary = BoundaryLoss()
        self.criterion = CompositeLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2, eta_min=0)
            
        self.early_stopping = EarlyStopping(logger=self.logger, patience=self.config.early_stop_patience, delta=0)
        self.writer = SummaryWriter(log_dir=config.result_dir)
        
    def train_one_epoch(self, epoch):
        
        bce_loss_record = AverageMeter()
        dice_loss_record = AverageMeter()
        focal_loss_record = AverageMeter()
        boundary_loss_record = AverageMeter()
        loss_record = AverageMeter()
        
        self.model.train()
        preds, targets = [], []
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)):

            images, masks = batch
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.use_amp_autocast and self.device == torch.device('cuda'), device_type='cuda'):
                logits = self.model(images)
                loss_bce = self.criterion_bce(logits, masks)
                loss_dice = self.criterion_dice(logits, masks)
                loss_focal = self.criterion_focal(logits, masks)
                loss_boundary = self.criterion_boundary(logits, masks)

                loss = self.config.bce_ratio * loss_bce + self.config.dice_ratio * loss_dice + self.config.focal_ratio * loss_focal + self.config.boundary_ratio * loss_boundary
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bce_loss_record.update(loss_bce.item(), masks.size(0))
            dice_loss_record.update(loss_dice.item(), masks.size(0))
            focal_loss_record.update(loss_focal.item(), masks.size(0))
            boundary_loss_record.update(loss_boundary.item(), masks.size(0))
            loss_record.update(loss.item(), masks.size(0))

            preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
            targets.append(masks.cpu().numpy())
        
        preds, targets = np.concatenate(preds), np.concatenate(targets)
        acc = calculate_acc(preds, targets)
        precision, recall, f1 = calculate_precision_recall_f1(preds, targets)
        iou = calculate_iou(preds, targets)
        self.logger.info(f'Train Epoch: {epoch + 1}, Avg Loss: {(loss_record.avg):.4f}')
        self.logger.info(f'BCE Loss: {bce_loss_record.avg:.4f}, Dice Loss: {dice_loss_record.avg:.4f}, Focal Loss: {focal_loss_record.avg:.4f}, Boundary Loss: {boundary_loss_record.avg:.4f}')
        self.logger.info(f'acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, IoU: {iou:.4f}')
        self.writer.add_scalar("BCE Loss/Train", bce_loss_record.avg, epoch)
        self.writer.add_scalar("Dice Loss/Train", dice_loss_record.avg, epoch)
        self.writer.add_scalar("Focal Loss/Train", focal_loss_record.avg, epoch)
        self.writer.add_scalar("Boundary Loss/Train", boundary_loss_record.avg, epoch)
        self.writer.add_scalar("Acc/Train", acc, epoch)
        self.writer.add_scalar("Precision/Train", precision, epoch)
        self.writer.add_scalar("Recall/Train", recall, epoch)
        self.writer.add_scalar("F1/Train", f1, epoch)
        self.writer.add_scalar("IoU/Train", iou, epoch)

    @torch.no_grad() 
    def validate(self, epoch):
        
        bce_loss_record = AverageMeter()
        dice_loss_record = AverageMeter()
        focal_loss_record = AverageMeter()
        boundary_loss_record = AverageMeter()
        loss_record = AverageMeter()

        self.model.eval()
        preds, targets = [], []
        start = time.time()
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", leave=True)):

            images, masks = batch
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)

            with autocast(enabled=self.config.use_amp_autocast and self.device == torch.device('cuda'), device_type='cuda'):
                logits = self.model(images)
                loss_bce = self.criterion_bce(logits, masks)
                loss_dice = self.criterion_dice(logits, masks)
                loss_focal = self.criterion_focal(logits, masks)
                loss_boundary = self.criterion_boundary(logits, masks)
                loss = self.config.bce_ratio * loss_bce + self.config.dice_ratio * loss_dice + self.config.focal_ratio * loss_focal + self.config.boundary_ratio * loss_boundary

            bce_loss_record.update(loss_bce.item(), masks.size(0))
            dice_loss_record.update(loss_dice.item(), masks.size(0))
            focal_loss_record.update(loss_focal.item(), masks.size(0))
            boundary_loss_record.update(loss_boundary.item(), masks.size(0))
            loss_record.update(loss.item(), masks.size(0))
            
            preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
            targets.append(masks.cpu().numpy())   
         
        preds, targets = np.concatenate(preds), np.concatenate(targets)
        acc = calculate_acc(preds, targets)
        precision, recall, f1 = calculate_precision_recall_f1(preds, targets)
        iou = calculate_iou(preds, targets)
        self.logger.info(f'Validate Epoch: {epoch + 1}, Avg Loss: {(loss_record.avg):.4f}')
        self.logger.info(f'BCE Loss: {bce_loss_record.avg:.4f}, Dice Loss: {dice_loss_record.avg:.4f}, Focal Loss: {focal_loss_record.avg:.4f}, Boundary Loss: {boundary_loss_record.avg:.4f}')
        self.logger.info(f'acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, IoU: {iou:.4f}')
        self.writer.add_scalar("BCE Loss/Validate", bce_loss_record.avg, epoch)
        self.writer.add_scalar("Dice Loss/Validate", dice_loss_record.avg, epoch)
        self.writer.add_scalar("Focal Loss/Train", focal_loss_record.avg, epoch)
        self.writer.add_scalar("Boundary Loss/Train", boundary_loss_record.avg, epoch)
        self.writer.add_scalar("Acc/Validate", acc, epoch)
        self.writer.add_scalar("Precision/Validate", precision, epoch)
        self.writer.add_scalar("Recall/Validate", recall, epoch)
        self.writer.add_scalar("F1/Validate", f1, epoch)
        self.writer.add_scalar("IoU/Validate", iou, epoch)

        return loss_record.avg, iou

    def train(self):
        
        best_val_iou = np.inf
        for epoch in range(self.config.epochs):
            
            self.train_one_epoch(epoch)
            val_loss, val_iou = self.validate(epoch)
            
            self.scheduler.step()

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_path = os.path.join(self.config.model_dir, f'{self.config.model_type}_best.pth')
                if torch.cuda.device_count() > 1 and self.config.use_data_parallel:
                    torch.save(self.model.module.state_dict(), best_path)
                else:
                    torch.save(self.model.state_dict(), best_path)
                
                self.logger.info(f"--Best model saved at epoch {epoch + 1} with IoU: {best_val_iou:.4f}")
                
            self.early_stopping(-val_iou, self)
            if self.early_stopping.early_stop:
                self.logger.info("--Early stopping triggered")
                break

        if torch.cuda.device_count() > 1 and self.config.use_data_parallel:
            torch.save(self.model.module.state_dict(), os.path.join(self.config.model_dir, f'{self.config.model_type}_last.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, f'{self.config.model_type}_last.pth'))
        
        self.writer.close()
    
    @torch.no_grad()
    def test(self):
        self.logger.info('------------------Starting Testing Model------------------')

        self.model.eval()
        all_imgs, all_masks, all_preds = [], [], []
        for images, masks in tqdm(self.test_loader, desc='Testing Model', leave=True):
            images, masks = images.to(self.device), masks.to(self.device)
            logits = self.model(images)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

            all_imgs.append(images.cpu().numpy())             # (B, C, H, W)
            all_masks.append(masks.cpu().numpy().astype(np.uint8))
            all_preds.append(preds)

        all_imgs  = np.concatenate(all_imgs,  axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        total = all_imgs.shape[0]

        for batch_start in range(0, total, 20):
            batch_end = min(batch_start + 20, total)
            n = batch_end - batch_start

            fig, axes = plt.subplots(5, 4, figsize=(16, 20))
            axes = axes.flatten()

            for i in range(n):
                idx = batch_start + i
                img  = all_imgs[idx].transpose(1, 2, 0).squeeze()  # H×W
                mask = all_masks[idx].squeeze()                   # H×W binary
                pred = all_preds[idx].squeeze()                   # H×W binary
                ax = axes[i]

                ax.imshow(img, cmap='gray')
                # 真值边界 (蓝色)
                for contour in measure.find_contours(mask, level=0.5):
                    ax.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1)
                # 预测边界 (红色)
                for contour in measure.find_contours(pred, level=0.5):
                    ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

                ax.axis('off')

            for j in range(n, 20):
                axes[j].axis('off')

            plt.tight_layout()
            save_path = os.path.join(self.config.result_dir, f'test_boundaries_{batch_start//20}.png')
            plt.savefig(save_path)
            plt.close(fig)