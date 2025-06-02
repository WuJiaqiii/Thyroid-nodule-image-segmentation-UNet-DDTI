import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from skimage import measure
import random

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
            
            if random.random() < self.config.mixup_prob and self.config.use_mixup:
                # 1. 从 Beta(alpha, alpha) 中采样一个 lambda
                lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)

                # 2. 打乱当前 batch 的索引，构造 permuted_idx
                batch_size = images.size(0)
                permuted_idx = torch.randperm(batch_size).cuda()

                # 3. 根据打乱索引取第二份图像和 Mask
                images_shuffled = images[permuted_idx]
                masks_shuffled  = masks[permuted_idx]

                # 4. 对图像和 Mask 做线性插值
                #    注意：masks 原本是 0/1 的二值（或者 float {0.0,1.0}），
                #    插值后会得到 [0,1] 之间的“软标签”mask
                images = lam * images + (1.0 - lam) * images_shuffled
                masks  = lam * masks  + (1.0 - lam) * masks_shuffled
            

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
        self.writer.add_scalar("Focal Loss/Validate", focal_loss_record.avg, epoch)
        self.writer.add_scalar("Boundary Loss/Validate", boundary_loss_record.avg, epoch)
        self.writer.add_scalar("Acc/Validate", acc, epoch)
        self.writer.add_scalar("Precision/Validate", precision, epoch)
        self.writer.add_scalar("Recall/Validate", recall, epoch)
        self.writer.add_scalar("F1/Validate", f1, epoch)
        self.writer.add_scalar("IoU/Validate", iou, epoch)

        return loss_record.avg, iou

    def train(self):
        
        best_val_iou = -np.inf
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

        # 1. 遍历测试集、收集所有原始图、真值 mask、预测 mask
        for images, masks in tqdm(self.test_loader, desc='Testing Model', leave=True):
            images, masks = images.to(self.device), masks.to(self.device)
            logits = self.model(images)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)  # 二值化

            all_imgs.append(images.cpu().numpy())               # (B, C, H, W)
            all_masks.append(masks.cpu().numpy().astype(np.uint8))  # (B, 1, H, W) 或者 (B, H, W)
            all_preds.append(preds)                             # (B, 1, H, W) 或者 (B, H, W)

        # 2. 拼成一个整体数组
        all_imgs  = np.concatenate(all_imgs,  axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        total = all_imgs.shape[0]

        # 3. 计算全局像素级 TP, FP, FN, TN
        #    注意：假设 all_masks 与 all_preds 都是形状 (N, 1, H, W) 或者 (N, H, W) 的二值化 (0/1) uint8 数组
        #    先把它们都扁平化成 1D：
        flat_masks = all_masks.reshape(-1)
        flat_preds = all_preds.reshape(-1)

        # 真阳性：预测 1 & 实际 1
        TP = np.logical_and(flat_preds == 1, flat_masks == 1).sum()
        # 假阳性：预测 1 & 实际 0
        FP = np.logical_and(flat_preds == 1, flat_masks == 0).sum()
        # 假阴性：预测 0 & 实际 1
        FN = np.logical_and(flat_preds == 0, flat_masks == 1).sum()
        # 真阴性：预测 0 & 实际 0
        TN = np.logical_and(flat_preds == 0, flat_masks == 0).sum()

        # 4. 由 TP, FP, FN, TN 计算各指标（加上数值保护以免除零错误）
        eps = 1e-8
        ACC      = (TP + TN) / (TP + TN + FP + FN + eps)
        Precision= TP / (TP + FP + eps)
        Recall   = TP / (TP + FN + eps)
        F1       = 2 * Precision * Recall / (Precision + Recall + eps)
        IoU      = TP / (TP + FP + FN + eps)

        # 5. 将结果打印并记录到日志
        msg = (
            f"Test Metrics  —  Total Images: {total}\n"
            f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}\n"
            f"  ACC={ACC:.4f}, Precision={Precision:.4f}, "
            f"Recall={Recall:.4f}, F1={F1:.4f}, IoU={IoU:.4f}"
        )
        print(msg)
        self.logger.info(msg)

        # 6. 接着在画图部分保留原有“轮廓可视化”逻辑，把边界画到图片上并保存
        #    每 20 张拼成一个 5×4 的子图，然后写到 result_dir
        for batch_start in range(0, total, 20):
            batch_end = min(batch_start + 20, total)
            n = batch_end - batch_start

            fig, axes = plt.subplots(5, 4, figsize=(16, 20))
            axes = axes.flatten()

            for i in range(n):
                idx = batch_start + i
                # 原始图：可能形状是 (C, H, W)，要转成 (H, W)
                img  = all_imgs[idx].transpose(1, 2, 0).squeeze()
                mask = all_masks[idx].squeeze()   # (H, W)
                pred = all_preds[idx].squeeze()   # (H, W)
                ax = axes[i]

                ax.imshow(img, cmap='gray')
                # 真值边界（蓝色）
                for contour in measure.find_contours(mask, level=0.5):
                    ax.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1)
                # 预测边界（红色）
                for contour in measure.find_contours(pred, level=0.5):
                    ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

                ax.axis('off')

            # 若本批次不足 20 张，剩余子图隐藏
            for j in range(n, 20):
                axes[j].axis('off')

            plt.tight_layout()
            save_path = os.path.join(
                self.config.result_dir,
                f'test_boundaries_{batch_start // 20}.png'
            )
            plt.savefig(save_path)
            plt.close(fig)
    
    # @torch.no_grad()
    # def test(self):
    #     self.logger.info('------------------Starting Testing Model------------------')

    #     self.model.eval()
    #     all_imgs, all_masks, all_preds = [], [], []
    #     for images, masks in tqdm(self.test_loader, desc='Testing Model', leave=True):
    #         images, masks = images.to(self.device), masks.to(self.device)
    #         logits = self.model(images)
    #         preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

    #         all_imgs.append(images.cpu().numpy())             # (B, C, H, W)
    #         all_masks.append(masks.cpu().numpy().astype(np.uint8))
    #         all_preds.append(preds)

    #     all_imgs  = np.concatenate(all_imgs,  axis=0)
    #     all_masks = np.concatenate(all_masks, axis=0)
    #     all_preds = np.concatenate(all_preds, axis=0)
    #     total = all_imgs.shape[0]

    #     for batch_start in range(0, total, 20):
    #         batch_end = min(batch_start + 20, total)
    #         n = batch_end - batch_start

    #         fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    #         axes = axes.flatten()

    #         for i in range(n):
    #             idx = batch_start + i
    #             img  = all_imgs[idx].transpose(1, 2, 0).squeeze()  # H×W
    #             mask = all_masks[idx].squeeze()                   # H×W binary
    #             pred = all_preds[idx].squeeze()                   # H×W binary
    #             ax = axes[i]

    #             ax.imshow(img, cmap='gray')
    #             # 真值边界 (蓝色)
    #             for contour in measure.find_contours(mask, level=0.5):
    #                 ax.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1)
    #             # 预测边界 (红色)
    #             for contour in measure.find_contours(pred, level=0.5):
    #                 ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

    #             ax.axis('off')

    #         for j in range(n, 20):
    #             axes[j].axis('off')

    #         plt.tight_layout()
    #         save_path = os.path.join(self.config.result_dir, f'test_boundaries_{batch_start//20}.png')
    #         plt.savefig(save_path)
    #         plt.close(fig)