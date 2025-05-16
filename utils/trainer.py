import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
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

        if self.config.use_amp_autocast and self.device == 'cuda':
            self.scaler = GradScaler(device="cuda")

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2, eta_min=0)
            
        self.early_stopping = EarlyStopping(logger=self.logger, patience=self.config.early_stop_patience, delta=0)
        self.writer = SummaryWriter(log_dir=config.result_dir)
        
    def train_one_epoch(self, epoch):
        
        loss_record = AverageMeter()
        
        self.model.train()
        preds, targets = [], []
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)):

            images, masks = batch
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)

            self.optimizer.zero_grad()

            if self.config.use_amp_autocast:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                self.optimizer.step()

            loss_record.update(loss.item(), masks.size(0))
            preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
            targets.append(masks.cpu().numpy())
        
        preds, targets = np.concatenate(preds), np.concatenate(targets)
        acc = calculate_acc(preds, targets)
        precision, recall, f1 = calculate_precision_recall_f1(preds, targets)
        iou = calculate_iou(preds, targets)
        self.logger.info(f'Train Epoch: {epoch + 1}, Avg Loss: {loss_record.avg:.4f}, acc: {acc:.4f}, \
                         precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, IoU: {iou:.4f}')
        self.writer.add_scalar("Loss/Train", loss_record.avg, epoch)
        self.writer.add_scalar("Acc/Train", acc, epoch)
        self.writer.add_scalar("Precision/Train", precision, epoch)
        self.writer.add_scalar("Recall/Train", recall, epoch)
        self.writer.add_scalar("F1/Train", f1, epoch)
        self.writer.add_scalar("IoU/Train", iou, epoch)

    @torch.no_grad() 
    def validate(self, epoch):
        
        loss_record = AverageMeter()

        self.model.eval()

        preds, targets = [], []
        start = time.time()
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", leave=True)):

            images, masks = batch
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)

            if self.config.use_amp_autocast:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            loss_record.update(loss.item(), masks.size(0))
            
            preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
            targets.append(masks.cpu().numpy())   
         
        preds, targets = np.concatenate(preds), np.concatenate(targets)
        acc = calculate_acc(preds, targets)
        precision, recall, f1 = calculate_precision_recall_f1(preds, targets)
        iou = calculate_iou(preds, targets)
        end = time.time()

        self.logger.info(f'Validate Epoch: {epoch + 1}, Avg Loss: {loss_record.avg:.4f}, acc: {acc:.4f}, '
                         f'precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, IoU: {iou:.4f}')
        self.logger.info(f'Spend: {(end - start)/60.0:.2f} minutes for evaluation')
        self.writer.add_scalar("Loss/Validation", loss_record.avg, epoch)
        self.writer.add_scalar("Acc/Validation", acc, epoch)
        self.writer.add_scalar("Precision/Validation", precision, epoch)
        self.writer.add_scalar("Recall/Validation", recall, epoch)
        self.writer.add_scalar("F1/Validation", f1, epoch)
        self.writer.add_scalar("IoU/Validation", iou, epoch)

        return loss_record.avg, iou

    def train(self):
        
        best_val_iou = np.inf
        for epoch in range(self.config.epochs):
            
            self.train_one_epoch(epoch)
            val_loss, val_iou = self.validate(epoch)
            
            self.scheduler.step()

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_path = os.path.join(self.config.model_dir, f'model_best.pth')
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
            torch.save(self.model.module.state_dict(), os.path.join(self.config.model_dir, f'model_last.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, f'model_last.pth'))
        
        self.writer.close()

    def test(self):
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(self.test_loader, desc='Testing Model', leave=True)):

                images, masks = images.to(self.config.device), masks.to(self.config.device)
                logits = self.model(images)  
                pred = (torch.sigmoid(logits) > 0.5)

                plt.figure(figsize=(20, 20), dpi=80)
                for i in range(self.config.batch_size):
                    ax = plt.subplot(3, 4, i + 1)
                    ax.imshow(images[i].permute(1, 2, 0).cpu())
                    ax = plt.subplot(3, 4, i + 1 + 4)
                    ax.imshow(masks[i].permute(1, 2, 0).cpu())
                    ax = plt.subplot(3, 4, i + 1 + 8)
                    ax.imshow(pred[i].permute(1, 2, 0).cpu())
                
                plt.savefig(os.path.join(self.config.result_dir, f'test_figure{batch_idx}.png'))