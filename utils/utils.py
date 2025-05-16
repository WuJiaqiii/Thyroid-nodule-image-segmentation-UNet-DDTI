import os
import torch
import logging
import numpy as np
import random
from datetime import datetime, timedelta, timezone
import yaml
import pytz
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

class Config:        
    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.make_dir()
        self.save_config()
        
    def save_config(self):
        file_path = os.path.join(self.cfg_dir, 'config.yaml')
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f)
            
    def make_dir(self):
        
        self.base_dir = 'experiments'
        os.makedirs(self.base_dir, exist_ok=True)
        
        current_time = datetime.now(pytz.utc)
        current_time = current_time.astimezone(pytz.timezone("Asia/Shanghai"))
        self.cfg_dir = '%s/%s' % (self.base_dir, str(current_time.strftime("%Y%m%d_%H%M%S")))
        self.model_dir = '%s/models' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.result_dir = '%s/result' % self.cfg_dir
        os.makedirs(self.cfg_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_logger(filename):
    
    def custom_time(*args):
        utc_plus_8 = datetime.now(tz=timezone.utc) + timedelta(hours=8)
        return utc_plus_8.timetuple()
        
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter.converter = custom_time
    ch.setFormatter(formatter)

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

class EarlyStopping:
    def __init__(self, logger, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.logger.info(
                f'--Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'--EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.logger.info(
                f'--Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0

def save_confusion_matrix(targets, preds, config, epoch):
    cm = confusion_matrix(targets, preds)
    class_names = [name.decode() for name in config.classes.keys()]

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 10}, 
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    ax = plt.gca()

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label_text = label.get_text()
        class_index = list(config.classes.values())[list(config.classes.keys()).index(label_text.encode())]
        if class_index in config.known_classes:
            label.set_fontweight('bold')

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig(os.path.join(config.result_dir, f'epoch_{epoch + 1}_confusion_matrix'), dpi=500, bbox_inches='tight')
    plt.close()

def calculate_iou(pred, target):
    """纯 NumPy 版二值掩码 IoU"""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union        = np.logical_or(pred, target).sum()
    return intersection / union

def calculate_acc(pred, target):
    pred = pred.astype(int)
    target = target.astype(int)
    correct = (pred == target).sum()
    total   = pred.size
    return correct / total

def calculate_precision_recall_f1(pred, target):
    pred = pred.astype(int)
    target = target.astype(int)

    TP = np.logical_and(pred==1, target==1).sum()
    FP = np.logical_and(pred==1, target==0).sum()
    FN = np.logical_and(pred==0, target==1).sum()

    precision = TP / (TP + FP) if TP+FP>0 else 0.0
    recall    = TP / (TP + FN) if TP+FN>0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision+recall>0 else 0.0
    return precision, recall, f1

def init_weights(m):
    """初始化模型权重"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)