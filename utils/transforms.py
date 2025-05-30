import sys
sys.path.append('./')

import torch
import torchvision.transforms.functional as TF
# from data import MedicalDataset
import matplotlib.pyplot as plt
import random

import random, numpy as np, cv2
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

# ---------- 1. Elastic Deformation ----------
class ElasticDeform:
    def __init__(self, alpha=(80,120), sigma=(8,12), p=0.7):
        self.alpha, self.sigma, self.p = alpha, sigma, p

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask
        # PIL → numpy
        img_np  = np.array(img)
        mask_np = np.array(mask)
        h, w = img_np.shape[:2]

        alpha = random.uniform(*self.alpha)
        sigma = random.uniform(*self.sigma)

        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1),
                              ksize=(17,17), sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1),
                              ksize=(17,17), sigmaX=sigma) * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        img_def  = cv2.remap(img_np,  map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_def = cv2.remap(mask_np, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(img_def), Image.fromarray(mask_def)

# ---------- 2. Speckle Noise ----------
class SpeckleNoise:
    def __init__(self, sigma=(0.05,0.15), p=0.5):
        self.sigma, self.p = sigma, p
    def __call__(self, img, mask):
        if random.random() > self.p: return img, mask
        img_np = np.array(img).astype(np.float32) / 255.
        noise  = np.random.normal(0, random.uniform(*self.sigma), img_np.shape)
        img_np = img_np + img_np * noise
        img_np = np.clip(img_np*255., 0, 255).astype(np.uint8)
        return Image.fromarray(img_np), mask

# ---------- 3. TGC 增强 ----------
class TGCAugment:
    """按深度分环随机增益；num_bins=10 表示分 10 个水平条带"""
    def __init__(self, num_bins=10, gain=(0.6,1.4), p=0.5):
        self.num_bins, self.gain, self.p = num_bins, gain, p
    def __call__(self, img, mask):
        if random.random() > self.p: return img, mask
        img_np = np.array(img).astype(np.float32)
        h, w = img_np.shape[:2]
        bin_h = h // self.num_bins
        for i in range(self.num_bins):
            g = random.uniform(*self.gain)
            img_np[i*bin_h : (i+1)*bin_h] *= g
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np), mask

# ---------- 4. CLAHE ----------
class CLAHE:
    def __init__(self, clip=2.0, grid=(8,8), p=0.3):
        self.clip, self.grid, self.p = clip, grid, p
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    def __call__(self, img, mask):
        if random.random() > self.p: return img, mask
        img_np = np.array(img)
        img_eq = self.clahe.apply(img_np)
        return Image.fromarray(img_eq), mask

class MaskCrop(object):
    """
    根据 mask 的前景区域裁剪图像和 mask，
    并在四周额外留出一定像素 / 百分比的 margin。
    """
    def __init__(self, margin=20):          # margin 可以写成 int 或 float
        self.margin = margin                # int: 直接像素；float(0~1): bbox 边长比例

    def __call__(self, image, mask):
        import numpy as np
        import torchvision.transforms.functional as TF

        # mask -> numpy，找前景坐标
        m = np.array(mask)
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:    # 极端情况：mask 全黑
            return image, mask

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        # 计算 margin
        if isinstance(self.margin, float):
            h, w = mask.size[::-1]          # PIL: size=(W,H)
            pad_y = int((y1 - y0) * self.margin)
            pad_x = int((x1 - x0) * self.margin)
        else:
            pad_y = pad_x = int(self.margin)

        # 加 margin & 边界裁剪
        y0 = max(y0 - pad_y, 0)
        y1 = min(y1 + pad_y, mask.height)
        x0 = max(x0 - pad_x, 0)
        x1 = min(x1 + pad_x, mask.width)

        # PIL 裁剪：crop(box) 的 box = (left, upper, right, lower)
        image = TF.crop(image, y0, x0, y1 - y0, x1 - x0)
        mask  = TF.crop(mask , y0, x0, y1 - y0, x1 - x0)
        return image, mask

class RandomMaskCropOrFull:
    """
    以概率 p_crop 对 (image, mask) 先做 MaskCrop，再走后续增强；
    否则保留全幅，直接进入后续增强。
    其余 Transform 列表由外层 Compose 负责。
    """
    def __init__(self, mask_crop_fn, p_crop=0.7):
        """
        mask_crop_fn : 一个可调用对象，执行真正的 ROI 裁剪
                       eg. MaskCrop(margin=0.1)
        p_crop       : 裁剪分支的采样概率
        """
        self.mask_crop_fn = mask_crop_fn
        self.p_crop = p_crop

    def __call__(self, img, mask):
        if random.random() < self.p_crop:
            # —— 裁剪分支 ——
            img, mask = self.mask_crop_fn(img, mask)
        # —— 全幅分支什么都不做 ——
        return img, mask

class AdjustBrightness(object):
    """调整图片亮度"""
    def __init__(self, adjust_prob):
        self.adjust_prob = adjust_prob
    
    def __call__(self, image, mask):
        if random.random() < self.adjust_prob:
            brightness_factor = random.uniform(0.5, 1.5)
            image = TF.adjust_brightness(image, brightness_factor=brightness_factor)
        return image, mask
    
class RandomCrop(object):
    """随机裁剪图片"""
    def __init__(self, crop_prob, crop_width, crop_height):
        self.crop_prob = crop_prob
        self.crop_width = crop_width
        self.crop_height = crop_height
    
    def __call__(self, image, mask):
        if random.random() < self.crop_prob:
            width, height = image.size
            # crop_width, crop_height = self.crop_size
            top = random.randint(0, height - self.crop_height)
            left = random.randint(0, width - self.crop_width)

            image = TF.crop(image, top, left, self.crop_height, self.crop_width)
            mask = TF.crop(mask, top, left, self.crop_height, self.crop_width)

        return image, mask

class Flip(object):
    """翻转图像"""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
    
    def __call__(self, image, mask):
        # 水平翻转
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 垂直翻转
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        return image, mask

class Rotate(object):
    def __init__(self, rotate_prob):
        self.rotate_prob = rotate_prob
    
    def __call__(self, image, mask):
        if random.random() < self.rotate_prob:
            random_angle = random.uniform(-180, 180)
            image = TF.rotate(img=image, angle=random_angle)
            mask = TF.rotate(img=mask, angle=random_angle)
        return image, mask

class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, mask):
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask