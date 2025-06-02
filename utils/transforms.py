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
    def __init__(self, alpha=(20,40), sigma=(6,10), p=0.3):
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
    def __init__(self, num_bins=10, gain=(0.8,1.2), p=0.5):
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
    def __init__(self, clip=2.0, grid=(4,4), p=0.3):
        self.clip, self.grid, self.p = clip, grid, p
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    def __call__(self, img, mask):
        if random.random() > self.p: return img, mask
        img_np = np.array(img)
        img_eq = self.clahe.apply(img_np)
        return Image.fromarray(img_eq), mask


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