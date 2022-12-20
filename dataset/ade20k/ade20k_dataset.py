import torch
import xml.etree.ElementTree as ET
import os
import numpy as np
from torch.utils.data import dataloader
from torchvision import transforms
import random
from torchvision.io import read_image
import math
from utils.cutmix import cutmix
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import PIL


#root_dir="/home/yunhao/datasets/ADEChallengeData2016"
# split = "train_20210_ade20k"
# split = "val_2000_ade20k"
class ADE20KDataset(torch.utils.data.Dataset):

    def __init__(self,root_dir, split='train_aug',use_difficult=False, is_train =True, augment =None, cutmix=False, base_size=520, crop_size=480, file_length=None, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        self.ade_root_dir= root_dir
        self.use_difficult = use_difficult
        self.imgset = split
        self.cutmix = cutmix
        # '/hpc/users/hpcuser01/SSL/data/VOCdevkit/VOC2012/Annotations'
        self.num_class = len(ADE20KDataset.CLASSES_NAME) # 150
        self.txt_path = os.path.join(self.ade_root_dir, "ImageSets")
        if is_train:
            self._imgpath = os.path.join(self.ade_root_dir, "images", "training", "%s.jpg")
            self._labelpath = os.path.join(self.ade_root_dir, "annotations", "training", "%s.png")
        else:
            self.imgset = "val_2000_ade20k"
            self._imgpath = os.path.join(self.ade_root_dir, "images", "validation", "%s.jpg")
            self._labelpath = os.path.join(self.ade_root_dir, "annotations", "validation", "%s.png")

        self._imgsetpath = os.path.join(self.txt_path, "%s.txt")
        self._labelsetpath = os.path.join(self.txt_path, "%s.txt")

        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.name2id=dict(zip(ADE20KDataset.CLASSES_NAME,range(len(ADE20KDataset.CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}
        # self.mean=[0.485, 0.456, 0.406]
        # self.std=[0.229, 0.224, 0.225]
        self.train = is_train
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.norm_mean= norm_mean
        self.norm_std = norm_std
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # make img to [-1, 1]
            transforms.Normalize(self.norm_mean, self.norm_std),
        ])
        self.target_transform = None
        print("INFO=====>ADE20k dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,index):

        img_id=self.img_ids[index]
        _img = Image.open(self._imgpath%img_id).convert('RGB')
        _target = Image.open(self._labelpath%img_id)
        # synchrosized transform, img is pil, target is tensor now
        if self.train:
            _img, _target = self._sync_transform( _img, _target)
        else:
            _img, _target = self._val_sync_transform( _img, _target)
    
        # general resize, normalize and toTensor
        if self.img_transform is not None:
            # "** image need to be transfromed to tensor here **"
            _img = self.img_transform(_img)
        # is None for now
        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target
    
    def my_collate(self,batch):
        '''
        ** Resize images and retain original ratio **
        image = [item[0] for item in batch]
        class_label = [item[1][:,0] for item in batch]
        '''
        return dataloader.default_collate(batch)

       
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size # 480
        # random scale (short edge)
        w, h = img.size
        # range: (260 -> 1040); assume: (600, 400)
        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            # left top Right Bottom
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))

        # final transform, should be of size: (crop_size, crop_size)
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        # ADE20K ignore 0, send 0 to 255 and move other one step
        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask == 0] = 255
        mask = mask  - 1
        mask[mask==254] = 255
        return mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size # 480,480
        short_size = outsize
        w, h = img.size
        # make short_size to crop_size: 480 for resize
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _construct_new_file_names(self, length): # length is the required length
        """ now it is useless 
        """
        # 183 -> 91xx
        assert isinstance(length, int)
        print(f'len {length}, img {len(self.img_ids)}')
        files_len = len(self.img_ids) # 原来一轮迭代的长度

        # 仅使用小部分数据
        if length < files_len:
            return self.img_ids[:length]
        
        # 按照设定获得的一轮迭代的长度
        new_file_names = self.img_ids * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self.img_ids[i] for i in new_indices]
        
        self.img_ids = new_file_names
        print(f'{self.txt_path}: {len(self.img_ids)}')
        print(self.img_ids[0], self.img_ids[183])

        return None

    CLASSES_NAME = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]
