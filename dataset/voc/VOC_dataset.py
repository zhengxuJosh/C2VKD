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

voc_annopath = '/hpc/users/CONNECT/xuzheng/ECCV-TCC/data/VOCdevkit/VOC2012/Annotations'
root_voc2012 = '/hpc/users/CONNECT/xuzheng/ECCV-TCC/data/VOCdevkit/VOC2012/'
class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
              
    def __init__(self,root_dir, split='train_aug',use_difficult=False, is_train = True, augment = None, cutmix=False, base_size=520, crop_size=480, file_length=None, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
        self.root_voc2012= root_voc2012
        self.txt_path = root_dir
        self.use_difficult=use_difficult
        self.imgset=split
        self.cutmix = cutmix
        # '/hpc/users/hpcuser01/SSL/data/VOCdevkit/VOC2012/Annotations'
        self._annopath = voc_annopath
        self.num_class = 21
        self._imgpath = os.path.join(self.root_voc2012, "JPEGImages", "%s.jpg")
        self._labelpath = os.path.join(self.root_voc2012, "SegmentationClassAug", "%s.png")

        self._imgsetpath = os.path.join(self.txt_path, "%s.txt")
        self._labelsetpath = os.path.join(self.txt_path, "%s.txt")

        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.name2id=dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
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
        print("INFO=====>voc dataset init finished  ! !")

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
            "image need to be transfromed to tensor here"
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

        
    def get_img_info(self, index):
        img_id=self.img_ids[index]
        anno=ET.parse(self._annopath+'/'+img_id+'.xml').getroot()
        height = 0
        width = 0
        for obj in anno.iter("size"):
            
            height = int(obj.find('height').text)
            width = int(obj.find('width').text)
        return height,width

       
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size # 480
        # random scale (short edge)
        w, h = img.size

        # print('img size', img.size)
        # print('label size', mask.size)
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

        # print(f'ow:{ow}, oh:{oh}')
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # print('img2 size', img.size)
        # print('label2 size', mask.size)

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
        return torch.from_numpy(np.array(mask)).long()

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
