import torch
import torch.nn as nn
import torch.nn.functional as F 
from .seghead import SegFormerHead
from . import MixT

class Seg(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,size = 512):
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(MixT, backbone)()
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if backbone == 'mit_b0':
            if pretrained:
                state_dict = torch.load('/hpc/users/CONNECT/xuzheng/mit_b0.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict,)
        if backbone == 'mit_b1':
            if pretrained:
                state_dict = torch.load('/hpc/users/CONNECT/xuzheng/workplace/mit_b1.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict,)
        self.backbone = backbone
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)
        
        self.alignhead0 =  nn.Sequential( 
                nn.Conv2d(256, 2048, kernel_size=1, bias=False),
                # nn.BatchNorm2d(512),
            )
        self.alignhead1 =  nn.Sequential( 
                nn.Conv2d(512, 2048, kernel_size=1, bias=False),
                # nn.BatchNorm2d(512),
            )
        self.alignheadl =  nn.Sequential( 
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            # nn.BatchNorm2d(512),
        )
        self.avgp = nn.AvgPool2d((16,16))

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x

        "linguistic feature"
        
        lf = self.avgp(_x4)
        lf = self.alignheadl(lf)
        lf = lf.squeeze(2).squeeze(2)
        
        "------------------------------"
        
        if self.backbone == "mit_b0": 
            global_feature = self.alignhead0(_x4)
        else:
            global_feature = self.alignhead1(_x4)

        feature =  self.decoder(_x)
        pred = F.interpolate(feature, size=(self.size,self.size), mode='bilinear', align_corners=False)

        global_feature = F.interpolate(global_feature, size=(32, 32), mode='bilinear', align_corners=False)

        return pred, global_feature, lf