# camera-ready

from json import encoder
from statistics import mode
from click import style
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3plus.asppv3p import ASPP
from models.attention.models import AttentionPool2d


"[implement v3+ for kd project]"
class att(nn.Module):
    def __init__(self, num_classes: int, encoder_type: str, aspp_dilate=[6, 12, 18], model_path=None,pretrained_imgnet=True,feature=False):
        super(att, self).__init__()
        self.feature = feature
        self.num_classes = num_classes
        self.encoder_type = encoder_type
        self.project = nn.Sequential( 
            nn.Conv2d(512, 48, 1, bias=False),#256
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.in_channels = 2048#2560
        self.aspp = ASPP(self.in_channels, aspp_dilate)
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), # 304 = 256 + 48
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # ======
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(),
            # ======
            nn.Conv2d(256, num_classes, 1)
        )

        if self.encoder_type == 'att':
            self.encoder = AttentionPool2d(16,2048,32,512)
        else:
            raise NotImplementedError('no this model here for now!')

        # if model_path is not None:
        #     model_state_dict = torch.load(model_path, map_location='cpu')
        #     self.encoder.load_state_dict(model_state_dict, strict=False)

    def forward(self, x):
        # high = x[0]
        # low = x[1]
        # low_feature_map = low
        # low_feature_map = self.project(low_feature_map)

        g,f = self.encoder(x)
        # high_feature_map = torch.cat((high,f),dim=1)

        # output_feature = self.aspp(high_feature_map) # (shape: (batch_size, 256, h/16?, w/16?)) output_stride=16
        # output_feature = F.interpolate(output_feature, size=low_feature_map.shape[2:], mode="bilinear", align_corners=False)
        # feature_map_cat = torch.cat( [ low_feature_map, output_feature ], dim=1) # 

        # feature_map_cat = self.classifier(feature_map_cat)

        # pred = F.interpolate(feature_map_cat, size=(512,512), mode='bilinear', align_corners=False) #TODO: Corner?

        return g
        

