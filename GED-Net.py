import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from gcn_lib import Grapher, act_layer


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
class GEDNet(nn.Module):
    def __init__(self, n_channels, num_classes, bilinear=True):
        super(GEDNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        k = 12 
        act = 'gelu' 
        norm = 'batch' 
        bias = True 
        epsilon =  0.2 
        stochastic =  False 
        conv = 'mr' 
        emb_dims = 1024 
        drop_path = 0.4
        
        blocks = [2,2,2,2] 
        self.n_blocks = sum(blocks)
        channels = [80, 160, 400, 640] 
        u_channels=[80,160,400,640]
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  
        max_dilation = 49 // max(num_knn)
        
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        HW = 224 // 4 * 224 // 4

        item=0
        HW_new = 224 // 4 * 224 // 4
        self.backbone1,item = self.basic_backbone(channels[0],blocks[0],dpr,num_knn,4,max_dilation,item,HW_new)
        self.downsample1=Down(channels[0],channels[1])
        HW_new = HW_new // 4
        self.backbone2,item = self.basic_backbone(channels[1],blocks[1],dpr,num_knn,2,max_dilation,item,HW_new)
        self.downsample2=Down(channels[1],channels[2])
        HW_new = HW_new // 4
        self.backbone3,item = self.basic_backbone(channels[2],blocks[2],dpr,num_knn,1,max_dilation,item,HW_new)
        self.downsample3=Down(channels[2],channels[3])
        HW_new = HW_new // 4
        self.backbone4,item = self.basic_backbone(channels[3],blocks[3],dpr,num_knn,1,max_dilation,item,HW_new)
        self.undownsample4= nn.Sequential(nn.Conv2d(channels[3], 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024))
        
        self.all_backbone = nn.ModuleList([])
        self.all_backbone.append(self.backbone1)
        self.all_backbone.append(self.backbone2)
        self.all_backbone.append(self.backbone3)
        self.all_backbone.append(self.backbone4)
        self.all_backbone = Seq(*self.all_backbone)
        
        self.downsample=nn.ModuleList([])
        self.downsample.append(self.downsample1)
        self.downsample.append(self.downsample2)
        self.downsample.append(self.downsample3)
        self.downsample.append(self.undownsample4)
        self.downsample = Seq(*self.downsample)
        
        dpr_deconv = [x.item() for x in torch.linspace(0, drop_path, 4)]  
        item=0
        HW_new_1 = 224 // 4 * 224 // 4
        self.conv_backbone1,item = self.basic_backbone(channels[0],1, dpr_deconv,num_knn,4,max_dilation,item,HW_new_1)
        HW_new_1 = HW_new_1 // 4
        self.conv_backbone2,item = self.basic_backbone(channels[1],1, dpr_deconv,num_knn,2,max_dilation,item,HW_new_1)
        HW_new_1 = HW_new_1 // 4
        self.conv_backbone3,item = self.basic_backbone(channels[2],1, dpr_deconv,num_knn,1,max_dilation,item,HW_new_1)
        HW_new_1 = HW_new_1 // 4
        self.conv_backbone4,item = self.basic_backbone(channels[3],1, dpr_deconv,num_knn,1,max_dilation,item,HW_new_1)

        self.conv_backbone = nn.ModuleList([])
        self.conv_backbone.append(self.conv_backbone1)
        self.conv_backbone.append(self.conv_backbone2)
        self.conv_backbone.append(self.conv_backbone3)
        self.conv_backbone.append(self.conv_backbone4)
        self.conv_backbone = Seq(*self.conv_backbone)
        
        
        self.deconv_layer0 = nn.Sequential(nn.Conv2d(1024, channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]))     
        self.deconv_layer1 = self.basic_deconv(channels[3], channels[2])
        self.deconv_layer2 = self.basic_deconv(channels[2], channels[1])
        self.deconv_layer3 = self.basic_deconv(channels[1], channels[0])
        self.deconv_layer4 = self.basic_deconv(channels[0], channels[0])
        self.deconv_layer5 = self.basic_deconv(channels[0], 40)
        
        self.after_concat0=nn.Conv2d(u_channels[3]+channels[3]*2, channels[3], kernel_size=1)
        self.after_concat1=nn.Conv2d(u_channels[2]+channels[2]*2, channels[2], kernel_size=1)
        self.after_concat2=nn.Conv2d(u_channels[1]+channels[1]*2, channels[1], kernel_size=1)
        self.after_concat3=nn.Conv2d(u_channels[0]+channels[0]*2, channels[0], kernel_size=1)
        
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, u_channels[0])
        self.down3 = Down(u_channels[0], u_channels[1])
        self.down4 = Down(u_channels[1], u_channels[2])
        self.down5 = Down(u_channels[2],u_channels[3])
        
        self.segmentation = nn.Conv2d(40, num_classes, kernel_size=1)
        
        
    def basic_backbone(self, in_channels, num_blocks, dpr, num_knn, reduce_ratios, max_dilation, idx,HW):
        self.layers = nn.ModuleList([])
        act = 'gelu' 
        norm = 'batch' 
        bias = True 
        epsilon =  0.2 
        stochastic =  False 
        conv = 'mr' 
        epsilon =  0.2 
        for j in range(num_blocks):
            self.layers += [
                    Seq(Grapher(in_channels, num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios, n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN(in_channels, in_channels * 4, act=act, drop_path=dpr[idx])
                         )]
            idx += 1
        return nn.Sequential(*self.layers),idx
        
                            
    def basic_deconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
                    

    def forward(self, inputs):
        u=self.inc(inputs)
        u0=self.down1(u)
        u1=self.down2(u0)
        u2=self.down3(u1)
        u3=self.down4(u2)
        u4=self.down5(u3)
        
    
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        x0=self.all_backbone[0](x)
        x1 = self.downsample[0](x0)
        x2 = self.all_backbone[1](x1)
        x2 = self.downsample[1](x2)

        x3 = self.all_backbone[2](x2)
        x3 = self.downsample[2](x3)
        x4 = self.all_backbone[3](x3)
        x4 = self.downsample[3](x4)    
        x_d0=self.deconv_layer0(x4)
        
        x_d0 = torch.cat([u4,x_d0, x3], dim=1)
        x_d0= self.after_concat0(x_d0)                                
        x_d0 = self.conv_backbone[3](x_d0)
        x_d1 = self.deconv_layer1(x_d0)
        x_d1 = torch.cat([u3,x_d1, x2], dim=1)
        x_d1= self.after_concat1(x_d1)                                      
        x_d1 = self.conv_backbone[2](x_d1)
        x_d2 = self.deconv_layer2(x_d1)
        
        x_d2 = torch.cat([u2,x_d2, x1], dim=1)
        x_d2= self.after_concat2(x_d2)                                   
        x_d2 = self.conv_backbone[1](x_d2)                                  
        x_d3 = self.deconv_layer3(x_d2)
        x_d3 = torch.cat([u1,x_d3, x0], dim=1)
        x_d3= self.after_concat3(x_d3)   
        x_d3 = self.conv_backbone[0](x_d3)
        x_d4 = self.deconv_layer4(x_d3)
        x_d5 = self.deconv_layer5(x_d4)
        seg = self.segmentation(x_d5)
        return seg
        
