# Add your custom network here
from .default import DRNet,LRNet,SubNet
import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)

def errnet_lite(in_channels, out_channels, **kwargs):
    return LRNet(in_channels, out_channels, 256, 5, 8, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=False, **kwargs), \
        SubNet(in_channels, out_channels, 256, 8, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=False, **kwargs)
