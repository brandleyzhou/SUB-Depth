# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
#from hr_layers import * 
#from visual_block import visual_block

class UncertDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        #  num_ch_enc = np.array([64, 64, 128, 256, 512])
        super(UncertDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        for i in range(4, -1, -1): #i=[4,3,2,1,0]
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)#CONV2D

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        #for s in self.scales:
        self.convs[("uncert_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()#why not relu?
    
    def forward(self, input_features):
        #block_list = [self.se_block0, self.se_block1, self.se_block2, self.se_block3]
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):#[4,3,2,1,0]
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]#this function in layers.py
            #x = upsample(x)#this function in layers.py
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i == 0:
                uncertainty_map = self.sigmoid(self.convs[("uncert_conv", i)](x))
        return uncertainty_map
