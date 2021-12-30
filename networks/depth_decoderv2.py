from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *

class DepthDecoderV2(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=2, use_skips=True, lite_model = None):
        super(DepthDecoderV2, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        # decoder
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

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()#why not relu?
    
    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):#[4,3,2,1,0]
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]#this function in layers.py
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                final = self.sigmoid(self.convs[("dispconv", i)](x))
                self.outputs[("disp", i)] = final[:,0,:,:].unsqueeze(1)
                self.outputs[("uncert", i)] = final[:,1,:,:].unsqueeze(1)
        return self.outputs
