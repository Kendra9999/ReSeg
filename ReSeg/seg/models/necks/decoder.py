# Copyright (c) nnUNet v2

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_activation_layer
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS


@NECKS.register_module()
class SegDecoder(BaseModule):
    def __init__(self,
                 in_channels,
                 aux_channels,
                 num_classes: int,
                 deep_supervision: bool = True,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=dict(
                     type='Xavier', layer='Conv3d', distribution='uniform')):
        super(SegDecoder, self).__init__(init_cfg)

        self.deep_supervision = deep_supervision

        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, len(in_channels)):
            transpconvs.append(nn.ConvTranspose3d(
                in_channels[-s], in_channels[-(s+1)], kernel_size=2, stride=2))
            stages.append(FeatureFusionModule(
                base_channels=in_channels[-(s+1)],
                aux_channels=aux_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ))
            seg_layers.append(nn.Conv3d(in_channels[-(s+1)], num_classes+1, kernel_size=1,
                                        stride=1, padding=0, bias=True))
        
        self.transpconvs = nn.ModuleList(transpconvs)
        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)  # num_classes+1

    @auto_fp16()
    def forward(self, inputs, aux_inputs):
        """Forward function."""
        lres_input = inputs[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, inputs[-(s+2)]), 1)
            x = self.stages[s](x, aux_inputs)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision or not self.training:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r
    
    @auto_fp16()
    def get_attention_map(self, inputs, aux_inputs):
        """Forward function."""
        lres_input = inputs[-1]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, inputs[-(s+2)]), 1)
            if s == (len(self.stages) - 1):
                attn_map = self.stages[s].get_attention_map(x, aux_inputs)
            else:
                x = self.stages[s](x, aux_inputs)
            lres_input = x
        
        return attn_map
    


class FeatureFusionModule(BaseModule):
    def __init__(self,
                 base_channels,
                 aux_channels,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=dict(
                     type='Xavier', layer='Conv3d', distribution='uniform')):
        super(FeatureFusionModule, self).__init__(init_cfg)
        self.x_conv = ConvModule(
                    base_channels*2, base_channels, kernel_size=3, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.aux_conv = AlignedModule(aux_channels=aux_channels, base_channels=base_channels,
                                      conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.attn_conv = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1,
                      bias=False, groups=base_channels),
            nn.BatchNorm3d(base_channels),
            nn.Conv3d(base_channels, base_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.y_conv = nn.Sequential(
            MultiScaleDWConv(base_channels, scale=(1, 3, 5, 7)),
            nn.BatchNorm3d(base_channels),
            ConvModule(base_channels, base_channels, kernel_size=1, padding=0,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

    @auto_fp16()
    def forward(self, x, x_aux):
        x = self.x_conv(x)

        x_aux = self.aux_conv(x_aux, x)        

        attn_aux = self.sigmoid(self.attn_conv(x_aux))

        y = x * attn_aux + x

        out = self.y_conv(y) + y

        return out
    
    @auto_fp16()
    def get_attention_map(self, x, x_aux):
        x = self.x_conv(x)

        x_aux = self.aux_conv(x_aux, x)        

        attn_aux = self.sigmoid(self.attn_conv(x_aux))

        return attn_aux
        

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv3d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i]//2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    @auto_fp16()      
    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class AlignedModule(nn.Module):
    def __init__(self, aux_channels: int, base_channels: int,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),):
        super(AlignedModule, self).__init__()

        self.low_down1 = ConvModule(
                    aux_channels, base_channels, kernel_size=3, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.low_down2 = nn.Conv3d(base_channels, base_channels//2, 1, 1, bias=False)
        
        self.high_down = nn.Conv3d(base_channels, base_channels//2, 1, 1, bias=False)
        self.flow_make = nn.Conv3d(base_channels, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, lres_fea, hres_fea):
        lres_fea = self.low_down1(lres_fea)
        l_fea = self.low_down2(lres_fea)
        l_fea = F.interpolate(l_fea, size=hres_fea.size()[2:], mode="trilinear", align_corners=True)
        h_fea = self.high_down(hres_fea)

        flow = self.flow_make(torch.cat([l_fea, h_fea], 1))

        l_feature = self.flow_warp(lres_fea, flow, size=hres_fea.size()[2:])

        return l_feature
    
    def flow_warp(self, input, flow, size):
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type_as(input).to(input.device)
        
        # new locations
        new_locs = grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(input, new_locs, align_corners=True, mode='bilinear')