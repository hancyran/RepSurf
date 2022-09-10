"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import torch
import torch.nn as nn
from modules.repsurface_utils import UmbrellaSurfaceConstructor, SurfaceAbstractionCD, SurfaceFeaturePropagationCD


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        center_channel = 6 if args.return_polar else 3
        repsurf_in_channel = 10
        repsurf_out_channel = 10

        self.sa1 = SurfaceAbstractionCD(4, 32, args.in_channel + repsurf_out_channel, center_channel, [32, 32, 64],
                                        True, args.return_polar, num_sector=4)
        self.sa2 = SurfaceAbstractionCD(4, 32, 64 + repsurf_out_channel, center_channel, [64, 64, 128],
                                        True, args.return_polar)
        self.sa3 = SurfaceAbstractionCD(4, 32, 128 + repsurf_out_channel, center_channel, [128, 128, 256],
                                        True, args.return_polar)
        self.sa4 = SurfaceAbstractionCD(4, 32, 256 + repsurf_out_channel, center_channel, [256, 256, 512],
                                        True, args.return_polar)

        self.fp4 = SurfaceFeaturePropagationCD(512, 256, [256, 256])
        self.fp3 = SurfaceFeaturePropagationCD(256, 128, [256, 256])
        self.fp2 = SurfaceFeaturePropagationCD(256, 64, [256, 128])
        self.fp1 = SurfaceFeaturePropagationCD(128, None, [128, 128, 128])

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, args.num_class),
        )

        self.surface_constructor = UmbrellaSurfaceConstructor(args.group_size + 1, repsurf_in_channel, repsurf_out_channel)

    def forward(self, pos_feat_off0):
        pos_nor_feat_off0 = [
            pos_feat_off0[0],
            self.surface_constructor(pos_feat_off0[0], pos_feat_off0[2]),
            torch.cat([pos_feat_off0[0], pos_feat_off0[1]], 1),
            pos_feat_off0[2]
        ]

        pos_nor_feat_off1 = self.sa1(pos_nor_feat_off0)
        pos_nor_feat_off2 = self.sa2(pos_nor_feat_off1)
        pos_nor_feat_off3 = self.sa3(pos_nor_feat_off2)
        pos_nor_feat_off4 = self.sa4(pos_nor_feat_off3)

        del pos_nor_feat_off0[1], pos_nor_feat_off1[1], pos_nor_feat_off2[1], pos_nor_feat_off3[1], pos_nor_feat_off4[1]
        pos_nor_feat_off3[1] = self.fp4(pos_nor_feat_off3, pos_nor_feat_off4)
        pos_nor_feat_off2[1] = self.fp3(pos_nor_feat_off2, pos_nor_feat_off3)
        pos_nor_feat_off1[1] = self.fp2(pos_nor_feat_off1, pos_nor_feat_off2)
        pos_nor_feat_off0[1] = self.fp1([pos_nor_feat_off0[0], None, pos_nor_feat_off0[2]], pos_nor_feat_off1)

        feature = self.classifier(pos_nor_feat_off0[1])

        return feature
