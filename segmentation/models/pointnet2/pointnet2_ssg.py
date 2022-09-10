"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import torch
import torch.nn as nn
from modules.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.sa1 = PointNetSetAbstraction(4, 32, 6 + 3, [32, 32, 64], num_sector=4)
        self.sa2 = PointNetSetAbstraction(4, 32, 64 + 3, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(4, 32, 128 + 3, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(4, 32, 256 + 3, [256, 256, 512])

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, args.num_class),
        )

    def forward(self, pos_feat_off0):
        pos_feat_off0[1] = torch.cat([pos_feat_off0[0], pos_feat_off0[1]], 1)

        pos_feat_off1 = self.sa1(pos_feat_off0)
        pos_feat_off2 = self.sa2(pos_feat_off1)
        pos_feat_off3 = self.sa3(pos_feat_off2)
        pos_feat_off4 = self.sa4(pos_feat_off3)

        pos_feat_off3[1] = self.fp4(pos_feat_off3, pos_feat_off4)
        pos_feat_off2[1] = self.fp3(pos_feat_off2, pos_feat_off3)
        pos_feat_off1[1] = self.fp2(pos_feat_off1, pos_feat_off2)
        pos_feat_off0[1] = self.fp1([pos_feat_off0[0], None, pos_feat_off0[2]], pos_feat_off1)

        feature = self.classifier(pos_feat_off0[1])

        return feature
