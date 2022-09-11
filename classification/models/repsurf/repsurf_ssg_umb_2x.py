"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch.nn as nn
import torch.nn.functional as F
from modules.repsurface_utils import SurfaceAbstractionCD, UmbrellaSurfaceConstructor


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        center_channel = 0 if not args.return_center else (6 if args.return_polar else 3)
        repsurf_channel = 10

        self.init_nsample = args.num_point
        self.return_dist = args.return_dist
        self.surface_constructor = UmbrellaSurfaceConstructor(args.group_size + 1, repsurf_channel,
                                                              return_dist=args.return_dist, aggr_type=args.umb_pool,
                                                              cuda=args.cuda_ops)
        self.sa1 = SurfaceAbstractionCD(npoint=512, radius=0.1, nsample=24, feat_channel=repsurf_channel,
                                        pos_channel=center_channel, mlp=[128, 128, 256], group_all=False,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        self.sa2 = SurfaceAbstractionCD(npoint=128, radius=0.2, nsample=24, feat_channel=256 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[256, 256, 512], group_all=False,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        self.sa3 = SurfaceAbstractionCD(npoint=32, radius=0.4, nsample=24, feat_channel=512 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[512, 512, 1024], group_all=False,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        self.sa4 = SurfaceAbstractionCD(npoint=None, radius=None, nsample=None, feat_channel=1024 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[1024, 1024, 2048], group_all=True,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        # modelnet40
        self.classfier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(256, args.num_class))

    def forward(self, points):
        # init
        center = points[:, :3, :]

        normal = self.surface_constructor(center)

        center, normal, feature = self.sa1(center, normal, None)
        center, normal, feature = self.sa2(center, normal, feature)
        center, normal, feature = self.sa3(center, normal, feature)
        center, normal, feature = self.sa4(center, normal, feature)

        feature = feature.view(-1, 2048)
        feature = self.classfier(feature)
        feature = F.log_softmax(feature, -1)

        return feature
