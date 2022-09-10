import torch
import torch.nn as nn
from modules.pointtransformer_utils import PointTransformerBlock, TransitionDown, TransitionUp


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        block = PointTransformerBlock
        num_block = [2, 3, 4, 6, 3]
        self.in_c = args.in_channel
        self.in_planes, planes = self.in_c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [16, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], num_block[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], num_block[1], share_planes, stride=stride[1],
                                   nsample=nsample[1], num_sector=4)  # N/4
        self.enc3 = self._make_enc(block, planes[2], num_block[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], num_block[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], num_block[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], args.num_class))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16, num_sector=1):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample, num_sector)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = [TransitionUp(self.in_planes, None if is_head else planes * block.expansion)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo, *args):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.in_c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x
