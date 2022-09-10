"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.polar_utils import xyz2sphere
from modules.recons_utils import cal_const, cal_normal, cal_center, check_nan_umb
from lib.pointops.functions import pointops


def sample_and_group(stride, nsample, center, normal, feature, offset, return_polar=False, num_sector=1, training=True):
    # sample
    if stride > 1:
        new_offset, sample_idx = [offset[0].item() // stride], offset[0].item() // stride
        for i in range(1, offset.shape[0]):
            sample_idx += (offset[i].item() - offset[i - 1].item()) // stride
            new_offset.append(sample_idx)
        new_offset = torch.cuda.IntTensor(new_offset)
        if num_sector > 1 and training:
            fps_idx = pointops.sectorized_fps(center, offset, new_offset, num_sector)  # [M]
        else:
            fps_idx = pointops.furthestsampling(center, offset, new_offset)  # [M]
        new_center = center[fps_idx.long(), :]  # [M, 3]
        new_normal = normal[fps_idx.long(), :]  # [M, 3]
    else:
        new_center = center
        new_normal = normal
        new_offset = offset

    # group
    N, M, D = center.shape[0], new_center.shape[0], normal.shape[1]
    group_idx, _ = pointops.knnquery(nsample, center, new_center, offset, new_offset)  # [M, nsample]
    group_center = center[group_idx.view(-1).long(), :].view(M, nsample, 3)  # [M, nsample, 3]
    group_normal = normal[group_idx.view(-1).long(), :].view(M, nsample, D)  # [M, nsample, 10]
    group_center_norm = group_center - new_center.unsqueeze(1)
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)

    if feature is not None:
        C = feature.shape[1]
        group_feature = feature[group_idx.view(-1).long(), :].view(M, nsample, C)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1)   # [npoint, nsample, C+D]
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1)

    return new_center, new_normal, new_feature, new_offset


def resort_points(points, idx):
    """
    Resort Set of points along G dim

    :param points: [N, G, 3]
    :param idx: [N, G]
    :return: [N, G, 3]
    """
    device = points.device
    N, G, _ = points.shape

    n_indices = torch.arange(N, dtype=torch.long).to(device).view([N, 1]).repeat([1, G])
    new_points = points[n_indices, idx, :]

    return new_points


def _fixed_rotate(xyz):
    # y-axis:45deg -> z-axis:45deg
    rot_mat = torch.FloatTensor([[0.5, -0.5, 0.7071], [0.7071, 0.7071, 0.], [-0.5, 0.5, 0.7071]]).to(xyz.device)
    return xyz @ rot_mat


def group_by_umbrella_v2(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(_fixed_rotate(group_xyz_norm))[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


def group_by_umbrella(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """
    group_idx, _ = pointops.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]
    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]
    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


def sort_factory(s_type):
    if s_type is None:
        return group_by_umbrella
    elif s_type == 'fix':
        return group_by_umbrella_v2
    else:
        raise Exception('No such sorting method')


class SurfaceAbstraction(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, stride, nsample, in_channel, mlp, return_polar=True, num_sector=1):
        super(SurfaceAbstraction, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.num_sector = num_sector
        self.return_polar = return_polar
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_nor_feat_off):
        center, normal, feature, offset = pos_nor_feat_off  # [N, 3], [N, 10], [N, C], [B]

        new_center, new_normal, new_feature, new_offset = sample_and_group(self.stride, self.nsample, center,
                                                                           normal, feature, offset,
                                                                           return_polar=self.return_polar,
                                                                           num_sector=self.num_sector,
                                                                           training=self.training)

        # new_center: sampled points position data, [M, 3]
        # new_normal: sampled normal feature data, [M, 3/10]
        # new_feature: sampled feature, [M, nsample, 3+3/10+C]
        new_feature = new_feature.transpose(1, 2).contiguous()  # [M, 3+C, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        return [new_center, new_normal, new_feature, new_offset]


class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module w/ Channel De-differentiation

    """

    def __init__(self, stride, nsample, feat_channel, pos_channel, mlp, return_normal=True, return_polar=False,
                 num_sector=1):
        super(SurfaceAbstractionCD, self).__init__()
        self.stride = stride
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.num_sector = num_sector
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel

        self.mlp_l0 = nn.Conv1d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv1d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm1d(mlp[0])
        self.bn_f0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_nor_feat_off):
        center, normal, feature, offset = pos_nor_feat_off  # [N, 3], [N, 10], [N, C], [B]

        new_center, new_normal, new_feature, new_offset = sample_and_group(self.stride, self.nsample, center,
                                                                           normal, feature, offset,
                                                                           return_polar=self.return_polar,
                                                                           num_sector=self.num_sector,
                                                                           training=self.training)

        # new_center: sampled points position data, [M, 3]
        # new_normal: sampled normal feature data, [M, 3/10]
        # new_feature: sampled feature, [M, nsample, 3+3/10+C]
        new_feature = new_feature.transpose(1, 2).contiguous()  # [M, 3+C, nsample]

        # init layer
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
        new_feature = loc + feat
        new_feature = F.relu(new_feature)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        return [new_center, new_normal, new_feature, new_offset]


class SurfaceFeaturePropagationCD(nn.Module):
    """
    Surface FP Module w/ Channel De-differentiation

    """

    def __init__(self, prev_channel, skip_channel, mlp):
        super(SurfaceFeaturePropagationCD, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.skip = skip_channel is not None

        self.mlp_f0 = nn.Linear(prev_channel, mlp[0])
        self.norm_f0 = nn.BatchNorm1d(mlp[0])
        if skip_channel is not None:
            self.mlp_s0 = nn.Linear(skip_channel, mlp[0])
            self.norm_s0 = nn.BatchNorm1d(mlp[0])

        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos_feat_off1, pos_feat_off2):
        xyz1, points1, offset1 = pos_feat_off1  # [N, 3], [N, C], [B]
        xyz2, points2, offset2 = pos_feat_off2  # [M, 3], [M, C], [B]

        # interpolation
        idx, dist = pointops.knnquery(3, xyz2, xyz1, offset2, offset1)  # [M, 3], [M, 3]
        dist_recip = 1.0 / (dist + 1e-8)  # [M, 3]
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm  # [M, 3]

        points2 = self.norm_f0(self.mlp_f0(points2))
        interpolated_points = torch.cuda.FloatTensor(xyz1.shape[0], points2.shape[1]).zero_()
        for i in range(3):
            interpolated_points += points2[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)

        # init layer
        if self.skip:
            skip = self.norm_s0(self.mlp_s0(points1))
            new_points = F.relu(interpolated_points + skip)
        else:
            new_points = F.relu(interpolated_points)

        # mlp
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella Surface Representation Constructor

    """

    def __init__(self, k, in_channel, out_channel, random_inv=True, sort='fix'):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.random_inv = random_inv

        self.mlps = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
            nn.Conv1d(out_channel, out_channel, 1, bias=True),
        )
        self.sort_func = sort_factory(sort)

    def forward(self, center, offset):
        # umbrella surface reconstruction
        group_xyz = self.sort_func(center, center, offset, offset, k=self.k)  # [N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center)
        # surface position
        group_pos = cal_const(group_normal, group_center)

        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
        new_feature = torch.cat([group_polar, group_normal, group_pos, group_center], dim=-1)  # P+N+SP+C: 10
        new_feature = new_feature.transpose(1, 2).contiguous()  # [N, C, G]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        new_feature = torch.sum(new_feature, 2)

        return new_feature
