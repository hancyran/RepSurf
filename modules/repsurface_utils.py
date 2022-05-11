"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.pointnet2_utils import farthest_point_sample, index_points, query_knn_point, query_ball_point
from modules.polar_utils import xyz2sphere
from modules.recons_utils import cal_const, cal_normal, cal_center, check_nan_umb


def sample_and_group(npoint, radius, nsample, center, normal, feature, return_normal=True, return_polar=False, cuda=False):
    """
    Input:
        center: input points position data
        normal: input points normal data
        feature: input points feature
    Return:
        new_center: sampled points position data
        new_normal: sampled points normal data
        new_feature: sampled points feature
    """
    # sample
    fps_idx = farthest_point_sample(center, npoint, cuda=cuda)  # [B, npoint, A]
    torch.cuda.empty_cache()
    # sample center
    new_center = index_points(center, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()
    # sample normal
    new_normal = index_points(normal, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()

    # group
    idx = query_ball_point(radius, nsample, center, new_center, cuda=cuda)
    torch.cuda.empty_cache()
    # group normal
    group_normal = index_points(normal, idx, cuda=cuda, is_group=True)  # [B, npoint, nsample, B]
    torch.cuda.empty_cache()
    # group center
    group_center = index_points(center, idx, cuda=cuda, is_group=True)  # [B, npoint, nsample, A]
    torch.cuda.empty_cache()
    group_center_norm = group_center - new_center.unsqueeze(2)
    torch.cuda.empty_cache()

    # group polar
    if return_polar:
        group_polar = xyz2sphere(group_center_norm)
        group_center_norm = torch.cat([group_center_norm, group_polar], dim=-1)
    if feature is not None:
        group_feature = index_points(feature, idx, cuda=cuda, is_group=True)
        new_feature = torch.cat([group_center_norm, group_normal, group_feature], dim=-1) if return_normal \
            else torch.cat([group_center_norm, group_feature], dim=-1)
    else:
        new_feature = torch.cat([group_center_norm, group_normal], dim=-1)

    return new_center, new_normal, new_feature


def sample_and_group_all(center, normal, feature, return_normal=True, return_polar=False):
    """
    Input:
        center: input centroid position data
        normal: input normal data
        feature: input feature data
    Return:
        new_center: sampled points position data
        new_normal: sampled points position data
        new_feature: sampled points data
    """
    device = center.device
    B, N, C = normal.shape

    new_center = torch.zeros(B, 1, 3).to(device)
    new_normal = new_center

    group_normal = normal.view(B, 1, N, C)
    group_center = center.view(B, 1, N, 3)
    if return_polar:
        group_polar = xyz2sphere(group_center)
        group_center = torch.cat([group_center, group_polar], dim=-1)

    new_feature = torch.cat([group_center, group_normal, feature.view(B, 1, N, -1)], dim=-1) if return_normal \
        else torch.cat([group_center, feature.view(B, 1, N, -1)], dim=-1)

    return new_center, new_normal, new_feature


def resort_points(points, idx):
    """
    Resort Set of points along G dim

    """
    device = points.device
    B, N, G, _ = points.shape

    view_shape = [B, 1, 1]
    repeat_shape = [1, N, G]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points


def group_by_umbrella(xyz, new_xyz, k=9, cuda=False):
    """
    Group a set of points into umbrella surfaces

    """
    idx = query_knn_point(k, xyz, new_xyz, cuda=cuda)
    torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
    torch.cuda.empty_cache()

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N', K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1]

    # [B, N', K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz


class SurfaceAbstraction(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, return_polar=True, return_normal=True, cuda=False):
        super(SurfaceAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all:
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_polar=self.return_polar,
                                                                       return_normal=self.return_normal)
        else:
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_polar=self.return_polar,
                                                                   return_normal=self.return_normal, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature


class SurfaceAbstractionCD(nn.Module):
    """
    Surface Abstraction Module

    """

    def __init__(self, npoint, radius, nsample, feat_channel, pos_channel, mlp, group_all,
                 return_normal=True, return_polar=False, cuda=False):
        super(SurfaceAbstractionCD, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel
        self.group_all = group_all

        self.mlp_l0 = nn.Conv2d(self.pos_channel, mlp[0], 1)
        self.mlp_f0 = nn.Conv2d(feat_channel, mlp[0], 1)
        self.bn_l0 = nn.BatchNorm2d(mlp[0])
        self.bn_f0 = nn.BatchNorm2d(mlp[0])

        # mlp_l0+mlp_f0 can be considered as the first layer of mlp_convs
        last_channel = mlp[0]
        for out_channel in mlp[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, center, normal, feature):
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all:
            new_center, new_normal, new_feature = sample_and_group_all(center, normal, feature,
                                                                       return_normal=self.return_normal,
                                                                       return_polar=self.return_polar)
        else:
            new_center, new_normal, new_feature = sample_and_group(self.npoint, self.radius, self.nsample, center,
                                                                   normal, feature, return_normal=self.return_normal,
                                                                   return_polar=self.return_polar, cuda=self.cuda)

        new_feature = new_feature.permute(0, 3, 2, 1)

        # init layer
        loc = self.bn_l0(self.mlp_l0(new_feature[:, :self.pos_channel]))
        feat = self.bn_f0(self.mlp_f0(new_feature[:, self.pos_channel:]))
        new_feature = loc + feat
        new_feature = F.relu(new_feature)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature = F.relu(bn(conv(new_feature)))
        new_feature = torch.max(new_feature, 2)[0]

        new_center = new_center.permute(0, 2, 1)
        new_normal = new_normal.permute(0, 2, 1)

        return new_center, new_normal, new_feature


class UmbrellaSurfaceConstructor(nn.Module):
    """
    Umbrella-based Surface Abstraction Module

    """

    def __init__(self, k, in_channel, aggr_type='sum', return_dist=False, random_inv=True, cuda=False):
        super(UmbrellaSurfaceConstructor, self).__init__()
        self.k = k
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type = aggr_type
        self.cuda = cuda

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        )

    def forward(self, center):
        center = center.permute(0, 2, 1)
        # surface construction
        group_xyz = group_by_umbrella(center, center, k=self.k, cuda=self.cuda)  # [B, N, K-1, 3 (points), 3 (coord.)]

        # normal
        group_normal = cal_normal(group_xyz, random_inv=self.random_inv, is_group=True)
        # coordinate
        group_center = cal_center(group_xyz)
        # polar
        group_polar = xyz2sphere(group_center)
        if self.return_dist:
            group_pos = cal_const(group_normal, group_center)
            group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)
            # new_feature = torch.cat([group_normal], dim=-1)  # N: 3
            # new_feature = torch.cat([group_normal, group_pos], dim=-1)  # N+P: 4
            # new_feature = torch.cat([group_center, group_normal], dim=-1)  # N+C: 6
            # new_feature = torch.cat([group_center, group_normal, group_pos], dim=-1)  # N+P+C: 7
            new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        else:
            group_normal, group_center = check_nan_umb(group_normal, group_center)
            new_feature = torch.cat([group_center, group_polar, group_normal], dim=-1)
        new_feature = new_feature.permute(0, 3, 2, 1)  # [B, C, G, N]

        # mapping
        new_feature = self.mlps(new_feature)

        # aggregation
        if self.aggr_type == 'max':
            new_feature = torch.max(new_feature, 2)[0]
        elif self.aggr_type == 'avg':
            new_feature = torch.mean(new_feature, 2)
        else:
            new_feature = torch.sum(new_feature, 2)

        return new_feature
