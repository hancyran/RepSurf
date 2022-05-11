"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch
from torch import nn
from modules.pointnet2_utils import query_knn_point, index_points


def _recons_factory(type, cuda=False):
    if type == 'knn':
        return knn_recons
    # elif type == 'lknn':
    #     return limited_knn_recons
    else:
        raise Exception('Not Implemented Reconstruction Type')


def knn_recons(k, center, context, cuda=False):
    idx = query_knn_point(k, context, center, cuda=cuda)
    torch.cuda.empty_cache()

    group_xyz = index_points(context, idx, cuda=cuda, is_group=True)  # [B, N, K, C]
    torch.cuda.empty_cache()
    return group_xyz


def cal_normal(group_xyz, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    :param group_xyz: [B, N, K=3, 3] / [B, N, G, K=3, 3]
    :param random_inv:
    :param return_intersect:
    :param return_const:
    :return: [B, N, 3]
    """
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [B, N, 3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [B, N, 3]

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
    unit_nor = nor / torch.norm(nor, dim=-1, keepdim=True)  # [B, N, 3] / [B, N, G, 3]
    if not is_group:
        pos_mask = (unit_nor[..., 0] > 0).float() * 2. - 1.  # keep x_n positive
    else:
        pos_mask = (unit_nor[..., 0:1, 0] > 0).float() * 2. - 1.
    unit_nor = unit_nor * pos_mask.unsqueeze(-1)

    # batch-wise random inverse normal vector (prob: 0.5)
    if random_inv:
        random_mask = torch.randint(0, 2, (group_xyz.size(0), 1, 1)).float() * 2. - 1.
        random_mask = random_mask.to(unit_nor.device)
        if not is_group:
            unit_nor = unit_nor * random_mask
        else:
            unit_nor = unit_nor * random_mask.unsqueeze(-1)

    return unit_nor


def pca(X, k, center=True):
    """
    Principal Components Analysis impl. with SVD function

    :param X:
    :param k:
    :param center:
    :return:
    """

    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    explained_variance = torch.mul(s[:k], s[:k]) / (n - 1)
    return {'X': X, 'k': k, 'components': components,
            'explained_variance': explained_variance}


def cal_center(group_xyz):
    """
    Calculate Global Coordinates of the Center of Triangle

    :param group_xyz: [B, N, K, 3] / [B, N, G, K, 3]; K >= 3
    :return: [B, N, 3] / [B, N, G, 3]
    """
    center = torch.mean(group_xyz, dim=-2)
    return center


def cal_area(group_xyz):
    """
    Calculate Area of Triangle

    :param group_xyz: [B, N, K, 3] / [B, N, G, K, 3]; K = 3
    :return: [B, N, 1] / [B, N, G, 1]
    """
    pad_shape = group_xyz[..., 0, None].shape
    det_xy = torch.det(torch.cat([group_xyz[..., 0, None], group_xyz[..., 1, None], torch.ones(pad_shape)], dim=-1))
    det_yz = torch.det(torch.cat([group_xyz[..., 1, None], group_xyz[..., 2, None], torch.ones(pad_shape)], dim=-1))
    det_zx = torch.det(torch.cat([group_xyz[..., 2, None], group_xyz[..., 0, None], torch.ones(pad_shape)], dim=-1))
    area = torch.sqrt(det_xy ** 2 + det_yz ** 2 + det_zx ** 2).unsqueeze(-1)
    return area


def cal_const(normal, center, is_normalize=True):
    """
    Calculate Constant Term (Standard Version, with x_normal to be 1)

    math::
        const = x_nor * x_0 + y_nor * y_0 + z_nor * z_0

    :param is_normalize:
    :param normal: [B, N, 3] / [B, N, G, 3]
    :param center: [B, N, 3] / [B, N, G, 3]
    :return: [B, N, 1] / [B, N, G, 1]
    """
    const = torch.sum(normal * center, dim=-1, keepdim=True)
    factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
    const = const / factor if is_normalize else const

    return const


def check_nan(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    :param pos: [B, N, 1]
    :param center: [B, N, 3]
    :param normal: [B, N, 3]
    :return:
    """
    B, N, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    mask_first = torch.argmax((~mask).int(), dim=-1)

    normal_first = normal[torch.arange(B), None, mask_first].repeat([1, N, 1])
    normal[mask] = normal_first[mask]
    center_first = center[torch.arange(B), None, mask_first].repeat([1, N, 1])
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[torch.arange(B), None, mask_first].repeat([1, N, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center


def check_nan_umb(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    :param pos: [B, N, G, 1]
    :param center: [B, N, G, 3]
    :param normal: [B, N, G, 3]
    :return:
    """
    B, N, G, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    mask_first = torch.argmax((~mask).int(), dim=-1)
    b_idx = torch.arange(B).unsqueeze(1).repeat([1, N])
    n_idx = torch.arange(N).unsqueeze(0).repeat([B, 1])

    normal_first = normal[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
    normal[mask] = normal_first[mask]
    center_first = center[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center


class SurfaceConstructor(nn.Module):
    """
    Surface Constructor for Point Clouds

    Formulation of A Surface:
        A * (x - x_0) + B * (y - y_0) + C * (z - z_0) = 0,
        where A^2 + B^2 + C^2 = 1 & A > 0
    """

    def __init__(self, r=None, k=3, recons_type='knn', return_dist=False, random_inv=True, cuda=False):
        super(SurfaceConstructor, self).__init__()
        self.K = k
        self.R = r
        self.recons = _recons_factory(recons_type)
        self.cuda = cuda

        self.return_dist = return_dist
        self.random_inv = random_inv

    def forward(self, center, context):
        """
        Input:
            center: input points position as centroid points, [B, 3, N]
            context: input points position as context points, [B, 3, N']

        Output:
            normal: normals of constructed triangles, [B, 3, N]
            center: centroids of constructed triangles, [B, 3, N]
            pos: position info of constructed triangles, [B, 1, N]
        """
        center = center.permute(0, 2, 1)
        context = context.permute(0, 2, 1)

        group_xyz = self.recons(self.K, center, context, cuda=self.cuda)
        normal = cal_normal(group_xyz, random_inv=self.random_inv)
        center = cal_center(group_xyz)

        if self.return_dist:
            pos = cal_const(normal, center)
            normal, center, pos = check_nan(normal, center, pos)
            normal = normal.permute(0, 2, 1)
            center = center.permute(0, 2, 1)
            pos = pos.permute(0, 2, 1)
            return normal, center, pos

        normal, center = check_nan(normal, center)
        normal = normal.permute(0, 2, 1)
        center = center.permute(0, 2, 1)

        return normal, center


if __name__ == '__main__':
    xyz = torch.rand(1, 3, 1024) * 2. - 1.
    constructor = SurfaceConstructor(return_dist=True)

    normal, center, pos = constructor(xyz, xyz)
    print(normal.shape)
    print(center.shape)
