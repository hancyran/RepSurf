"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import torch
import numpy as np


def cal_normal(group_xyz, offset, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    :param group_xyz: [N, K=3, 3] / [N, G, K=3, 3]
    """
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [N, 3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [N, 3]

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
    unit_nor = nor / torch.norm(nor, dim=-1, keepdim=True)  # [B, N, 3] / [B, N, G, 3]
    if not is_group:
        pos_mask = (unit_nor[..., 0] > 0).float() * 2. - 1.  # keep x_n positive
    else:
        pos_mask = (unit_nor[..., 0:1, 0] > 0).float() * 2. - 1.
    unit_nor = unit_nor * pos_mask.unsqueeze(-1)

    # batch-wise random inverse normal vector (prob: 0.5)
    if random_inv:
        batch_prob = np.random.rand(offset.shape[0]) < 0.5
        random_mask = []
        sample_offset = [0] + list(offset.cpu().numpy())
        for idx in range(len(sample_offset) - 1):
            sample_mask = torch.ones((sample_offset[idx+1] - sample_offset[idx], 1), dtype=torch.float32)
            if not batch_prob[idx]:
                sample_mask *= -1
            random_mask.append(sample_mask)
        random_mask = torch.cat(random_mask, dim=0).to(unit_nor.device)
        # random_mask = torch.randint(0, 2, (group_xyz.size(0), 1)).float() * 2. - 1.
        # random_mask = random_mask.to(unit_nor.device)
        if not is_group:
            unit_nor = unit_nor * random_mask
        else:
            unit_nor = unit_nor * random_mask.unsqueeze(-1)

    return unit_nor


def cal_center(group_xyz):
    """
    Calculate Global Coordinates of the Center of Triangle

    :param group_xyz: [N, K, 3] / [N, G, K, 3]; K >= 3
    :return: [N, 3] / [N, G, 3]
    """
    center = torch.mean(group_xyz, dim=-2)
    return center


def cal_area(group_xyz):
    """
    Calculate Area of Triangle

    :param group_xyz: [N, K, 3] / [N, G, K, 3]; K = 3
    :return: [N, 1] / [N, G, 1]
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
    :param normal: [N, 3] / [N, G, 3]
    :param center: [N, 3] / [N, G, 3]
    :return: [N, 1] / [N, G, 1]
    """
    const = torch.sum(normal * center, dim=-1, keepdim=True)
    factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
    const = const / factor if is_normalize else const

    return const


def check_nan(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    :param pos: [N, 1]
    :param center: [N, 3]
    :param normal: [N, 3]
    """
    N, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    mask_first = torch.argmax((~mask).int(), dim=-1)

    normal_first = normal[None, mask_first].repeat([N, 1])
    normal[mask] = normal_first[mask]
    center_first = center[None, mask_first].repeat([N, 1])
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[None, mask_first].repeat([N, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center


def check_nan_umb(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    :param pos: [N, G, 1]
    :param center: [N, G, 3]
    :param normal: [N, G, 3]
    """
    N, G, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    mask_first = torch.argmax((~mask).int(), dim=-1)

    normal_first = normal[torch.arange(N), None, mask_first].repeat([1, G, 1])
    normal[mask] = normal_first[mask]
    center_first = center[torch.arange(N), None, mask_first].repeat([1, G, 1])
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[torch.arange(N), None, mask_first].repeat([1, G, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center
