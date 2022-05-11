"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch
try:
    from pointops.functions.pointops import furthestsampling, gathering, ballquery, knnquery_naive, knnquery, \
        grouping, interpolation, nearestneighbor
except:
    pass


def pc_normalize(pc, norm='instance'):
    """
    Batch Norm to Instance Norm
    Normalize Point Clouds | Pytorch Version | Range: [-1, 1]

    """
    points = pc[:, :3, :]
    centroid = torch.mean(points, dim=2, keepdim=True)
    points = points - centroid
    if norm == 'instance':
        m = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)), dim=1)[0]
        pc[:, :3, :] = points / m.view(-1, 1, 1)
    else:
        m = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
        pc[:, :3, :] = points / m
    return pc


def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def euclidean_distance(src, dst):
    """
    Calculate Euclidean distance

    """
    return torch.norm(src.unsqueeze(-2) - dst.unsqueeze(-3), p=2, dim=-1)


def index_points(points, idx, cuda=False, is_group=False):
    if cuda:
        if is_group:
            points = grouping(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 3, 1).contiguous()
        else:
            points = gathering(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 1).contiguous()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, cuda=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]

    FLOPs:
        S * (3 + 3 + 2)
    """
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        return furthestsampling(xyz, npoint)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz, debug=False, cuda=False):
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return ballquery(radius, nsample, xyz, new_xyz)
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    if debug:
        num_miss = torch.sum(mask)
        num_over = torch.sum(torch.clamp(torch.sum(sqrdists < radius ** 2, dim=2) - nsample, min=0))
        return num_miss, num_over
    return group_idx


def query_knn_point(k, xyz, new_xyz, cuda=False):
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return knnquery(k, xyz, new_xyz)
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :k]
    return group_idx


def sample(nsample, feature, cuda=False):
    feature = feature.permute(0, 2, 1)
    xyz = feature[:, :, :3]

    fps_idx = farthest_point_sample(xyz, nsample, cuda=cuda)  # [B, npoint, C]
    torch.cuda.empty_cache()
    feature = index_points(feature, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()
    feature = feature.permute(0, 2, 1)

    return feature
