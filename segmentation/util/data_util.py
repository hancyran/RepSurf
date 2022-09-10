import numpy as np
import SharedArray as SA
import torch

from modules.voxelize_utils import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)

    return torch.cat(coord), torch.cat(feat), torch.cat(label) if label[0] is not None else None, torch.IntTensor(
        offset)


def data_prepare(coord, feat, label, args, split, coord_transform, rgb_transform,
                 rgb_mean=None, rgb_std=None, shuffle_index=True, stop_transform=False):
    dataset = args.dataset.split('_')[0]

    # coordinate augment
    if coord_transform and not stop_transform:
        coord, _, _ = coord_transform(coord, None, None)

    # rgb augment
    if rgb_transform and not stop_transform:
        _, feat, _ = rgb_transform(None, feat, None)

    # grid sampling
    if args.voxel_size:
        uniq_idx = voxelize(coord - np.min(coord, 0), args.voxel_size)
        coord, feat = coord[uniq_idx], feat[uniq_idx]
        if label is not None:
            label = label[uniq_idx]

    # drop points when overflow
    if split != 'val' and args.voxel_max and coord.shape[0] > args.voxel_max:
        init_idx = np.random.randint(coord.shape[0]) if 'train' in split else coord.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:args.voxel_max]
        coord, feat = coord[crop_idx], feat[crop_idx]
        if label is not None:
            label = label[crop_idx]

    # shuffle points
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat = coord[shuf_idx], feat[shuf_idx]
        if label is not None:
            label = label[shuf_idx]

    # coord norm
    if args.data_norm == 'mean':
        coord -= np.mean(coord, 0)
    elif args.data_norm == 'min':
        coord -= np.min(coord, 0)

    # rgb norm
    if dataset in ['S3DIS', 'ScanNet']:
        feat = feat / 255.
        if rgb_mean is not None and rgb_std is not None:
            feat = (feat - rgb_mean) / rgb_std

    return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label) if label is not None else None
