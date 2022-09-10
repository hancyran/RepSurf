"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import os
import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create, data_prepare

NUM_CLASS = 13


class S3DIS(Dataset):
    def __init__(self, args, split, coord_transform=None, rgb_transform=None,
                 rgb_mean=None, rgb_std=None, shuffle_index=False):
        super().__init__()
        self.args, self.split, self.coord_transform, self.rgb_transform, self.rgb_mean, self.rgb_std, self.shuffle_index = \
            args, split, coord_transform, rgb_transform, rgb_mean, rgb_std, shuffle_index
        self.stop_aug = False
        data_list = sorted(os.listdir(args.data_dir))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(args.test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(args.test_area) in item]
        self.data_idx = np.arange(len(self.data_list))

        for item in self.data_list:
            if not os.path.exists("/dev/shm/s3dis_{}".format(item)):
                data_path = os.path.join(args.data_dir, item + '.npy')
                data = np.load(data_path).astype(np.float32)  # xyzrgbl, N*7
                sa_create("shm://s3dis_{}".format(item), data)

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://s3dis_{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = \
            data_prepare(coord, feat, label, self.args, self.split, self.coord_transform, self.rgb_transform,
                         self.rgb_mean, self.rgb_std, self.shuffle_index, self.stop_aug)

        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.args.loop

    @staticmethod
    def print_weight(data_root, data_list):
        print('Computing label weight...')
        num_point_list = []
        label_freq = np.zeros(NUM_CLASS)
        label_total = np.zeros(NUM_CLASS)
        # load data
        for idx, item in enumerate(data_list):
            data_path = os.path.join(data_root, item + '.npy')
            data = np.load(data_path)
            labels = data[:, 6]
            freq = np.histogram(labels, range(NUM_CLASS + 1))[0]
            label_freq += freq
            label_total += (freq > 0).astype(np.float) * labels.size
            num_point_list.append(labels.size)

        # label weight
        label_freq = label_freq / label_total
        label_weight = np.median(label_freq) / label_freq
        print(label_weight)

    @staticmethod
    def print_mean_std(data_root, data_list):
        print('Computing color mean & std...')
        point_list = []
        for idx, item in enumerate(data_list):
            data_path = os.path.join(data_root, item + '.npy')
            data = np.load(data_path)
            point_list.append(data[:, 3:6])

        points = np.vstack(point_list) / 255.
        mean = np.mean(points, 0)
        std = np.std(points, 0)
        print(f'mean: {mean}, std:{std}')
