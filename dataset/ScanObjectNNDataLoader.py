"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import h5py
import numpy as np
import warnings
import os
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


class ScanObjectNNDataLoader(Dataset):
    def __init__(self, root, split='training', bg=True):
        self.root = root

        assert (split == 'training' or split == 'test')
        if bg:
            print('Use data with background points')
            dir_name = 'main_split'
        else:
            print('Use data without background points')
            dir_name = 'main_split_nobg'
        file_name = '_objectdataset_augmentedrot_scale75.h5'
        h5_name = '{}/{}/{}'.format(self.root, dir_name, split + file_name)
        with h5py.File(h5_name, mode="r") as f:
            self.data = f['data'][:].astype('float32')
            self.label = f['label'][:].astype('int64')
        print('The size of %s data is %d' % (split, self.data.shape[0]))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].T, self.label[index]
