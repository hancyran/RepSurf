import importlib
import argparse
import random
import logging
import sys

import numpy as np
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

from dataset.S3DISDataLoader import S3DIS
from lib.pointops.functions import pointops


def main_process(args):
    """
    Determine whether the main process

    """
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def set_seed(seed):
    """
    Setting of Global Seed

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=None):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


def get_logger(log_dir, model):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logging.root.handlers = []

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
        handlers=[
            logging.FileHandler('%s/%s.txt' % (log_dir, model)),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logger


def get_aug_args(args):
    dataset = args.dataset
    if 'S3DIS' in dataset:
        aug_args = {'scale_factor': 0.1, 'scale_ani': True, 'scale_prob': 1.,
                    'pert_factor': 0.03, 'pert_prob': 1., 'rot_prob': 0.5,
                    'shifts': [0.1, 0.1, 0.1], 'shift_prob': 1.}
        return aug_args
    else:
        raise Exception('No such dataset')


def get_dataset_obj(args):
    dataset_name = args.dataset
    if 'S3DIS' in dataset_name:
        return S3DIS


def get_dataset_description(args):
    dataset_name = args.dataset
    if 'S3DIS' in dataset_name:
        return '%s_A%d' % (dataset_name, args.test_area)
    if 'ScanNet' in dataset_name:
        return dataset_name


def get_loop(args):
    if 'S3DIS' in args.dataset:
        return 30
    if 'ScanNet' in args.dataset:
        return 6
    else:
        raise Exception('No Fixed Loop for the Dataset')


def get_class_weights(dataset_name):
    # pre-calculate the class weight
    if dataset_name == 'S3DIS_A1':
        num_per_class = [0.27362621, 0.3134626, 0.18798782, 1.38965602, 1.44210271, 0.86639497, 1.07227331,
                         1., 1.05912352, 1.92726327, 0.52329938, 2.04783419, 0.5104427]
    elif dataset_name == 'S3DIS_A2':
        num_per_class = [0.29036634, 0.34709631, 0.19514767, 1.20129272, 1.39663689, 0.87889087, 1.11586938,
                         1., 1.54599972, 1.87057415, 0.56458097, 1.87316536, 0.51576885]
    elif dataset_name == 'S3DIS_A3':
        num_per_class = [0.27578885, 0.32039725, 0.19055443, 1.14914046, 1.46885687, 0.85450877, 1.05414776,
                         1., 1.09680025, 2.09280004, 0.59355243, 1.95746691, 0.50429199]
    elif dataset_name == 'S3DIS_A4':
        num_per_class = [0.27667177, 0.32612854, 0.19886974, 1.18282174, 1.52145143, 0.8793782, 1.14202999,
                         1., 1.0857859, 1.89738584, 0.5964717, 1.95820557, 0.52113351]
    elif dataset_name == 'S3DIS_A5':
        num_per_class = [0.28459923, 0.32990557, 0.1999722, 1.20798185, 1.33784535, 1., 0.93323316, 1.0753585,
                         1.00199521, 1.53657772, 0.7987055, 1.82384844, 0.48565471]
    elif dataset_name == 'S3DIS_A6':
        num_per_class = [0.29442441, 0.37941846, 0.21360804, 0.9812721, 1.40968965, 0.88577139, 1.,
                         1.09387107, 1.53238009, 1.61365643, 1.15693894, 1.57821041, 0.47342451]
    elif dataset_name == 'ScanNet_train':
        num_per_class = [0.32051547, 0.1980627, 0.2621471, 0.74563083, 0.52141879, 0.65918949, 0.73560561, 1.03624985,
                         1.00063147, 0.90604468, 0.43435155, 3.91494446, 1.94558718, 1., 0.54871637, 2.13587716,
                         1.13931665, 2.06423695, 5.59103054, 1.08557339, 1.35027497]
    elif dataset_name == 'ScanNet_trainval':
        num_per_class = [0.32051547, 0.1980627, 0.2621471, 0.74563083, 0.52141879, 0.65918949, 0.73560561, 1.03624985,
                         1.00063147, 0.90604468, 0.43435155, 3.91494446, 1.94558718, 1., 0.54871637, 2.13587716,
                         1.13931665, 2.06423695, 5.59103054, 1.08557339, 1.35027497]
    else:
        raise Exception('No Prepared Class Weights of Dataset')
    return torch.FloatTensor(num_per_class)


def get_rgb_stat(args):
    if 'S3DIS' in args.dataset:
        mean, std = [0.52146571, 0.50457911, 0.44939377], [0.19645595, 0.19576158, 0.20104336]
    elif 'ScanNet' in args.dataset:
        mean, std = [0.08400667, 0.08400667, 0.08400667], [0.28983903, 0.28983903, 0.28983903]
    else:
        return None, None
    return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)


def get_model(args):
    module = importlib.import_module('models.%s' % args.model)
    return module.Model(args)


def get_optimizer(args, model):
    param_dicts = model.parameters()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(param_dicts, lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise Exception('Not impl. such optimizer')
    return optimizer


def get_scheduler(args, optimizer):
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay)
    else:
        raise Exception('Not impl. such scheduler')
    return scheduler


def get_loss(weight=None, ignore_label=None):
    return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)


def get_test_args():
    return argparse.Namespace()


def pc_median_filter_gpu(coord, label, group_size=16):
    """
    :param coord: coordinates of a whole point cloud [N, 3]
    :param label: segmentation results of a whole point cloud [N,]
    :param group_size: num of neighbors for filtering
    """
    offset = torch.IntTensor([coord.shape[0]]).to(coord.device)
    group_idx, _ = pointops.knnquery(group_size, coord, coord, offset, offset)  # [N, group_size]
    group_label = label[group_idx.view(-1).long()].view(coord.shape[0], group_size)  # [N, group_size]
    median_label = torch.median(group_label, 1)[0]
    return median_label.cpu().numpy()
