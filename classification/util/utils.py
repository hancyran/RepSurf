import importlib
import argparse
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def set_seed(seed):
    """
    Setting of Global Seed

    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu


def weight_init(m, init_type):
    if init_type == 'xavier':
        init_func = torch.nn.init.xavier_normal_
    elif init_type == 'kaiming':
        init_func = torch.nn.init.kaiming_normal_
    else:
        raise Exception('No such init type')

    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
        init_func(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        torch.nn.init.constant_(m.weight, 1)  # constant
        # torch.nn.init.normal_(m.weight, 1.0, 0.02)  # normal
        torch.nn.init.constant_(m.bias, 0)


class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss


class SmoothClsLoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothClsLoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        # log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * pred).sum(dim=1).mean()
        return loss


def get_model(args):
    module = importlib.import_module('models.%s' % args.model)
    return module.Model(args)


def get_loss():
    return SmoothClsLoss()


def get_test_args():
    return argparse.Namespace()
