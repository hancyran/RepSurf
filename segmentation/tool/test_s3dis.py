"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import json
import os
import time
import random
import numpy as np
import argparse
import collections
from pathlib import Path

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util.utils import AverageMeter, intersectionAndUnion, get_rgb_stat, pc_median_filter_gpu
from util.utils import get_model, get_logger
from modules.voxelize_utils import voxelize


LABEL2COLOR = collections.OrderedDict([
    ('ceiling', [0, 255, 0]), ('floor', [0, 0, 255]), ('wall', [0, 255, 255]), ('beam', [255, 255, 0]),
    ('column', [255, 0, 255]), ('window', [100, 100, 255]), ('door', [200, 200, 100]), ('chair', [170, 120, 200]),
    ('table', [255, 0, 0]), ('bookcase', [200, 100, 100]), ('sofa', [10, 200, 100]), ('board', [200, 200, 200]),
    ('clutter', [50, 50, 50])])
LABEL2CLASS = list(LABEL2COLOR.keys())
PALETTE = np.array(list(LABEL2COLOR.values()), dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser('Model')

    # Basic
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--data_dir', type=str, default='./data/S3DIS/trainval_fullarea', help='data dir')
    parser.add_argument('--log_root', type=str, default='./log', help='log root dir')
    parser.add_argument('--model_path', type=str, default=None, help='saved model weight')
    parser.add_argument('--model', default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1000, help='Test Seed')

    # Test
    parser.add_argument('--batch_size_test', type=int, default=12, help='batch size in test [default: 24]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test [default: 5]')
    parser.add_argument('--filter', action='store_true', default=False, help='Apply median filter [default: False]')
    parser.add_argument('--data_norm', type=str, default='mean', help='initializer for model [mean, min, z_min]')
    parser.add_argument('--visual', action='store_true', default=False, help='Output visual results [default: False]')

    # Modeling
    parser.add_argument('--group_size', type=int, default=8, help='Size of umbrella group [default: 8]')
    parser.add_argument('--return_polar', action='store_true', default=False,
                        help='Whether to return polar coordinate in surface abstraction [default: False]')

    return parser.parse_args()


def main():
    global args, logger

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.dataset, args.num_class, args.voxel_max, args.voxel_size, args.in_channel, args.ignore_label = \
        'S3DIS', 13, 80000, 0.04, 6, 255

    experiment_dir = Path(os.path.join(args.log_root, 'PointAnalysis', 'log', 'S3DIS'))
    experiment_dir = experiment_dir.joinpath(args.log_dir)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    args.ckpt_dir = str(checkpoints_dir)
    log_dir = experiment_dir.joinpath('logs/')
    args.log_dir = str(log_dir)
    result_dir = experiment_dir.joinpath('visual/')
    result_dir.mkdir(exist_ok=True)
    args.result_dir = str(result_dir)

    logger = get_logger(args.log_dir, 'test_%s' % args.model)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info("=> creating models ...")
    model = get_model(args).cuda()
    logger.info(model)

    ckpt_file = os.path.join(args.ckpt_dir, 'model_best.pth') if args.model_path is None else args.model_path
    if os.path.isfile(ckpt_file):
        logger.info("=> loading checkpoint '{}'".format(ckpt_file))
        checkpoint = torch.load(ckpt_file)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v.cpu()
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}'".format(ckpt_file))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(ckpt_file))

    test(model)


def data_prepare():
    """ Return area names of the test dataset """
    data_list = sorted(os.listdir(args.data_dir))
    data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    print("Totally {} samples in val set.".format(len(data_list)))

    return data_list


def data_load(data_name):
    """ Load data by area name """
    data_path = os.path.join(args.data_dir, data_name + '.npy')
    data = np.load(data_path)  # xyzrgbl, N*7
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

    idx_data = []
    if args.voxel_size:
        idx_sort, count = voxelize(coord - np.min(coord, 0), args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))

    return coord, feat, label, idx_data


def data_process(coord, feat, idx_data):
    """ Split points into batches by index """
    idx_size = len(idx_data)
    idx_list, coord_list, feat_list, offset_list = [], [], [], []
    for i in range(idx_size):
        idx_part = idx_data[i]
        coord_part, feat_part = coord[idx_part], feat[idx_part]
        if args.voxel_max and coord_part.shape[0] > args.voxel_max:
            coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
            while idx_uni.size != idx_part.shape[0]:
                init_idx = np.argmin(coord_p)
                dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                idx_crop = np.argsort(dist)[:args.voxel_max]
                coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                dist = dist[idx_crop]
                delta = np.square(1 - dist / np.max(dist))
                coord_p[idx_crop] += delta
                coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(
                    feat_sub), offset_list.append(idx_sub.size)
                idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
        else:
            coord_part, feat_part = input_normalize(coord_part, feat_part)
            idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(
                feat_part), offset_list.append(idx_part.size)

    return idx_list, coord_list, feat_list, offset_list


def input_normalize(coord, feat):
    # normalize
    if args.data_norm == 'mean':
        coord -= np.mean(coord, 0)
    elif args.data_norm == 'min':
        coord -= np.min(coord, 0)
    else:
        raise Exception('No such data norm type')

    feat = feat / 255.
    if args.color_mean is not None and args.color_std is not None:
        feat = (feat - args.color_mean) / args.color_std
    return coord, feat


def visualize_scene(coord, pred, label, name):
    color_pred = PALETTE[pred.astype(np.int64)]
    color_gt = PALETTE[label.astype(np.int64)]
    pred_save_path = os.path.join(args.result_dir, '{}_pred.txt'.format(name))
    label_save_path = os.path.join(args.result_dir, '{}_label.txt'.format(name))
    np.savetxt(pred_save_path, np.hstack([coord, color_pred]), fmt="%f " * 3 + "%d " * 3)
    np.savetxt(label_save_path, np.hstack([coord, color_gt]), fmt="%f " * 3 + "%d " * 3)


def test(model):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model.eval()

    pred_list, label_list = [], []
    data_list = data_prepare()
    args.color_mean, args.color_std = get_rgb_stat(args)

    for idx_scene, scene_name in enumerate(data_list):
        end = time.time()
        coord, feat, label, idx_data = data_load(scene_name)
        idx_list, coord_list, feat_list, offset_list = data_process(coord, feat, idx_data)

        pred = torch.zeros((label.size, args.num_class)).cuda(non_blocking=True)
        pred_count = torch.zeros((label.size, args.num_class)).cuda(non_blocking=True)
        num_batch = int(np.ceil(len(idx_list) / args.batch_size_test))
        for idx_batch in range(num_batch):
            idx_start = idx_batch * args.batch_size_test
            idx_end = min((idx_batch + 1) * args.batch_size_test, len(idx_list))
            idx_part, coord_part, feat_part, offset_part = \
                idx_list[idx_start:idx_end], coord_list[idx_start:idx_end], \
                feat_list[idx_start:idx_end], offset_list[idx_start:idx_end]

            idx_part = np.concatenate(idx_part)
            coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
            feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
            offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)

            with torch.no_grad():
                pred_part = torch.nn.functional.softmax(model([coord_part, feat_part, offset_part]), dim=1)  # (n, k)
                torch.cuda.empty_cache()

            pred[idx_part, :] += pred_part
            pred_count[idx_part, :] += 1.
            logger.info('Scene {}/{}, {}/{}, {}/{}'.format(idx_scene + 1, len(data_list), idx_end, len(idx_list), args.voxel_max, idx_part.shape[0]))

        # IoU per scene
        pred_choice = np.argmax((pred/pred_count).cpu().numpy(), 1)
        coord = coord
        label = label

        # median filter
        if args.filter:
            coord_gpu = torch.from_numpy(coord).float().cuda(non_blocking=True)
            pred_gpu = torch.from_numpy(pred_choice).int().cuda(non_blocking=True)
            pred_choice = pc_median_filter_gpu(coord_gpu, pred_gpu, 32)

        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}]-{} Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
            idx_scene + 1, len(data_list), label.size, batch_time=batch_time))
        pred_list.append(pred_choice)
        label_list.append(label)

        if args.visual:
            visualize_scene(coord, pred_choice, label, scene_name)

    # mIoU
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_list), np.concatenate(label_list),
                                                       args.num_class, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU / mAcc / OA {:.2f} / {:.2f} / {:.2f}.'.format(mIoU * 100, mAcc * 100, allAcc * 100))

    for i in range(args.num_class):
        logger.info('Class_{} Result: IoU / Acc {:.2f} / {:.2f}, name: {}.'.format(
            i, iou_class[i] * 100, accuracy_class[i] * 100, LABEL2CLASS[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
