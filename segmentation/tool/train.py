"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import json
import os
import time
from functools import partial

import numpy as np
import argparse
from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from tensorboardX import SummaryWriter
from modules.aug_utils import transform_point_cloud_coord, transform_point_cloud_rgb

from util.utils import AverageMeter, intersectionAndUnionGPU, find_free_port, get_dataset_description, get_optimizer, \
    get_scheduler, get_loop, get_aug_args, get_loss, get_dataset_obj, get_rgb_stat, worker_init_fn
from util.data_util import collate_fn
from util.utils import get_model, get_class_weights, set_seed, main_process, get_logger


def parse_args():
    parser = argparse.ArgumentParser('Model')

    # Basic
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--log_root', type=str, default='./log', help='log root dir')
    parser.add_argument('--dataset', type=str, default='S3DIS', help='dataset name')
    parser.add_argument('--model', default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--gpus', nargs='+', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2000, help='Training Seed')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes participating in the job')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument('--multiprocessing_distributed', action='store_false', default=True,
                        help='Whether to use multiprocessing [default: True]')
    parser.add_argument('--sync_bn', action='store_true', default=False,
                        help='Whether to use sync bn [default: False]')

    # Training
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training [default: 32]')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader Workers Number [default: 4]')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training [SGD, AdamW]')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum [default: 0.9]')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='decay rate [default: 1e-2]')
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler for training [step]')
    parser.add_argument('--learning_rate', default=0.006, type=float, help='init learning rate [default: 0.5]')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='decay rate [default: 0.1]')
    parser.add_argument('--data_norm', type=str, default='mean', help='initializer for model [mean, min]')
    parser.add_argument('--lr_decay_epochs', type=int, default=[60, 80], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start Training Epoch [default: 0]')
    parser.add_argument('--train_freq', type=int, default=250, help='Training frequency [default: 250]')
    parser.add_argument('--resume', type=str, default=None, help='Trained checkpoint path')
    parser.add_argument('--pretrain', type=str, default=None, help='Pretrain model path')

    # Evaluation
    parser.add_argument('--batch_size_val', type=int, default=8, help='batch size in validation [default: 4]')
    parser.add_argument('--min_val', type=int, default=60, help='Min val epoch [default: 60]')
    parser.add_argument('--val_freq', type=int, default=1, help='Val frequency [default: 1]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test [default: 5]')

    # Augmentation
    parser.add_argument('--aug_scale', action='store_true', default=False,
                        help='Whether to augment by scaling [default: False]')
    parser.add_argument('--aug_rotate', type=str, default=None,
                        help='Type to augment by rotation [pert, pert_z, rot, rot_z]')
    parser.add_argument('--aug_jitter', action='store_true', default=False,
                        help='Whether to augment by shifting [default: False]')
    parser.add_argument('--aug_flip', action='store_true', default=False,
                        help='Whether to augment by flipping [default: False]')
    parser.add_argument('--aug_shift', action='store_true', default=False,
                        help='Whether to augment by shifting [default: False]')
    parser.add_argument('--color_contrast', action='store_true', default=False,
                        help='Whether to augment by RGB contrasting [default: False]')
    parser.add_argument('--color_shift', action='store_true', default=False,
                        help='Whether to augment by RGB shifting  [default: False]')
    parser.add_argument('--color_jitter', action='store_true', default=False,
                        help='Whether to augment by RGB jittering [default: False]')
    parser.add_argument('--hs_shift', action='store_true', default=False,
                        help='Whether to augment by HueSaturation shifting [default: False]')
    parser.add_argument('--color_drop', action='store_true', default=False,
                        help='Whether to augment by RGB Dropout [default: False]')

    # RepSurf
    parser.add_argument('--group_size', type=int, default=8, help='Size of umbrella group [default: 8]')
    parser.add_argument('--return_polar', action='store_true', default=False,
                        help='Whether to return polar coordinate in surface abstraction [default: False]')
    parser.add_argument('--freeze_epoch', default=1e6, type=int,
                        help='number of epoch to freeze repsurf [default: 1e6]')

    return parser.parse_args()


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    set_seed(args.seed + args.rank)

    """MODEL BUILDING"""
    # model
    model = get_model(args)

    if main_process(args):
        global logger, writer
        logger = get_logger(args.log_dir, args.model)
        writer = SummaryWriter(args.log_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))  # print args
        logger.info("=> creating models ...")
        logger.info("Classes: {}".format(args.num_class))
        logger.info(model)
        # print num of params
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Total Number of Parameters: {} M'.format(str(float(num_param) / 1e6)[:5]))

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

        if args.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        # model parallel (Note: During DDP Training, enable 'find_unused_parameters' to freeze repsurf)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],
                                                          find_unused_parameters='repsurf' in args.model)
    else:
        # model
        model.cuda()
        model = torch.nn.DataParallel(model)

    """DATA LOADING"""
    coord_transform = transform_point_cloud_coord(args)
    rgb_transform = transform_point_cloud_rgb(args)

    dataset_obj = get_dataset_obj(args)
    rgb_mean, rgb_std = get_rgb_stat(args)
    if 'trainval' not in args.dataset:
        TRAIN_DATASET = dataset_obj(args, 'train', coord_transform, rgb_transform, rgb_mean, rgb_std, True)
        VAL_DATASET = dataset_obj(args, 'val', None, None, rgb_mean, rgb_std, False)
        VAL_DATASET.stop_aug = True
        if main_process(args):
            logger.info("Totally {} samples in {} set.".format(len(TRAIN_DATASET) // args.loop, 'train'))
            logger.info("Totally {} samples in {} set.".format(len(VAL_DATASET) // args.loop, 'val'))
    else:
        TRAIN_DATASET = dataset_obj(args, 'trainval', coord_transform, rgb_transform, rgb_mean, rgb_std, True)
        VAL_DATASET = None
        if main_process(args):
            logger.info("Totally {} samples in {} set.".format(len(TRAIN_DATASET) // args.loop, 'trainval'))

    """DATALOADER BUILDING"""
    train_sampler = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=train_sampler is None,
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True, collate_fn=collate_fn,
                                               worker_init_fn=partial(worker_init_fn, seed=args.seed))

    if VAL_DATASET is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(VAL_DATASET) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler,
                                                 collate_fn=collate_fn,
                                                 worker_init_fn=partial(worker_init_fn, seed=args.seed + 100))

    """TRAINING UTILS"""
    # loss
    label_weight = get_class_weights(args.description).cuda()
    criterion = get_loss(label_weight, args.ignore_label).cuda()
    # optimizer
    optimizer = get_optimizer(args, model)
    # scheduler
    scheduler = get_scheduler(args, optimizer)

    """MODEL RESTORE"""
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if main_process(args):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_iou = checkpoint['best_iou']
            if main_process(args):
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process(args):
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if args.pretrain is not None:
        if os.path.isfile(args.pretrain):
            if main_process(args):
                logger.info("=> loading pretrained model '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=lambda storage, loc: storage.cuda())
            model.load_state_dict(checkpoint['state_dict'], strict=True)

    """TRAINING"""
    for epoch in range(args.start_epoch, args.epoch):
        # train
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch)

        epoch_log = epoch + 1
        if main_process(args):
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        # validate
        is_best = False
        if args.min_val < epoch_log and (epoch_log % args.val_freq == 0) and args.is_eval:
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process(args):
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        # save model
        if is_best and main_process(args):
            filename = args.ckpt_dir + '/model_best.pth'
            # save for publish
            torch.save({'state_dict': model.state_dict()}, filename)
            # save for training
            # torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(),
            #             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            #             'best_iou': best_iou, 'is_best': is_best}, filename)
            logger.info('Best validation mIoU updated to: {:.2f}'.format(best_iou * 100))

    if main_process(args):
        writer.close()
        logger.info('==>Training done!\nBest Iou: {:.2f}'.format(best_iou * 100, ))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()

    # freeze weight
    if args.freeze_epoch < epoch + 1:
        # freeze params
        for n, p in model.module.named_parameters():
            if "surface_constructor" in n and p.requires_grad:
                p.requires_grad = False

    end = time.time()
    max_iter = args.epoch * len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(train_loader):  # [N, 3], [N, C], [N], [B]
        data_time.update(time.time() - end)
        coord, target, feat, offset = \
            coord.cuda(non_blocking=True), target.cuda(non_blocking=True), feat.cuda(non_blocking=True), \
            offset.cuda(non_blocking=True)

        output = model([coord, feat, offset])
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        output = output[:, 1:].max(1)[1] + 1 if 'ScanNet' in args.dataset else output.max(1)[1]  # remove unclassified label
        intersection, union, target = intersectionAndUnionGPU(output, target, args.num_class, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.train_freq == 0 and main_process(args):
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.2f}'.format(epoch + 1, args.epoch, i + 1, len(train_loader),
                                                         batch_time=batch_time, remain_time=remain_time,
                                                         loss_meter=loss_meter, accuracy=accuracy * 100))
        if main_process(args):
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    iou_class = iou_class[1:] if 'ScanNet' in args.dataset else iou_class
    accuracy_class = accuracy_class[1:] if 'ScanNet' in args.dataset else accuracy_class
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process(args):
        for class_idx, class_iou in enumerate(iou_class):
            writer.add_scalar(f'class_{class_idx}_train_iou', class_iou, epoch)
    if main_process(args):
        logger.info('Train result at epoch [{}/{}]: mIoU / mAcc / OA {:.2f} / {:.2f} / {:.2f}'.format(
            epoch + 1, args.epoch, mIoU * 100, mAcc * 100, allAcc * 100))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process(args):
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, target, feat, offset = \
            coord.cuda(non_blocking=True), target.cuda(non_blocking=True), feat.cuda(non_blocking=True), \
            offset.cuda(non_blocking=True)

        with torch.no_grad():
            output = model([coord, feat, offset])

        loss = criterion(output, target)
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        choice = output[:, 1:].max(1)[1] + 1 if 'ScanNet' in args.dataset else output.max(1)[1]  # remove unclassified label
        intersection, union, target = intersectionAndUnionGPU(choice, target, args.num_class, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # remove unlabeled class
    iou_class = iou_class[1:] if 'ScanNet' in args.dataset else iou_class
    accuracy_class = accuracy_class[1:] if 'ScanNet' in args.dataset else accuracy_class
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process(args):
        logger.info('Val result: mIoU / mAcc / OA {:.2f} / {:.2f} / {:.2f}'.format(
            mIoU * 100, mAcc * 100, allAcc * 100))
        logger.info('Val loss: {:.4f}'.format(loss_meter.avg))
        for i in range(len(iou_class)):
            logger.info('Class_{} Result: IoU / Acc {:.2f}/{:.2f}'.format(
                i, iou_class[i] * 100, accuracy_class[i] * 100))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc

    gc.collect()

    """HYPER PARAMETER"""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus)
    if 'A6000' in torch.cuda.get_device_name(0):
        os.environ["NCCL_P2P_DISABLE"] = '1'

    """DDP SETTING"""
    args.dist_url = 'tcp://localhost:8888'
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.gpus)
    if len(args.gpus) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    """CREATE DIR"""
    experiment_dir = Path(os.path.join(args.log_root, 'PointAnalysis', 'log', args.dataset.split('_')[0]))
    if args.log_dir is None:
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    args.ckpt_dir = str(checkpoints_dir)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    args.log_dir = str(log_dir)

    """DATASET INIT"""
    dataset_obj = get_dataset_obj(args)
    args.loop = get_loop(args)
    if args.dataset == 'S3DIS':
        args.num_class, args.voxel_max, args.voxel_size, args.in_channel, args.ignore_label = \
            13, 80000, 0.04, 6, 255
        args.data_dir = './data/S3DIS/trainval_fullarea'
        dataset_obj(args, 'train')
        dataset_obj(args, 'val')
    elif args.dataset == 'ScanNet_train':
        args.num_class, args.voxel_max, args.voxel_size, args.in_channel, args.ignore_label = \
            21, 120000, 0.02, 6, 0
        args.data_dir = './data/ScanNet'
        dataset_obj(args, 'train')
        dataset_obj(args, 'val')
    elif args.dataset == 'ScanNet_trainval':
        args.num_class, args.voxel_max, args.voxel_size, args.in_channel, args.ignore_label = \
            21, 120000, 0.02, 6, 0
        args.data_dir = './data/ScanNet'
        dataset_obj(args, 'trainval')
    else:
        raise Exception('Not Impl. Dataset')

    args.is_eval = 'trainval' not in args.dataset
    args.aug_args = get_aug_args(args)
    args.description = get_dataset_description(args)
    print('Train Model on %s' % args.description)

    """RUNNING"""
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.gpus, args.ngpus_per_node, args)
