"""
Author: Haoxi Ran
Date: 05/10/2022
"""

from functools import partial

import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path

from dataset.ScanObjectNNDataLoader import ScanObjectNNDataLoader
from modules.ptaug_utils import transform_point_cloud, scale_point_cloud, get_aug_args
from modules.pointnet2_utils import sample
from utils.utils import get_model, get_loss, set_seed, weight_init


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('RepSurf')
    # Basic
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--data_dir', type=str, default='./data', help='data dir')
    parser.add_argument('--log_root', type=str, default='./log', help='log root dir')
    parser.add_argument('--model', default='repsurf.scanobjectnn.repsurf_ssg_umb',
                        help='model file name [default: repsurf_ssg_umb]')
    parser.add_argument('--gpus', nargs='+', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2800, help='Training Seed')
    parser.add_argument('--cuda_ops', action='store_true', default=False,
                        help='Whether to use cuda version operations [default: False]')

    # Training
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training [default: 64]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [Adam, SGD]')
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler for training')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training [default: 500]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--decay_step', default=20, type=int, help='number of epoch per decay [default: 20]')
    parser.add_argument('--n_workers', type=int, default=4, help='DataLoader Workers Number [default: 4]')
    parser.add_argument('--init', type=str, default=None, help='initializer for model [kaiming, xavier]')

    # Evaluation
    parser.add_argument('--min_val', type=int, default=100, help='Min val epoch [default: 100]')

    # Augmentation
    parser.add_argument('--aug_scale', action='store_true', default=False,
                        help='Whether to augment by scaling [default: False]')
    parser.add_argument('--aug_shift', action='store_true', default=False,
                        help='Whether to augment by shifting [default: False]')

    # Modeling
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--return_dist', action='store_true', default=False,
                        help='Whether to use signed distance [default: False]')
    parser.add_argument('--return_center', action='store_true', default=False,
                        help='Whether to return center in surface abstraction [default: False]')
    parser.add_argument('--return_polar', action='store_true', default=False,
                        help='Whether to return polar coordinate in surface abstraction [default: False]')
    parser.add_argument('--group_size', type=int, default=8, help='Size of umbrella group [default: 8]')
    parser.add_argument('--umb_pool', type=str, default='sum', help='pooling for umbrella repsurf [sum, mean, max]')

    return parser.parse_args()


def test(model, loader, num_class=15, num_point=1024, num_votes=10, total_num=1):
    vote_correct = 0
    sing_correct = 0
    classifier = model.eval()

    for j, data in enumerate(loader):
        points, target = data
        points, target = points.cuda(), target.cuda()

        # preprocess
        points = sample(num_point, points)

        # vote
        vote_pool = torch.zeros(target.shape[0], num_class).cuda()
        for i in range(num_votes):
            new_points = points.clone()
            # scale
            if i > 0:
                new_points[:, :3] = scale_point_cloud(new_points[:, :3])
            # predict
            pred = classifier(new_points)
            # single
            if i == 0:
                sing_pred = pred
            # vote
            vote_pool += pred
        vote_pred = vote_pool / num_votes

        # single pred
        sing_pred_choice = sing_pred.data.max(1)[1]
        sing_correct += sing_pred_choice.eq(target.long().data).cpu().sum()
        # vote pred
        vote_pred_choice = vote_pred.data.max(1)[1]
        vote_correct += vote_pred_choice.eq(target.long().data).cpu().sum()

    sing_acc = sing_correct.item() / total_num
    vote_acc = vote_correct.item() / total_num

    return sing_acc, vote_acc


def main(args):
    def log_string(s):
        logger.info(s)
        print(s)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus)
    set_seed(args.seed)

    '''CREATE DIR'''
    experiment_dir = Path(os.path.join(args.log_root, 'PointAnalysis', 'log'))
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('ScanObjectNN')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    args.num_class = 15
    args.dataset = 'ScanObjectNN'
    args.normal = False
    aug_args = get_aug_args(args)
    DATA_PATH = os.path.join(args.data_dir, 'ScanObjectNN')
    TRAIN_DATASET = ScanObjectNNDataLoader(root=DATA_PATH, split='training')
    TEST_DATASET = ScanObjectNNDataLoader(root=DATA_PATH, split='test')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.n_workers, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.n_workers)

    '''MODEL BUILDING'''
    classifier = torch.nn.DataParallel(get_model(args)).cuda()
    criterion = get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        if args.init:
            init_func = partial(weight_init, init_type=args.init)
            classifier = classifier.apply(init_func)

    '''OPTIMIZER'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9)

    '''LR SCHEDULER'''
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.7)
    else:
        raise Exception('No Such Scheduler')

    global_epoch = 0
    global_step = 0
    best_sing_acc = 0.0
    best_vote_acc = 0.0
    loader_len = len(trainDataLoader)

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_loss = []
        train_correct = 0

        scheduler.step()
        for batch_id, data in enumerate(trainDataLoader):
            '''INPUT'''
            points, target = data
            points, target = points.cuda(), target.cuda()

            '''PREPROCESS'''
            points = sample(args.num_point, points)
            points = transform_point_cloud(points, args, aug_args)

            '''FORWARD'''
            optimizer.zero_grad()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            classifier = classifier.train()
            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            train_correct += correct
            train_loss.append(loss.item())

            '''BACKWARD'''
            loss.backward()
            optimizer.step()
            global_step += 1

            if batch_id % 80 == 0:
                print('Epoch: [{0}][{1}/{2}] lr {lr:.6f} loss {loss:.4f}'.
                      format(epoch, batch_id, len(trainDataLoader), lr=lr, loss=loss.item()))

        train_instance_acc = train_correct.item() / (loader_len * args.batch_size)
        train_mean_loss = np.mean(train_loss)
        log_string('Train Instance Accuracy: %.2f, Loss: %f' % (train_instance_acc * 100, train_mean_loss))

        if epoch >= args.min_val:
            with torch.no_grad():
                sing_acc, vote_acc = test(classifier.eval(), testDataLoader, num_point=args.num_point,
                                          total_num=len(TEST_DATASET))

                if sing_acc >= best_sing_acc:
                    best_sing_acc = sing_acc
                if vote_acc >= best_vote_acc:
                    best_vote_acc = vote_acc
                    best_epoch = epoch + 1

                log_string('Test Single Accuracy: %.2f' % (sing_acc * 100))
                log_string('Best Single Accuracy: %.2f' % (best_sing_acc * 100))
                log_string('Test Vote Accuracy: %.2f' % (vote_acc * 100))
                log_string('Best Vote Accuracy: %.2f' % (best_vote_acc * 100))

                if vote_acc >= best_vote_acc:
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'vote_acc': vote_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
