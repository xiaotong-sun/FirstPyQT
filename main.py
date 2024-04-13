from __future__ import print_function
import os
import warnings

# 该语句是python2的概念，在python2的环境下超前使用python3的print函数。
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss

from tensorboardX import SummaryWriter
import numpy as np
import netvlad

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument(
    '--mode', type=str, default='test', help='Mode', choices=['train', 'test', 'cluster']
)
parser.add_argument(
    '--batchSize',
    type=int,
    default=4,
    help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.',
)
parser.add_argument(
    '--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing'
)
parser.add_argument(
    '--cacheRefreshRate',
    type=int,
    default=1000,
    help='How often to refresh cache, in number of queries. 0 for off',
)
parser.add_argument('--nEpochs', type=int, default=5, help='number of epochs to train for')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)',
)
# metavar参数用来控制部分命令行参数的显示，不影响代码内部获取命令行参数的对象。
parser.add_argument('--nGPU', type=int, default=4, help='number of GPU to use.')
parser.add_argument(
    '--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM']
)
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument(
    '--threads', type=int, default=8, help='Number of threads for each data loader to use'
)
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='./data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
parser.add_argument(
    '--savePath',
    type=str,
    default='./checkpoints/',
    help='Path to save checkpoints to in logdir. Default=checkpoints/',
)
parser.add_argument('--cachePath', type=str, default='./tmp/', help='Path to save cache to.')
parser.add_argument(
    '--resume',
    type=str,
    default='runs/Aug28_13-18-14_vgg16_netvlad',
    help='Path to load checkpoint from, for resuming training or testing.',
)
parser.add_argument(
    '--ckpt',
    type=str,
    default='latest',
    help='Resume from latest or best checkpoint.',
    choices=['latest', 'best'],
)
parser.add_argument(
    '--evalEvery', type=int, default=1, help='Do a validation set run, and save, every N epochs.'
)
parser.add_argument(
    '--patience', type=int, default=10, help='Patience for early stopping. 0 is off.'
)
parser.add_argument(
    '--dataset', type=str, default='nanjing', help='Dataset to use', choices=['nanjing']
)
parser.add_argument(
    '--arch', type=str, default='vgg16', help='basenetwork to use', choices=['vgg16', 'alexnet']
)
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument(
    '--pooling',
    type=str,
    default='netvlad',
    help='type of pooling to use',
    choices=['netvlad', 'max', 'avg'],
)
parser.add_argument(
    '--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64'
)
parser.add_argument(
    '--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1'
)
parser.add_argument(
    '--split',
    type=str,
    default='test',
    help='Data split to use for testing. Default is val',
    choices=['test', 'train', 'val'],
)
parser.add_argument(
    '--fromscratch',
    action='store_true',
    help='Train from scratch rather than using pretrained models',
)


def train(epoch):
    epoch_loss = 0
    startIter = (
        1  # keep track of batch iter across subsets for logging  跟踪跨子集批量迭代进行日志记录
    )

    if opt.cacheRefreshRate > 0:  # 缓存刷新率>0
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)  # 子集数
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)  # 子集索引
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()  # 评估
        train_set.cache = join(opt.cachePath, train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5:  # 写入
            pool_size = encoder_dim  # 池化层尺寸与用到的神经网络维度相匹配
            if opt.pooling.lower() == 'netvlad':
                pool_size *= opt.num_clusters
            h5feat = h5.create_dataset(
                "features", [len(whole_train_set), pool_size], dtype=np.float32
            )
            # h5存储图像特征
            with torch.no_grad():  # 上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
                for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                    # 对于一个可迭代的（iterable）/可遍历的对象，enumerate将其组成一个索引序列，利用它可以同时获得索引和值
                    # enumerate用在for循环中得到计数
                    input = input.to(device)
                    image_encoding = model.encoder(input)
                    vlad_encoding = model.pool(image_encoding)
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    # detach().numpy()=data.numpy()=item(), 取出数值
                    del input, image_encoding, vlad_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(
            dataset=sub_train_set,
            num_workers=0,
            batch_size=opt.batchSize,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            pin_memory=cuda,
        )
        # 为解决报错， num_workers=opt.threads => num_workers=0
        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_cached())

        model.train()  # 训练
        for iteration, (query, positives, negatives, negCounts, indices) in enumerate(
            training_data_loader, startIter
        ):
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])

            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()  # 梯度置零

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i : i + 1], vladP[i : i + 1], vladN[negIx : negIx + 1])

            loss /= nNeg.float().to(device)  # normalize by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print(
                    "==> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                        epoch, iteration, nBatches, batch_loss
                    ),
                    flush=True,
                )
                writer.add_scalar('Train/Loss', batch_loss, ((epoch - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, ((epoch - 1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache)  # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)


def test(eval_set, opt, epoch=0, write_tboard=False):
    test_data_loader = DataLoader(
        dataset=eval_set,
        num_workers=0,
        batch_size=opt.cacheBatchSize,
        shuffle=False,
        pin_memory=cuda,
    )

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad':
            pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb :].astype('float32')
    dbFeat = dbFeat[: eval_set.dbStruct.numDb].astype('float32')

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20]

    _, predictions = faiss_index.search(qFeat, max(n_values))

    # for each query get those within threshold distance
    gt = eval_set.getPositives()

    correct_at_n = np.zeros(len(n_values))

    predIndex = []

    for qIx, pred in enumerate(predictions):
        predIndex = pred
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    # TODO 处理predIndex以得到图片的路径
    predIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    file_path = []
    for index in predIndex:
        path = './data/database/' + eval_set.dbStruct.dbImage[index]
        file_path.append(path)

    return recalls, file_path


def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors / nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(
        dataset=cluster_set,
        num_workers=0,
        batch_size=opt.cacheBatchSize,
        shuffle=False,
        pin_memory=cuda,
        sampler=sampler,
    )

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(
        opt.dataPath,
        'centroids',
        opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5',
    )  # Str()将数字型变量或常量改变成字符型变量或常量
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", [nDescriptors, encoder_dim], dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = (
                    model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)
                )

                batchix = (iteration - 1) * opt.cacheBatchSize * nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix * nPerImage
                    dbFeat[startix : startix + nPerImage, :] = (
                        image_descriptors[ix, sample, :].detach().cpu().numpy()
                    )
                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print(
                        "==> Batch ({}/{})".format(iteration, ceil(nIm / opt.cacheBatchSize)),
                        flush=True,
                    )
                del input, image_descriptors

        print('====> Clustering..')
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, verbose=False)
        # niter：迭代次数 verbose：是否打印迭代情况
        # pool_size *= opt.num_clusters
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def searchForImg(img, utm):
    warnings.filterwarnings('ignore')
    opt = parser.parse_args()

    restore_var = [
        'lr',
        'lrStep',
        'lrGamma',
        'weightDecay',
        'momentum',
        'runsPath',
        'savePath',
        'arch',
        'num_clusters',
        'pooling',
        'optim',
        'margin',
        'seed',
        'patience',
    ]
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {
                    '--' + k: str(v) for k, v in json.load(f).items() if k in restore_var
                }
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del:
                    del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                # print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    if opt.dataset.lower() == 'nanjing':
        import nanjing as dataset
    else:
        raise Exception('Unknown dataset')

    global cuda
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    global device
    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')

    if opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_test_set_for_one_query(img=img, utm=utm)
            print('===> Evaluating on test set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)

    print('===> Building model')

    pretrained = not opt.fromscratch
    global encoder_dim
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

    encoder = nn.Sequential(*layers)
    global model
    model = nn.Module()
    model.add_module('encoder', encoder)

    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(
                num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2
            )
            if not opt.resume:
                if opt.mode.lower() == 'test':
                    initcache = join(
                        opt.dataPath,
                        'centroids',
                        opt.arch
                        + '_'
                        + whole_test_set.dataset
                        + '_'
                        + str(opt.num_clusters)
                        + '_desc_cen.hdf5',
                    )

                if not exists(initcache):
                    raise FileNotFoundError(
                        'Could not find clusters, please run with --mode=cluster before proceeding'
                    )

                with h5py.File(initcache, mode='r') as h5:
                    print('h5_type:', type(h5))
                    print('h5:', h5)
                    print('h5.get("centroids"):', h5.get("centroids"))  # h5.get("centroids"): None
                    print('h5.get("descriptors"):', h5.get("descriptors"))

                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                net_vlad.init_params(clsts, traindescs)
                del clsts, traindescs

            model.add_module('pool', net_vlad)
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        _, file_path = test(whole_test_set, opt=opt, epoch=epoch, write_tboard=False)

    return file_path


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    opt = parser.parse_args()

    restore_var = [
        'lr',
        'lrStep',
        'lrGamma',
        'weightDecay',
        'momentum',
        'runsPath',
        'savePath',
        'arch',
        'num_clusters',
        'pooling',
        'optim',
        'margin',
        'seed',
        'patience',
    ]
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {
                    '--' + k: str(v) for k, v in json.load(f).items() if k in restore_var
                }
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del:
                    del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    if opt.dataset.lower() == 'nanjing':
        import nanjing as dataset
    else:
        raise Exception('Unknown dataset')

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(
            dataset=whole_train_set,
            num_workers=0,
            batch_size=opt.cacheBatchSize,
            shuffle=False,
            pin_memory=cuda,
        )

        train_set = dataset.get_training_query_set(opt.margin)

        print('====> Training query set:', len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on test set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif opt.split.lower() == 'val':
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    print('===> Building model')

    pretrained = not opt.fromscratch
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

    if opt.mode.lower() == 'cluster' and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)

    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(
                num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2
            )
            if not opt.resume:
                if opt.mode.lower() == 'train':
                    initcache = join(
                        opt.dataPath,
                        'centroids',
                        opt.arch
                        + '_'
                        + train_set.dataset
                        + '_'
                        + str(opt.num_clusters)
                        + '_desc_cen.hdf5',
                    )
                else:
                    initcache = join(
                        opt.dataPath,
                        'centroids',
                        opt.arch
                        + '_'
                        + whole_test_set.dataset
                        + '_'
                        + str(opt.num_clusters)
                        + '_desc_cen.hdf5',
                    )

                if not exists(initcache):
                    raise FileNotFoundError(
                        'Could not find clusters, please run with --mode=cluster before proceeding'
                    )

                with h5py.File(initcache, mode='r') as h5:
                    print('h5_type:', type(h5))
                    print('h5:', h5)
                    print('h5.get("centroids"):', h5.get("centroids"))  # h5.get("centroids"): None
                    print('h5.get("descriptors"):', h5.get("descriptors"))

                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                net_vlad.init_params(clsts, traindescs)
                del clsts, traindescs

            model.add_module('pool', net_vlad)
        elif opt.pooling.lower() == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1, 1))
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)

    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr
            )  # , betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay,
            )
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.lrStep, gamma=opt.lrGamma
            )
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recalls = test(whole_test_set, opt=opt, epoch=epoch, write_tboard=False)
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_train_set)
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(
            log_dir=join(
                opt.runsPath,
                datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.arch + '_' + opt.pooling,
            )
        )

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        # opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps({k: v for k, v in vars(opt).items()}))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch + 1, opt.nEpochs + 1):

            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, opt=opt, epoch=epoch, write_tboard=True)
                is_best = recalls[5] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else:
                    not_improved += 1
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)

                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer': optimizer.state_dict(),
                        'parallel': isParallel,
                    },
                    is_best,
                )

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
