import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

root_dir = './data/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to nanjing dataset')

database_dir = join(root_dir, 'database/')
queries_dir = join(root_dir, 'query/')


def input_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_whole_training_set(onlyDB=False):
    structFile = join(root_dir, 'train.txt')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform(), onlyDB=onlyDB)


def get_whole_val_set():
    structFile = join(root_dir, 'val.txt')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform())


def get_whole_test_set():
    structFile = join(root_dir, 'test.txt')
    return WholeDatasetFromStruct(structFile, input_transform=input_transform())


def get_training_query_set(margin=0.1):
    structFile = join(root_dir, 'train.txt')
    return QueryDatasetFromStruct(structFile, input_transform=input_transform(), margin=margin)


def get_val_query_set():
    structFile = join(root_dir, 'val.txt')
    return QueryDatasetFromStruct(structFile, input_transform=input_transform())


def get_test_query_set():
    structFile = join(root_dir, 'test.txt')
    return QueryDatasetFromStruct(structFile, input_transform=input_transform())


dbStruct = namedtuple(
    'dbStruct',
    [
        'whichSet',
        'dataset',
        'dbImage',
        'utmDb',
        'qImage',
        'utmQ',
        'numDb',
        'numQ',
        'posDistThr',
        'posDistSqThr',
        'nonTrivPosDistSqThr',
    ],
)


def parse_dbStruct(path):
    with open(path, 'r') as f:
        data = f.readlines()

    for i in range(len(data)):
        data[i] = data[i][:-1]

    dataset = 'nanjing'

    file_name = path[7:]
    whichSet = file_name[:-4]

    if whichSet == 'train':
        dbImage = data[:28281]
        utmDb_list = data[28281:56562]
        utmDb = [line.split(',') for line in utmDb_list]
        for i in range(len(utmDb)):
            for j in range(2):
                utmDb[i][j] = float(utmDb[i][j])
        utmDb = np.array(utmDb)

        qImage = data[56562:65989]
        utmQ_list = data[65989:75416]
        utmQ = [line.split(',') for line in utmQ_list]
        for i in range(len(utmQ)):
            for j in range(2):
                utmQ[i][j] = float(utmQ[i][j])
        utmQ = np.array(utmQ)

        numDb = 28281
        numQ = 9427

    elif whichSet == 'val':
        dbImage = data[:21210]
        utmDb_list = data[21210:42420]
        utmDb = [line.split(',') for line in utmDb_list]
        for i in range(len(utmDb)):
            for j in range(2):
                utmDb[i][j] = float(utmDb[i][j])
        utmDb = np.array(utmDb)

        qImage = data[42420:49490]
        utmQ_list = data[49490:56560]
        utmQ = [line.split(',') for line in utmQ_list]
        for i in range(len(utmQ)):
            for j in range(2):
                utmQ[i][j] = float(utmQ[i][j])
        utmQ = np.array(utmQ)

        numDb = 21210
        numQ = 7070

    elif whichSet == 'test':
        # dbImage = data[:21216]
        dbImage = data[:10]
        # utmDb_list = data[21216:42432]
        utmDb_list = data[21216:21226]
        utmDb = [line.split(',') for line in utmDb_list]
        for i in range(len(utmDb)):
            for j in range(2):
                utmDb[i][j] = float(utmDb[i][j])
        utmDb = np.array(utmDb)

        # qImage = data[42432:49504]
        qImage = data[42432:40433]
        # utmQ_list = data[49504:56576]
        utmQ_list = data[49504:49505]
        utmQ = [line.split(',') for line in utmQ_list]
        for i in range(len(utmQ)):
            for j in range(2):
                utmQ[i][j] = float(utmQ[i][j])
        utmQ = np.array(utmQ)

        # numDb = 21216
        numDb = 10
        # numQ = 7072
        numQ = 1

    posDistThr = 25
    posDistSqThr = 625
    nonTrivPosDistSqThr = 100

    return dbStruct(
        whichSet,
        dataset,
        dbImage,
        utmDb,
        qImage,
        utmQ,
        numDb,
        numQ,
        posDistThr,
        posDistSqThr,
        nonTrivPosDistSqThr,
    )


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(database_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])  # 根据index加载对应的图片

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):  # 返回列表长度
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range 评估的积极因素是那些在微不足道的阈值范围内的因素
        # fit NN to find them, search by radius 拟合NN找到它们，按半径搜索
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(
                self.dbStruct.utmQ, radius=self.dbStruct.posDistThr
            )

        return self.positives


def collate_fn(batch):  #
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives). 从元组列表创建小批量张量

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))  # 过滤为None的数据
    if len(batch) == 0:
        return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools

    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices


class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample  # number of negatives to randomly sample 随机抽样的底片数
        self.nNeg = nNeg  # number of negatives used for training

        # potential positives are those within nontrivial threshold range 在非平凡的阈值范围内 => 潜在的积极因素
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        self.nontrivial_positives = list(
            knn.radius_neighbors(
                self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5,
                return_distance=False,
            )
        )
        # radius returns unsorted, sort once now so we dont have to later
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives 可能有些查询没有任何非平凡的潜在积极因素
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(
            self.dbStruct.utmQ, radius=self.dbStruct.posDistThr, return_distance=False
        )

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(
                np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True)
            )

        self.cache = None  # filepath of HDF5 containing feature vectors for images 包含图像特征向量的HDF5文件路径

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index]  # re-map index to match dataset 重新映射索引以匹配数据集
        # print('index: ',index)
        with h5py.File(self.cache, mode='r') as h5:
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index + qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1, -1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))
            # print('negsample-type: ',type(negSample))
            # print('negsample-shape: ',negSample.shape)
            # print('negsample: ',negSample)
            # print('negSample.tolist(): ',negSample.tolist())

            int_negSample = list(map(int, negSample.tolist()))
            # print('int_negSample:', int_negSample)
            # negFeat = h5feat[negSample.tolist()]
            negFeat = h5feat[int_negSample]
            # print('negFeat-type:',type(negFeat))
            # print('negFeat:',negFeat)

            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(
                qFeat.reshape(1, -1), self.nNeg * 10
            )  # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none 找出在空白范围内的负片
            violatingNeg = dNeg < dPos + self.margin**0.5

            if np.sum(violatingNeg) < 1:
                # if none are violating then skip this query 确认无违背
                return None

            negNN = negNN[violatingNeg][: self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = Image.open(join(queries_dir, self.dbStruct.qImage[index]))
        positive = Image.open(join(database_dir, self.dbStruct.dbImage[posIndex]))

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(database_dir, self.dbStruct.dbImage[negIndex]))
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self):
        return len(self.queries)


if __name__ == '__main__':
    # a = get_250k_val_set()
    # print("whichSet: " + str(a.dbStruct.whichSet))
    # print("dbImage: " + str(a.dbStruct.dbImage))
    # print("utmDb: " + str(a.dbStruct.utmDb))
    # # print("qImage: " + str(a.dbStruct.qImage))
    # print("utmQ: " + str(a.dbStruct.utmQ))
    # print("numDb: " + str(a.dbStruct.numDb))
    # print("numQ: " + str(a.dbStruct.numQ))
    # print("posDistThr: " + str(a.dbStruct.posDistThr))
    # print("posDistSqThr: " + str(a.dbStruct.posDistSqThr))
    # print("nonTrivPosDistSqThr: " + str(a.dbStruct.nonTrivPosDistSqThr))
    #
    # b = get_250k_val_query_set()
    # print("whichSet: " + str(b.dbStruct.whichSet))
    # # print("dbImage: " + str(b.dbStruct.dbImage))
    # print("utmDb: " + str(b.dbStruct.utmDb))
    # # print("qImage: " + str(b.dbStruct.qImage))
    # print("utmQ: " + str(b.dbStruct.utmQ))
    # print("numDb: " + str(b.dbStruct.numDb))
    # print("numQ: " + str(b.dbStruct.numQ))
    # print("posDistThr: " + str(b.dbStruct.posDistThr))
    # print("posDistSqThr: " + str(b.dbStruct.posDistSqThr))
    # print("nonTrivPosDistSqThr: " + str(b.dbStruct.nonTrivPosDistSqThr))
    parse_dbStruct('./data/train.txt')
