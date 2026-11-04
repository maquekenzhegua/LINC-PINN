import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import random
from typing import List, Sequence, Tuple
from sklearn.model_selection import train_test_split
from utils.util import write_to_txt

# Inner train/validation partition of the training batteries.
#   pooled_random  legacy behaviour: concatenate every training battery, then
#                  sklearn train_test_split(test_size=0.2, random_state=420).
#   chronological  per-battery cycle-order split: the first 80% of each
#                  battery's pairs train, the tail 20% validate, with the pair
#                  straddling the cut dropped so no raw row is shared.
INNER_SPLIT_CHOICES = ("pooled_random", "chronological")

# Cumulative train fractions per segment.  Condition 1 keeps the legacy
# 80/20-then-80/20 structure (64/16/20 over rows); Condition 2 is the 80/20
# split the paper's train_2/valid_2 loaders expose.
CHRONO_FRACTIONS_CONDITION_1 = (0.64, 0.8)
CHRONO_FRACTIONS_CONDITION_2 = (0.8,)


def _resolve_inner_split(args) -> str:
    """Read ``inner_split`` from an argparse namespace (default: legacy)."""
    value = getattr(args, "inner_split", None)
    if value is None:
        return "pooled_random"
    if value not in INNER_SPLIT_CHOICES:
        raise ValueError(
            "inner_split must be one of %s, got %r" % (INNER_SPLIT_CHOICES, value))
    return value


def _chronological_segment_bounds(n_pairs: int, fractions: Sequence[float]) -> List[Tuple[int, int]]:
    """Pair-index bounds of ``len(fractions) + 1`` chronological segments.

    ``fractions`` are cumulative train fractions measured on the pair count
    (``(0.8,)`` for train/valid, ``(0.64, 0.8)`` for train/valid/test).  Each
    cut index ``k`` starts the next segment at ``k + 1``: pair ``k`` spans raw
    rows ``k`` and ``k + 1`` and is therefore dropped, so neighbouring segments
    never share a raw row.  Every segment keeps at least one pair.
    """
    n_segments = len(fractions) + 1
    minimum_pairs = 2 * n_segments - 1
    if n_pairs < minimum_pairs:
        raise ValueError(
            "chronological inner split needs at least %d pairs per battery, got %d"
            % (minimum_pairs, n_pairs))
    bounds: List[Tuple[int, int]] = []
    start = 0
    for index, fraction in enumerate(fractions):
        cut = int(np.floor(fraction * n_pairs))
        cut = max(cut, start + 1)
        # keep at least one pair for every segment that still has to follow
        remaining = n_segments - 1 - index
        cut = min(cut, n_pairs - remaining - 1)
        bounds.append((start, cut))
        start = cut + 1
    bounds.append((start, n_pairs))
    return bounds


def _split_pairs_chronologically(battery_pairs, fractions: Sequence[float]):
    """Split per-battery ``(x1, x2, y1, y2)`` blocks in cycle order.

    ``battery_pairs`` is one ``(x1, x2, y1, y2)`` tuple of numpy arrays per
    training battery, in the caller's battery order.  Returns one
    ``(X1, X2, Y1, Y2)`` tuple of concatenated float tensors per segment, in
    the same battery order, so training/validation keep the original cycle
    layout.
    """
    segments = [[] for _ in range(len(fractions) + 1)]
    for x1, x2, y1, y2 in battery_pairs:
        bounds = _chronological_segment_bounds(len(x1), fractions)
        for segment, (start, stop) in zip(segments, bounds):
            if stop > start:
                segment.append((x1[start:stop], x2[start:stop],
                                y1[start:stop], y2[start:stop]))
    packaged = []
    for segment in segments:
        if not segment:
            raise ValueError("chronological inner split produced an empty segment")
        x1, x2, y1, y2 = (
            torch.from_numpy(np.concatenate([block[col] for block in segment], axis=0)).float()
            for col in range(4))
        packaged.append((x1, x2, y1.view(-1, 1), y2.view(-1, 1)))
    return packaged


class DF():
    def __init__(self,args):
        self.normalization = True
        self.normalization_method = args.normalization_method # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        '''
        :param Ser1:
        :return: index
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self,df):
        '''
        :param df: DataFrame
        :return: DataFrame
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df

    def read_one_csv(self,file_name,nominal_capacity=None):
        '''
        read a csv file and return a DataFrame
        :param file_name: str
        :return: DataFrame
        '''
        df = pd.read_csv(file_name)
        df.insert(df.shape[1]-1,'cycle index',np.arange(df.shape[0]))

        df = self.delete_3_sigma(df)

        if nominal_capacity is not None:
            #print(f'nominal_capacity:{nominal_capacity}, capacity max:{df["capacity"].max()}',end=',')
            df['capacity'] = df['capacity']/nominal_capacity
            #print(f'SOH max:{df["capacity"].max()}')
            f_df = df.iloc[:,:-1]
            if self.normalization_method == 'min-max':
                f_df = 2*(f_df - f_df.min())/(f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean())/f_df.std()

            df.iloc[:,:-1] = f_df

        if getattr(self.args, 'temporal_delta', False):
            # User-specified adjacent-cycle contrast block: central difference
            # (x[i+1]-x[i-1])/2 on interior rows, one-sided differences on the
            # first/last retained rows.  Computed on the min-maxed features of
            # each battery after 3-sigma filtering; bounded in [-2, 2].
            arr = df.iloc[:, :-1].to_numpy(dtype=float)
            feats_ = arr[:, :-1]
            n_rows = feats_.shape[0]
            delta_ = np.zeros_like(feats_)
            if n_rows >= 2:
                delta_[0] = feats_[1] - feats_[0]
                delta_[-1] = feats_[-1] - feats_[-2]
                if n_rows >= 3:
                    delta_[1:-1] = 0.5 * (feats_[2:] - feats_[:-2])
            for k in range(delta_.shape[1]):
                df.insert(df.shape[1] - 1, 'delta_%d' % k, delta_[:, k])

        return df

    def load_one_battery(self,path,nominal_capacity=None):
        '''
        Read a csv file and divide the data into x and y
        :param path:
        :param nominal_capacity:
        :return:
        '''
        df = self.read_one_csv(path,nominal_capacity)
        x = df.iloc[:,:-1].values
        y = df.iloc[:,-1].values
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        return (x1,y1),(x2,y2)

    def load_all_battery(self,path_list,nominal_capacity):
        '''
        Read multiple csv files, divide the data into X and Y, and then package it into a dataloader
        :param path_list: list of file paths
        :param nominal_capacity: nominal capacity, used to calculate SOH
        :param batch_size: batch size
        :return: Dataloader
        '''
        inner_split = _resolve_inner_split(self.args)
        X1, X2, Y1, Y2 = [], [], [], []
        battery_pairs = []
        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder,self.args.log_dir)
            write_to_txt(save_name,'data path:')
            write_to_txt(save_name,str(path_list))
            write_to_txt(save_name,'inner split: ' + inner_split)
        for path in path_list:
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
            # print(path)
            # print(x1.shape, x2.shape, y1.shape, y2.shape)
            battery_pairs.append((x1, x2, y1, y2))
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)


        tensor_X1 = torch.from_numpy(X1).float()
        tensor_X2 = torch.from_numpy(X2).float()
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1,1)
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1,1)
        # print('X shape:',tensor_X1.shape)
        # print('Y shape:',tensor_Y1.shape)

        # 有时候需要指定训练集和测试集的电池ID，因此这个函数返回一个字典，里面包含多种情况，
        # 可根据需要选择
        # 1. 传入的path_list是【训练集】、【验证集】和【测试集】的电池ID，这时候按照前80%训练，后20%测试划分，再从训练集中随机划分出验证集，比例为8:2
        # 2. 传入的path_list是【训练集】和【测试集】的电池ID，这种情况只需要按照8:2随机化划分训练集和测试集即可
        # 3. 传入的path_list是【测试集】的电池ID，则不需要划分，直接封装成dataloader即可
        ## English version
        # Sometimes it is necessary to specify the battery ID of the training set and test set,
        # so this function returns a dictionary containing a variety of situations,
        # You can choose according to your needs
        # 1. The incoming path_list is the battery ID of [training set], [validation set] and [test set].
        #     At this time, it is divided into the first 80% for training and the last 20% for testing,
        #     and then the validation set is randomly divided from the training set. The ratio is 8:2
        # 2. The incoming path_list is the battery ID of [training set] and [testing set].
        #     In this case, you only need to randomly divide the training set and test set according to 8:2.
        # 3. The incoming path_list is the battery ID of the [test set], so there is no need to divide it and it can be directly encapsulated into a dataloader.
        #
        # inner_split='chronological' replaces both random partitions below with a
        # per-battery cycle-order split (see _split_pairs_chronologically): every
        # training battery contributes its earliest pairs to train and its latest
        # pairs to validation, and the pair straddling the cut is dropped so the
        # two sides never share a raw row.  inner_split='pooled_random' (default)
        # reproduces the legacy behaviour bit for bit.
        if inner_split == 'chronological':
            (train_X1, train_X2, train_Y1, train_Y2), \
                (valid_X1, valid_X2, valid_Y1, valid_Y2), \
                (test_X1, test_X2, test_Y1, test_Y2) = _split_pairs_chronologically(
                    battery_pairs, CHRONO_FRACTIONS_CONDITION_1)
            (train_X1_2, train_X2_2, train_Y1_2, train_Y2_2), \
                (valid_X1_2, valid_X2_2, valid_Y1_2, valid_Y2_2) = _split_pairs_chronologically(
                    battery_pairs, CHRONO_FRACTIONS_CONDITION_2)
        else:
            # Condition 1
            # 1.1 划分训练集和测试集
            split = int(tensor_X1.shape[0] * 0.8)
            train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
            train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
            train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
            train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]
            # 1.2 划分训练集和验证集
            train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
                train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)

            # Condition 2
            train_X1_2, valid_X1_2, train_X2_2, valid_X2_2, train_Y1_2, valid_Y1_2, train_Y2_2, valid_Y2_2 = \
                train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)

        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)

        train_loader_2 = DataLoader(TensorDataset(train_X1_2, train_X2_2, train_Y1_2, train_Y2_2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        valid_loader_2 = DataLoader(TensorDataset(valid_X1_2, valid_X2_2, valid_Y1_2, valid_Y2_2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)

        # Condition 3
        test_loader_3 = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)


        loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'train_2': train_loader_2,'valid_2': valid_loader_2,
                  'test_3': test_loader_3}
        return loader




class XJTUdata(DF):
    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.file_list = sorted(os.listdir(root))
        self.variables = pd.read_csv(os.path.join(root, self.file_list[0])).columns
        self.num = len(self.file_list)
        self.batch_names = ['2C','3C','R2.5','R3','RW','satellite']
        self.batch_size = args.batch_size

        if self.normalization:
            self.nominal_capacity = 2.0
        else:
            self.nominal_capacity = None
        #print('-'*20,'XJTU data','-'*20)

    def read_one_batch(self,batch='2C'):
        '''
        读取一个批次的csv文件,并把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version: Read a batch of csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :param batch: int or str:batch
        :return: dict
        '''
        if isinstance(batch,int):
            batch = self.batch_names[batch]
        assert batch in self.batch_names, 'batch must be in {}'.format(self.batch_names)
        file_list = []
        for i in range(self.num):
            if batch in self.file_list[i]:
                path = os.path.join(self.root,self.file_list[i])
                file_list.append(path)
        return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件，并把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version: Read all csv files, divide the data into four parts: x1, y1, x2, y2, and encapsulate it into a dataloader
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for file in self.file_list:
                path = os.path.join(self.root, file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list,nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list,nominal_capacity=self.nominal_capacity)


class HUSTdata(DF):
    def __init__(self,root='../data/HUST data',args=None):
        super(HUSTdata, self).__init__(args)
        self.root = root
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-'*20,'HUST data','-'*20)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件:如果指定了specific_path_list,则读取指定的文件；否则读取所有文件；
        把数据分成x1,y1,x2,y2四部分，并封装成dataloader
        English version:
        Read all csv files.
        If specific_path_list is not None, read the specified file;
        otherwise read all files;
        :param self:
        :param specific_path:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            files = sorted(os.listdir(self.root))
            for file in files:
                path = os.path.join(self.root,file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)


class MITdata(DF):
    def __init__(self,root='../data/MIT data',args=None):
        super(MITdata, self).__init__(args)
        self.root = root
        self.batchs = ['2017-05-12','2017-06-30','2018-04-12']
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None
        #print('-' * 20, 'MIT data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        读取一个批次的csv文件
        English version: Read a batch of csv files
        :param batch: int,可选[1,2,3]
        :return: dict
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = sorted(os.listdir(root))
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)

    def read_all(self,specific_path_list=None):
        '''
        读取所有csv文件。如果指定了specific_path_list,则读取指定的文件；否则读取所有文件；封装成dataloader
        English version:
        Read all csv files.
        If specific_path_list is not None, read the specified file; otherwise read all files;
        :param self:
        :return: dict
        '''
        if specific_path_list is None:
            file_list = []
            for batch in self.batchs:
                root = os.path.join(self.root,batch)
                files = sorted(os.listdir(root))
                for file in files:
                    path = os.path.join(root,file)
                    file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)

class TJUdata(DF):
    def __init__(self,root='../data/TJU data',args=None):
        super(TJUdata, self).__init__(args)
        self.root = root
        self.batchs = ['Dataset_1_NCA_battery','Dataset_2_NCM_battery','Dataset_3_NCM_NCA_battery']
        if self.normalization:
            self.nominal_capacities = [3.5,3.5,2.5]
        else:
            self.nominal_capacities = [None,None,None]
        #print('-' * 20, 'TJU data', '-' * 20)

    def read_one_batch(self,batch):
        '''
        读取一个批次的csv文件
        English version: Read a batch of csv files
        :param batch: int,可选[1,2,3]; optional[1,2,3]
        :return: DataFrame
        '''
        assert batch in [1,2,3], 'batch must be in {}'.format([1,2,3])
        root = os.path.join(self.root,self.batchs[batch-1])
        file_list = sorted(os.listdir(root))
        df = pd.DataFrame()
        path_list = []
        for file in file_list:
            file_name = os.path.join(root,file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacities[batch - 1])

    def read_all(self,specific_path_list):
        '''
        读取所有csv文件,封装成dataloader
        English version: Read all csv files and encapsulate them into a dataloader
        :param self:
        :return: dict
        '''
        for i,batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)


if __name__ == '__main__':
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data',type=str,default='MIT',help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch',type=int,default=1,help='1,2,3')
        parser.add_argument('--batch_size',type=int,default=256,help='batch size')
        parser.add_argument('--normalization_method',type=str,default='z-score',help='min-max,z-score')
        parser.add_argument('--log_dir',type=str,default='test.txt',help='log dir')
        return parser.parse_args()

    args = get_args()

    # xjtu = XJTUdata(root='../data/XJTU data',args=args)
    # path = '../data/XJTU data/2C_battery-1.csv'
    # xjtu.read_one_batch('2C')
    # xjtu.read_all()
    #
    # hust = HUSTdata(args=args)
    # hust.read_all()
    #
    mit = MITdata(args=args)
    mit.read_one_batch(batch=1)
    loader = mit.read_all()

    # tju = HUSTdata(args=args)
    # loader = tju.read_all()
    train_loader = loader['train']
    test_loader = loader['test']
    valid_loader = loader['valid']
    all_loader = loader['test_3']
    print('train_loader:',len(train_loader),'test_loader:',len(test_loader),'valid_loader:',len(valid_loader),'all_loader:',len(all_loader))

    for iter,(x1,x2,y1,y2) in enumerate(train_loader):
        print('x1 shape:',x1.shape)
        print('x2 shape:',x2.shape)
        print('y1 shape:',y1.shape)
        print('y2 shape:',y2.shape)
        print('y1 max:',y1.max())
        break







