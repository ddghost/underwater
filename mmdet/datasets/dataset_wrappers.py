import numpy as np
import math
import random
from torch.utils.data import dataset
from .registry import DATASETS


class mixDataset(Dataset):

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "mixDataset does not support IterableDataset"
        self.pickPossibility = 0.2

    def __len__(self):
        return self.datasets[0]

    def __getitem__(self, idx):

        if(random.random() < self.pickPossibility):
            dataset_idx = 0
        else:
            dataset_idx = 1
            if(idx >= self.__len__() ):
                 idx = random.random() * len(self.datasets[1])
                 idx = math.floor(idx)
        return self.datasets[dataset_idx][idx]

@DATASETS.register_module
class ConcatDataset(mixDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = datasets[0].flag
            print(datasets[0].flag,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(datasets[1].flag,'***********************************')


@DATASETS.register_module
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len
