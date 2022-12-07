import h5py
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.h5_file = None
        self.length = len(h5py.File(h5_path, 'r'))
        self.transform = transform

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.h5_file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, index):
        record = self.h5_file[str(index)]

        if self.transform:
            x = Image.fromarray(record['data'][()])
            x = self.transform(x)
        else:
            x = torch.from_numpy(record['data'][()])

        y = record['target'][()]
        y = torch.from_numpy(np.asarray(y))

        return (x,y)

    def __len__(self):
        return self.length
