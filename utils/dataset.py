# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Siva Karthik Mustikovela.
# --------------------------------------------------------

from io import BytesIO
import lmdb
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import code
import random
import numpy as np
import torch


class lmdbDataset(Dataset):
    def __init__(self, path, transform):
        self.env = lmdb.open(path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False,)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = 128
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)

        img = self.transform(img)
        fl_img = torch.flip(img, [2])

        return img, fl_img


class lmdbDataset_withGT(Dataset):
    def __init__(self, path, transform):
        self.env = lmdb.open(path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False,)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = 128
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'im-{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            key_a = f'gt-{self.resolution}-{str(index).zfill(5)}_a'.encode('utf-8')
            key_e = f'gt-{self.resolution}-{str(index).zfill(5)}_e'.encode('utf-8')
            key_t = f'gt-{self.resolution}-{str(index).zfill(5)}_t'.encode('utf-8')
            img_bytes = txn.get(key)

            gt_a = txn.get(key_a).decode()
            gt_e = txn.get(key_e).decode()
            gt_t = txn.get(key_t).decode()
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img, gt_a, gt_e, gt_t


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path       