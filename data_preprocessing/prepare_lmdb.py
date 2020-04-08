# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Siva Karthik Mustikovela.
# --------------------------------------------------------

import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import code
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as f

def resize_and_convert(img, quality=100):
    img = f.resize(f.center_crop(img,192),128)
    buffer = BytesIO()
    img.save(buffer, format='png', quality=quality)
    val = buffer.getvalue()
    return val


def resize_img(img, quality=100):
    imgs = []
    imgs.append(resize_and_convert(img, quality))
    return imgs

def resize_worker(img_file):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_img(img)
    return i, out

def prepare(transaction, dataset, n_worker):
    resize_fn = partial(resize_worker)
    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip((128,), imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                txn.put(key, img)
            total += 1
        txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=2)
    parser.add_argument('--center_crop_size', type=int, default=192)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imgset, args.n_worker)