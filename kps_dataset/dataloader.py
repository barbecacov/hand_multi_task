# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/6 17:53
@Auth ： Zhang Di
@File ：dataloader.py
@IDE ：PyCharm
"""
import pdb

import cv2
from torch.utils.data import Dataset
from kps_dataset.transforms import init_data
from kps_dataset.dataset import kps_dataset

from kps_dataset.transforms import texture_aug, Compose, val_resize, ToTensor, padding_resize, init_test_data, Normalize

import torch 
import numpy as np


def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ImageDataset(Dataset):

    def __init__(self, dataset, mode):

        self.dataset = dataset
        self.mode = mode
        if mode == "train":
            self.trans = texture_aug()
        self.pad_reize = padding_resize(160., 160.)
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, img_size, img_box, img_kps = self.dataset[index]

        img_box = np.array(img_box).astype(np.float32)

        img = read_image(img_path)

        if self.mode == "train":
            img, kpts, box = init_data(img, img_kps, img_box, img_size)
            img, kpts, box = self.trans(img, kpts, box)
        else:
            img, kpts, box = init_test_data(img, img_kps, img_box, img_size)

        image, kpts, scale, top_size, left_size = self.pad_reize(img, kpts, box)
        image, kpts = self.to_tensor(image, kpts)
        image, kpts = self.normalize(image, kpts)
        if self.mode == "train":
            return image, kpts
        else:
            return image, kpts, img_box, scale, top_size, left_size

if __name__ == '__main__':

    dataset = kps_dataset(root="/media/tcl2/facepro/HandData")
    dataloader = ImageDataset(dataset.test)
    trans = Compose([
        texture_aug(),
        val_resize(160, 160),
        ToTensor()
    ])

    for i, (im, kp, bx) in enumerate(dataloader):
        im, kp, bx = trans(im, kp, bx)
        # draw_kps(im, kp, bx)
        pdb.set_trace()



