# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/6 17:53
@Auth ： Zhang Di
@File ：dataloader.py
@IDE ：PyCharm
"""
import json
import os
import pdb
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from cls_dataset.dataset import cls_dataset
from cls_dataset.transforms import init_test_data, init_data

# from transforms import texture_aug, padding_resize, ToTensor, Compose, val_resize
# from dataset import cls_dataset

from torchvision.transforms import transforms

def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ImageDataset(Dataset):

    def __init__(self, dataset, branch, trans, mode):
        self.dataset = dataset
        self.branch = branch
        self.trans = trans
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, img_box, label = self.dataset[index]
        assert os.path.exists(img_path)
        img = read_image(img_path)
        if self.mode == "test":
            img = init_test_data(img, img_box, self.branch)
        else:
            img = init_data(img, img_box, self.branch)
        assert img.shape[0] > 0
        if self.trans is not None:
            img = self.trans(img)
        

        return img, label

if __name__ == '__main__':

    train_trans = transforms.Compose([
        texture_aug(),
        padding_resize(160, 160),
        ToTensor()
    ])

    dataset = cls_dataset(root="/media/tcl2/facepro/HandData", branch="fb")
    img_path, img_box, label = dataset.test[0]
    img = read_image(img_path)
    img = init_test_data(img, img_box, "fb")
    img = train_trans(img)
    pdb.set_trace()
        





