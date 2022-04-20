# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/6 14:40
@Auth ： Zhang Di
@File ：kps.py
@IDE ：PyCharm
"""
import os
import pdb
import json
import glob
import random
from tqdm import tqdm
from collections import Counter

# branch : fb,  isHand, multi_cls, object


class cls_dataset(object):
    def __init__(self, root, branch):
        self.root = root
        self.branch = branch
        if branch == 'fb':
            self.train_json = os.path.join(root, "fb_train.json")
            self.test_json = os.path.join(root, "fb_test.json")
            self.train_img_dir = os.path.join(root, "KeypointDataOrigi")
            self.test_img_dir = os.path.join(root, "hand_keypoint_test_0911")

            self.train = self._process_only_json(self.train_img_dir, self.train_json)
            self.test = self._process_only_json(self.test_img_dir, self.test_json)

        elif branch in ["isHand", "multi_cls"]:
            self.train_img_dir = os.path.join(root, "./train/database")
            self.train_bg_dir = os.path.join(root, "./train/bg")
            self.test_img_dir = os.path.join(root, "test/database")
            self.test_bg_dir = os.path.join(root, "test/bg")

            self.train = self._process_every_json([self.train_img_dir, self.train_bg_dir])
            self.test = self._process_every_json([self.test_img_dir, self.test_bg_dir])

        elif branch == "object":
            self.img_dir = os.path.join(root, "object_or_not")
            all_data = self._process_every_json([self.img_dir])
            random.shuffle(all_data)
            nums = len(all_data)
            test_num = int(0.03 * nums)
            self.test = all_data[:test_num]
            self.train = all_data[test_num:]

        print("Train : there are #{}# samples in {} branch.".format(len(self.train), self.branch))
        print("Test  : there are #{}# samples in {} branch.".format(len(self.test), self.branch))

    def _process_only_json(self, img_dir, json_path):
        dataset = []
        labels = []
        print("Reading Dataset ...")
        with open(json_path, "r") as f:
            anno = json.load(f)
            img_info = anno["images"]
            anno_info = anno["annotations"]
            for im_info, an_info in zip(img_info, anno_info):
                img_path = os.path.join(self.train_img_dir, im_info["file_name"])  # cause all data locate in train_img_dir for fb
                assert os.path.exists(img_path)
                img_box = an_info["bbox"]
                label = an_info[self.branch] - 1  ##### special for fb label
                dataset.append((img_path, img_box, label))
                labels.append(label)
        f.close()

        
        print("labels : ", list(set(labels)))
        co = Counter(labels)
        for k, v in co.items():
            print("label {} have {} samples ".format(k, v))
        print("****************************")
        return dataset

    def _process_every_json(self, dir_list):
        dataset = []
        all_img = []
        labels = []
        for one_dir in dir_list:
            img_list = glob.glob(one_dir + "/*/*/*.jpg")
            all_img.extend(img_list)
        print("Reading Dataset ...")
        for img_path in tqdm(all_img):
            json_path = img_path.replace(".jpg", "_new.json")

            if not os.path.exists(json_path) and "bg" in json_path:
                label = 0
                img_box = None
            else:
                with open(json_path, "r") as f:
                    anno = json.load(f)
                    label = anno[self.branch]
                    img_box = anno["box"]
                    if label == -1:
                        continue
                f.close()
            labels.append(label)

            dataset.append((img_path, img_box, label))

        print("labels : ", list(set(labels)))
        co = Counter(labels)
        for k, v in co.items():
            print("label {} have {} samples ".format(k, v))
        print("****************************")
        return dataset


if __name__ == '__main__':
    dataset = cls_dataset(root="/media/tcl3/facepro/zm/tcl_gesture_data/isHand_multiCls", branch="multi_cls")





