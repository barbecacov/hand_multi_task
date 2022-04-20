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


class kps_dataset(object):
    def __init__(self, root):
        self.train_dir = os.path.join(root, "KeypointDataOrigi")
        self.test_dir = os.path.join(root, "hand_keypoint_test_0911")
        self.train_gt = os.path.join(root, "tcl_gesture_imgs_20211222.json")
        self.test_gt = os.path.join(root, "tcl_gesture_imgs_clean_test.json")

        self.train = self._process_dir(dir_path=self.train_dir, gt_path=self.train_gt)
        self.test = self._process_dir(dir_path=self.test_dir, gt_path=self.test_gt)

        print("######################")
        print("train dataset nums: ", len(self.train))
        print("test dataset nums:", len(self.test))

    def _process_dir(self, dir_path, gt_path):
        dataset = []
        with open(gt_path, "r") as anno_file:
            anno = json.load(anno_file)
            img_info = anno["images"]
            anno_info = anno["annotations"]
            for im_info, an_info in zip(img_info, anno_info):
                img_path = os.path.join(dir_path, im_info["file_name"])
                img_size = (int(im_info["height"]), int(im_info["width"]))
                img_kps = an_info["keypoints"]
                img_box = an_info["bbox"]

                dataset.append((img_path, img_size, img_box, img_kps))

        anno_file.close()
        return dataset


if __name__ == '__main__':
    dataset = kps_dataset(root="/media/tcl2/facepro/HandData")








