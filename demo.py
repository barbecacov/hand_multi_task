# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/11 10:10
@Auth ： Zhang Di
@File ：cal_reg_pck.py
@IDE ：PyCharm
"""
from model.reslike_ml import resnet_mt
import torch
import os
import numpy as np
import torch.nn as nn
import json
import argparse
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2
import math
import time
import pdb


parser = argparse.ArgumentParser(description="OKS benchmark")
parser.add_argument("--anno_json_path", type=str, default="/media/tcl2/facepro/HandData/tcl_gesture_imgs_clean_test.json")
parser.add_argument("--img_path", type=str, default="/media/tcl2/facepro/HandData/hand_keypoint_test_0911")
parser.add_argument("--model-path", type=str, default="pretrained_model/pytorch_fp32/model_best_0.95408.pth")
args = parser.parse_args()


point_size = 2
point_color = (0, 0, 255)  # BGR


def draw_points(img, coord):
    points = coord
    for i, p in enumerate(points):
        if i < 5:
            color = (255, 0, 0)
        elif i >= 5 and i <= 8:
            color = (0, 0, 255)
        elif i >= 9 and i <= 12:
            color = (0, 255, 0)
        elif i >= 13 and i <= 16:
            color = (255, 255, 0)
        elif i >= 17 and i <= 20:
            color = (0, 255, 255)
        cv2.circle(img, (int(p[0]+0.5), int(p[1]+0.5)), point_size, color, -1)
        # cv2.putText(crop_img, str(i), p, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        # cv2.imshow("1", img)
        # cv2.waitKey(0)

    # 画直线
    edges = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]
    for x, y in edges:
        if points[x][0] > 0 and points[x][1] > 0 and points[y][0] > 0 and points[y][1] > 0:
            cv2.line(img, points[x], points[y], point_color, 1)

    return img


anno = json.load(open(args.anno_json_path))
coco = COCO(args.anno_json_path)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
nms = set([cat['supercategory'] for cat in cats])

images_anno = {}
keypoint_annos = {}
bbox_annos = {}
cs_annos = {}
transform = list(zip(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
))
img_list = []

for img_info, anno_info in zip(anno["images"], anno["annotations"]):
    images_anno[img_info['id']] = img_info
    if len(anno_info['keypoints']) != 3 * 21:
        print(img_info)
        print(len(anno_info['keypoints']))
    prev_xs = anno_info['keypoints'][0::3]
    prev_ys = anno_info['keypoints'][1::3]
    cs = anno_info['keypoints'][2::3]
    new_kp = []
    for idx, idy in transform:
        new_kp.append(
            (prev_xs[idx - 1], prev_ys[idy - 1])
        )

    keypoint_annos[anno_info['image_id']] = new_kp
    bbox_annos[anno_info['image_id']] = anno_info['bbox']
    cs_annos[anno_info['image_id']] = cs

max_pck = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet_mt(branch="kps")
check_point = torch.load(args.model_path, map_location="cpu")["net"]
model.load_state_dict(check_point, strict=False)
model.to(device)
model.eval()

use_times = []
count = {}
for i in range(21):
    count[i] = 0

####### 导入模型 ########
stat_error = [0] * 22
stat_pck = np.array([0] * 21)


for key in tqdm(images_anno.keys()):
    ori_img = cv2.imread(os.path.join(args.img_path, images_anno[key]['file_name']))
    # if os.path.join(args.img_path, images_anno[key]['file_name']) == '/media/tcl2/facepro/HandData/hand_keypoint_test_0911/tj_20200519_task4_0035.jpg':
    #     pdb.set_trace()
    h_inp, w_inp = ori_img.shape[0], ori_img.shape[1]
    bboxs = bbox_annos[key]
    keypoints = keypoint_annos[key]
    css = cs_annos[key]
    x1, y1, w, h = bboxs
    x2 = x1 + w
    y2 = y1 + h
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    wh = max(w, h)

    alpha = 0.1
    x1_ = int(max(0, xc - (alpha+0.5) * wh) + 0.5)
    x2_ = int(min(w_inp, xc + (alpha+0.5) * wh) + 0.5)
    y1_ = int(max(0, yc - (alpha+0.5) * wh) + 0.5)
    y2_ = int(min(h_inp, yc + (alpha+0.5) * wh) + 0.5)
    temp = ori_img[y1_:y2_, x1_:x2_]
    crop_img = temp.copy()
    crop_img_gt = temp.copy()

    gt_keypoints = []
    for point in keypoints:
        if x1_ < point[0] < x2_ and y1_ < point[1] < y2_:
            p0 = point[0] - x1_
            p1 = point[1] - y1_
            gt_keypoints.append((int(p0 + 0.5), int(p1 + 0.5)))
        else:
            gt_keypoints.append((-1, -1))
    h, w = crop_img.shape[0], crop_img.shape[1]
    input_h, input_w = 160, 160
    constraint = int(input_w / 2 - 1)
    scale_h, scale_w = input_h / h, input_w / w
    scale = min(scale_w, scale_h)
    inp_img = cv2.resize(crop_img, (int(w * scale + 0.5), int(h * scale + 0.5)))
    re_h, re_w = inp_img.shape[0], inp_img.shape[1]
    if input_h > re_h:
        top_size = int((input_h - re_h) / 2)
        bottom_size = input_h - top_size - re_h
        left_size = 0
        right_size = 0
    elif input_w > re_w:
        left_size = int((input_w - re_w) / 2)
        right_size = input_w - left_size - re_w
        top_size = 0
        bottom_size = 0
    else:
        left_size = 0
        right_size = 0
        top_size = 0
        bottom_size = 0
    image_border = cv2.copyMakeBorder(inp_img, top_size, bottom_size, left_size, right_size,
                                      borderType=cv2.BORDER_CONSTANT, value=0)

    ##### infer #######
    if 1:
        st = time.time()

        input = cv2.cvtColor(image_border, cv2.COLOR_BGR2RGB)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        input = torch.from_numpy(input).div(255.)
        input = torch.unsqueeze(input, 0)
        input = input.to(device)
        reg = model(input)
        # print(reg)
        et = time.time()
        reg_ = reg[0].cpu().detach().numpy() * input_h

        # print("img_id = %d, cost_time = %.2f ms" % (key, et - st))
        use_times.append(et - st)

        reg_tup = [(int(reg_[i * 2] + 0.5), int(reg_[i * 2 + 1] + 0.5)) for i in range(21)]
        # 把点对应回crop_img
        coord = []
        pdb.set_trace()
        for xs, ys in reg_tup:
            coord.append((int((xs - left_size) / scale + 0.5), int((ys - top_size) / scale + 0.5)))


        # for i in range(21):
        #     gt_co = gt_keypoints[i]
        #     pre_co = coord[i]
        #     cv2.circle(crop_img, (int(gt_co[0]+0.5), int(gt_co[1]+0.5)), point_size, (0,255,0), thickness) #G
            # cv2.circle(crop_img, (pre_co[0], pre_co[1]), point_size, point_color, thickness) #R
        # cv2.imshow("crop", crop_img)
        # cv2.waitKey(0)

        if 1:#"right" in images_anno[key]['file_name']:
            img_res = draw_points(crop_img, coord)
            cv2.imwrite("rr.jpg", img_res)
            # img_gt = draw_points(crop_img_gt, gt_keypoints)
            # cv2.imshow("img_res", img_res)
            # cv2.waitKey(1)

            ### 计算pck ####
            pck = [1]*21
            for i in range(21):
                gt_co = gt_keypoints[i]
                pre_co = coord[i]
                dis = math.sqrt((gt_co[0] - pre_co[0]) ** 2 + (gt_co[1] - pre_co[1]) ** 2)
                dis_norm = dis/((bboxs[3] + bboxs[2])/2)
                # print("dis_norm ", str(i), ": ", dis_norm)
                if dis_norm >= 0.1:
                    pck[i] = 0

            pck_np = np.array(pck)
            stat_pck += pck_np
            stat_error[sum(pck)] += 1
            # print(stat_pck)
            # print(stat_error)
            # if sum(pck) < 5:
            #     cv2.imshow("img_res", img_res)
            #     cv2.imshow("img_gt", img_gt)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            #     error_pic_save_path = "F:\GestureKeypointData\error_pic/test_clean_1"
            #     # print(images_anno[key]['file_name'])
            #     cv2.imwrite(os.path.join(error_pic_save_path, images_anno[key]['file_name'].split("/")[-1]), img_gt)
print("正确点数统计（0-21）：", stat_error)
print("21个点每点pck统计：", stat_pck/len(anno['images']))
print("21个点每点正确个数统计：", stat_pck)
print("总pck值：", np.mean(stat_pck/len(anno['images'])))
print("评均耗时：", np.mean(use_times))
if np.mean(stat_pck/len(anno['images'])) > max_pck:
    max_pck = np.mean(stat_pck/len(anno['images']))
print("==========max=============")
print(max_pck)
