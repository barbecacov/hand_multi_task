# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/7 9:23
@Auth ： Zhang Di
@File ：transforms.py
@IDE ：PyCharm
"""
import cv2
from torchvision import transforms
import numpy as np
import math
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import torch
import pdb
set_ratio = 0.5


def check_value(value, length):
    if 0 <= value <= length:
        return value
    elif value < 0:
        return 0
    elif value > length:
        return length

def RandomRGB2YUV2RGB(image):
    if random.randint(0,2):
        image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
        image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
        return image
    return image


def RandomColor(image,lower=0.4, upper=1.6):
    if random.randint(0,2):
        image=Image.fromarray(image)
        alpha = random.uniform(lower, upper)
        image = ImageEnhance.Color(image).enhance(alpha)
        image = np.asarray(image)
        return image
    return image


def RandomSharpness(image, lower=0.4, upper=1.6):
    if random.randint(0,2):
        image=Image.fromarray(image)
        alpha = random.uniform(lower, upper)
        image = ImageEnhance.Sharpness(image).enhance(alpha)
        image = np.asarray(image)
        return image
    return image


def RandomBrightness(image, lower=0.4, upper=1.6):
    if random.randint(0,2):
        image=Image.fromarray(image)
        alpha = random.uniform(lower, upper)
        image = ImageEnhance.Brightness(image).enhance(alpha)
        image = np.asarray(image)
        return image
    return image


def RandomContrast(image, lower=0.4, upper=1.6):
    if random.randint(0,2):
        image = Image.fromarray(image)
        alpha = random.uniform(lower, upper)
        image = ImageEnhance.Contrast(image).enhance(alpha)
        image = np.asarray(image)
        return image
    return image


def init_test_data(img, kps, box, img_size):
    img_h, img_w = img_size[0], img_size[1]

    x_, y_, w_, h_ = round(box[0]), round(box[1]), round(box[2]), round(box[3])
    x_c, y_c = (2 * x_ + w_) / 2, (2 * y_ + h_) / 2
    new_wh = max(w_, h_)
    alpha = 0.1
    half_length = (alpha + 0.5) * new_wh
    x1_ = round(x_c - half_length)
    x2_ = round(x_c + half_length)
    y1_ = round(y_c - half_length)
    y2_ = round(y_c + half_length)

    x1, y1, x2, y2 = [check_value(x1_, img_w), check_value(y1_, img_h),
                      check_value(x2_, img_w), check_value(y2_, img_h)]

    box = np.array([x_ - x1, y_ - y1, x_ + w_ - x1, y_ + h_ - y1]).astype(np.float32)
    
    img = img[int(y1):int(y2), int(x1):int(x2)]

    kp = np.array(kps)
    kpts = []
    ys = np.array(kp[1::3])
    xs = np.array(kp[0::3])
    for i in range(21):
        if y1<ys[i]<y2 and x1 < xs[i] < x2:
            kpts.append((round(xs[i] - x1), round(ys[i] - y1)))
        else:
            kpts.append((-1, -1))
    kpts = np.array(kpts).astype(np.float32)
    return img, kpts, box


def init_data(img, kps, box, img_size):
    img_h, img_w = img_size[0], img_size[1]

    x_, y_, w_, h_ = round(box[0]), round(box[1]), round(box[2]), round(box[3])
    x_c, y_c = (2 * x_ + w_) / 2, (2 * y_ + h_) / 2
    new_wh = max(w_, h_)
    random_half_length = (random.uniform(0.1, 0.3) + 0.5) * new_wh
    x1_ = x_c - random.uniform(0.8, 1.1) * random_half_length
    x2_ = x_c + random.uniform(0.8, 1.1) * random_half_length
    y1_ = y_c - random.uniform(0.8, 1.1) * random_half_length
    y2_ = y_c + random.uniform(0.8, 1.1) * random_half_length

    x1, y1, x2, y2 = [check_value(x1_, img_w), check_value(y1_, img_h),
                      check_value(x2_, img_w), check_value(y2_, img_h)]

    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    box = np.array([x_ - x1, y_ - y1, x_ + w_ - x1, y_ + h_ - y1]).astype(np.float32)

    img = img[y1:y2, x1:x2]

    kp = np.array(kps)
    kpts = []
    ys = np.array(kp[1::3]) - y1
    xs = np.array(kp[0::3]) - x1
    for i in range(21):
        kpts.append([xs[i], ys[i]])
    kpts = np.array(kpts).astype(np.float32)
    return img, kpts, box


def motion_blur(image):
    degree_list = [6, 8, 10, 12, 14]
    idx = random.randint(0, 4)
    degree = degree_list[idx]

    angle = random.randint(1, 360)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


class texture_aug(object):

    def __init__(self):

        self.color_aug = color_aug()
        self.random_rotate = random_rotate()
        self.random_flip = random_flip()
        self.fuzzy_process = fuzzy_process()

    def __call__(self, image, kpts, box):
        # aug_color
        image, kpts, box = self.color_aug(image, kpts, box)
        image, kpts, box = self.random_rotate(image, kpts, box)
        image, kpts, box = self.random_flip(image, kpts, box)
        if box[2] - box[0] > 70 and box[3] - box[1] > 70:
            image, kpts, box = self.fuzzy_process(image, kpts, box)
        return image, kpts, box


class color_aug(object):
    def __call__(self, image, kpts, box):
        image = RandomRGB2YUV2RGB(image)
        beta = random.randint(0, 3)
        if beta == 0:
            image = RandomBrightness(image)
            image = RandomColor(image)
            image = RandomContrast(image)
            image = RandomSharpness(image)
        elif beta == 1:
            image = RandomContrast(image)
            image = RandomSharpness(image)
            image = RandomColor(image)
            image = RandomBrightness(image)
        elif beta == 2:
            image = RandomColor(image)
            image = RandomBrightness(image)
            image = RandomContrast(image)
            image = RandomSharpness(image)

        return image, kpts, box


class random_rotate(object):
    def __init__(self, max_degree=20):
        self.max_degree = max_degree

    def __call__(self, image, kpts, box):
        height, width = image.shape[0], image.shape[1]
        kpts_flag = True
        b_flag = True

        degree = random.uniform(-self.max_degree, self.max_degree)

        img_center = (int(width / 2), int(height/2))
        rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)

        rot_img = cv2.warpAffine(image, rotateMat, (width, height))

        new_joint_list = []

        p1 = np.array([box[0], box[1], 1])
        p2 = np.array([box[0], box[3], 1])
        p3 = np.array([box[2], box[1], 1])
        p4 = np.array([box[2], box[3], 1])

        p1_x, p1_y = rotateMat.dot(p1)
        p2_x, p2_y = rotateMat.dot(p2)
        p3_x, p3_y = rotateMat.dot(p3)
        p4_x, p4_y = rotateMat.dot(p4)

        x1 = min(p1_x, p2_x, p3_x, p4_x)
        x2 = max(p1_x, p2_x, p3_x, p4_x)
        y1 = min(p1_y, p2_y, p3_y, p4_y)
        y2 = max(p1_y, p2_y, p3_y, p4_y)

        if x1 < 0 or y1 < 0 or x1 > width or y1 > height:
            b_flag = False
        if x2 < 0 or y2 < 0 or x2 > width or y2 > height:
            b_flag = False

        for idx in range(kpts.shape[0]):
            point = kpts[idx]

            p = np.array([point[0], point[1], 1])
            new_x, new_y = rotateMat.dot(p)
            new_joint_list.append([new_x, new_y])

            if new_x < 0 or new_y < 0 or new_x > width or new_y > height:
                kpts_flag = False


        if b_flag:
            box = np.array([x1, y1, x2, y2]).astype(np.float32)
            image = rot_img
            if kpts_flag:
                kpts = np.array(new_joint_list).astype(np.float32)

        return image, kpts, box


class random_flip(object):

    def __call__(self, image, kpts, box):
        new_joint_list = []
        height, width = image.shape[0], image.shape[1]
        if random.randint(0, 2):
            image = cv2.flip(image, 1)
            x1, y1, x2, y2 = box
            box = np.array([width - x2, y1, width - x1, y2]).astype(np.float32)
            for idx in range(kpts.shape[0]):
                point = kpts[idx]
                new_joint_list.append([width - point[0], point[1]])
            kpts = np.array(new_joint_list).astype(np.float32)

        return image, kpts, box


class fuzzy_process(object):
    def __call__(self, image, kpts, box):
        case_id = random.randint(0, 10)
        height, width = image.shape[0], image.shape[1]
        if case_id == 0:
            # 高斯滤波
            k_size = [3, 5, 7, 9]
            idx = random.randint(0, 3)
            image = cv2.GaussianBlur(image, ksize=(k_size[idx], k_size[idx]), sigmaX=0, sigmaY=0)
        elif case_id in [1, 4, 5, 6, 7]:
            # 先resize到小尺寸，再resize到
            re_size = [60, 70, 80, 90, 100]
            re_interpolation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC,
                                cv2.INTER_LANCZOS4]
            idx = random.randint(0, len(re_size) - 1)
            interpolation_idx_1 = random.randint(0, len(re_interpolation) - 1)
            # print(re_size[idx], re_interpolation[interpolation_idx_1])
            image = cv2.resize(image, (re_size[idx], re_size[idx]),
                               interpolation=re_interpolation[interpolation_idx_1])
            interpolation_idx_2 = random.randint(0, len(re_interpolation) - 1)
            image = cv2.resize(image, (width, height), interpolation=re_interpolation[interpolation_idx_2])
        elif case_id == 2 or case_id == 3:
            # print("motion_blur prepocessing!!!!!!")
            image = motion_blur(image)
        else:
            pass
        return image, kpts, box


class padding_resize(object):
    def __init__(self, input_h, input_w):
        self.h = input_h
        self.w = input_w

    def __call__(self, image, kpts, box):

        img_h, img_w = image.shape[0], image.shape[1]
        scale = min(self.h / img_h, self.w / img_w)
        image = cv2.resize(image, (round(img_w * scale), round(img_h * scale)))

        if self.h > image.shape[0]:
            top_size = int((self.h - image.shape[0]) / 2)
            bottom_size = int(self.h - top_size - image.shape[0])
            left_size = 0
            right_size = 0
        elif self.w > image.shape[1]:
            left_size = int((self.w - image.shape[1]) / 2)
            right_size = int(self.w - left_size - image.shape[1])
            top_size = 0
            bottom_size = 0
        else:
            left_size = 0
            right_size = 0
            top_size = 0
            bottom_size = 0

        image = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size,
                                          borderType=cv2.BORDER_CONSTANT, value=0)

        kpts[:, 0] = kpts[:, 0] * scale + left_size
        kpts[:, 1] = kpts[:, 1] * scale + top_size

        box[0] = box[0] * scale + left_size
        box[1] = box[1] * scale + top_size
        box[2] = box[2] * scale + left_size
        box[3] = box[3] * scale + top_size
        assert image.shape[0] == self.h
        return image.astype(np.float32), kpts, scale, top_size, left_size

class val_resize(object):
    def __init__(self, input_h, input_w):
        self.h = input_h
        self.w = input_w

    def __call__(self, image, kpts, box):
        height, width = image.shape[0], image.shape[1]
        scale_h, scale_w = self.h / height, self.w / width

        image = cv2.resize(image, (self.w, self.h))
        kpts[:, 0] = kpts[:, 0] * scale_w
        kpts[:, 1] = kpts[:, 1] * scale_h

        box[0] = box[0] * scale_w
        box[1] = box[1] * scale_h
        box[2] = box[2] * scale_w
        box[3] = box[3] * scale_h

        return image.astype(np.float32), kpts, box


class ToTensor(object):

    def __call__(self, image, kpts):
        width = image.shape[1]
        height = image.shape[0]
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).div(255.)
        kpts[:, 0] = kpts[:, 0] / width
        kpts[:, 1] = kpts[:, 1] / height
        # pdb.set_trace()
        kpts = torch.from_numpy(kpts).flatten()
        return image, kpts

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]), 
    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std

    def __call__(self, tensor, kpts):
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor, kpts


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args



