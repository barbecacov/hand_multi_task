# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/7 9:23
@Auth ： Zhang Di
@File ：transforms.py
@IDE ：PyCharm
"""
import cv2
import numpy as np
import math
import random
from PIL import Image, ImageEnhance
import torch
import pdb

set_ratio = 0.5

# branch: fb, isHand, multi_cls, object

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

def init_test_data(img, box, branch):
    if box is None:
        return img
    img_h, img_w = img.shape[0], img.shape[1]
    if branch == "fb":
        x_, y_, w_, h_ = round(box[0]), round(box[1]), round(box[2]), round(box[3])

        x_c, y_c = (2 * x_ + w_) / 2, (2 * y_ + h_) / 2
        new_wh = max(w_, h_)
        alpha = 0.1
    else:
        y1_, x1_, y2_, x2_ = round(box[0] * img_h), round(box[1] * img_w), round(box[2] * img_h), round(box[3] * img_w)

        x_c, y_c = (x1_ + x2_) / 2, (y1_ + y2_) / 2
        new_wh = max(x2_ - x1_, y2_ - y1_)
        alpha = 0.2

    half_length = (alpha + 0.5) * new_wh
    x1_new = round(x_c - half_length)
    x2_new = round(x_c + half_length)
    y1_new = round(y_c - half_length)
    y2_new = round(y_c + half_length)

    x1, y1, x2, y2 = [check_value(x1_new, img_w), check_value(y1_new, img_h),
                      check_value(x2_new, img_w), check_value(y2_new, img_h)]
    img = img[int(y1):int(y2), int(x1):int(x2)]
    return img


def init_data(img, box, branch):
    if box is None:
        return img

    img_h, img_w = img.shape[0], img.shape[1]
    if branch == "fb":
        x_, y_, w_, h_ = round(box[0]), round(box[1]), round(box[2]), round(box[3])

        x_c, y_c = (2 * x_ + w_) / 2, (2 * y_ + h_) / 2
        new_wh = max(w_, h_)
        alpha = 0.1
    else:
        y1_, x1_, y2_, x2_ = round(box[0]*img_h), round(box[1]*img_w), round(box[2]*img_h), round(box[3]*img_w)

        x_c, y_c = (x1_ + x2_) / 2, (y1_ + y2_) / 2
        new_wh = max(x2_ - x1_, y2_ - y1_)
        alpha = 0.2

    random_half_length = (random.uniform(alpha, alpha+0.2) + 0.5) * new_wh
    x1_new = x_c - random.uniform(0.8, 1.1) * random_half_length
    x2_new = x_c + random.uniform(0.8, 1.1) * random_half_length
    y1_new = y_c - random.uniform(0.8, 1.1) * random_half_length
    y2_new = y_c + random.uniform(0.8, 1.1) * random_half_length

    x1, y1, x2, y2 = [check_value(x1_new, img_w), check_value(y1_new, img_h),
                      check_value(x2_new, img_w), check_value(y2_new, img_h)]

    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
    img = img[y1:y2, x1:x2]
    return img

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

    def __call__(self, image):
        # aug_color
        image = self.color_aug(image)
        image = self.random_rotate(image)
        image = self.random_flip(image)
        height, width = image.shape[0], image.shape[1]
        if height > 70 and width > 70:
            image = self.fuzzy_process(image)
        return image


class color_aug(object):
    def __call__(self, image):
        image = RandomRGB2YUV2RGB(image)
        beta = random.randint(0, 3)
        if beta == 0:
            image = RandomBrightness(image)
            image = RandomColor(image)
            image = RandomContrast(image)
            image = RandomSharpness(image)
        if beta == 1:
            image = RandomContrast(image)
            image = RandomSharpness(image)
            image = RandomColor(image)
            image = RandomBrightness(image)
        if beta == 2:
            image = RandomColor(image)
            image = RandomBrightness(image)
            image = RandomContrast(image)
            image = RandomSharpness(image)

        return image


class random_rotate(object):
    def __init__(self, max_degree=20):
        self.max_degree = max_degree

    def __call__(self, image):
        height, width = image.shape[0], image.shape[1]
        degree = random.uniform(-self.max_degree, self.max_degree)

        img_center = (int(width / 2), int(height / 2))
        rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)

        image = cv2.warpAffine(image, rotateMat, (width, height))

        return image


class random_flip(object):
    def __call__(self, image):
        if random.randint(0, 2):
            image = cv2.flip(image, 1)
        return image


class fuzzy_process(object):
    def __call__(self, image):
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
        return image


class padding_resize(object):
    def __init__(self, input_h, input_w):
        self.h = input_h
        self.w = input_w

    def __call__(self, image):

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
        assert image.shape[0] == self.h
        return image.astype(np.float32)


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    # mean = [] 
    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor


class val_resize(object):
    def __init__(self, input_h, input_w):
        self.h = input_h
        self.w = input_w

    def __call__(self, image):
        image = cv2.resize(image, (self.w, self.h))
        return image.astype(np.float32)


class ToTensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).div(255.)

        return image

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    # mean = [] 
    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor


# class Compose(object):
#     """Composes several transforms together.
#     """

#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, *args):
#         for t in self.transforms:
#             args = t(*args)
#         return args



