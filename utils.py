# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/7 17:05
@Auth ： Zhang Di
@File ：utils.py
@IDE ：PyCharm
"""
import numpy as np
import copy
import torch
import pdb
import os
import sys
import datetime
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.state_dict()['param_groups'][0]['lr']


def batch_pck(pred, target, input_size, img_box, scale, top_size, left_size, device):

    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1, 2)
    pred = pred * input_size[0]
    target = target.view(batch_size, -1, 2)
    target = target * input_size[0]

    # if device == "cuda":
    scale = scale.to(device)
    top_size = top_size.to(device)
    left_size = left_size.to(device)
    img_box = img_box.to(device)

    scale = scale.reshape(batch_size, 1)
    scale = scale.repeat(1, 21)
    top_size = top_size.reshape(batch_size, 1)
    top_size = top_size.repeat(1, 21)

    left_size = left_size.reshape(batch_size, 1)
    left_size = left_size.repeat(1, 21)

    pred[:, :, 0] = ((pred[:, :, 0] - left_size) / scale).round()
    pred[:, :, 1] = ((pred[:, :, 1] - top_size) / scale).round()

    target[:, :, 0] = ((target[:, :, 0] - left_size) / scale).round()
    target[:, :, 1] = ((target[:, :, 1] - top_size) / scale).round()

    diff = torch.pow(pred - target, 2)
    diff = diff.sum(-1).sqrt()

    box_size = torch.mean(img_box[:, 2:4], dim=1, keepdim=True).repeat(1, 21)
    diff = diff / box_size
    pck = torch.where(diff >= 0.1, 0, 1)

    pck = pck.sum() / pck.numel()
    return pck.data.item()


def batch_scale_pad(input_size, crop_size):
    h, w = input_size[0], input_size[1]
    scale = copy.deepcopy(crop_size)
    re_hw = copy.deepcopy(crop_size)
    scale[:, 0] = h / crop_size[:, 0]
    scale[:, 1] = w / crop_size[:, 1]
    scale = torch.min(scale, dim=1)[0]
    re_hw[:, 0] = (scale * crop_size[:, 0]).round()
    re_hw[:, 1] = (scale * crop_size[:, 1]).round()

    top_left = torch.zeros(size=crop_size.shape)

    top_left[:, 0][re_hw[:, 0] < h] = ((h - re_hw[:, 0]) / 2.)[re_hw[:, 0] < h].round()
    top_left[:, 1][re_hw[:, 1] < w] = ((h - re_hw[:, 1]) / 2.)[re_hw[:, 1] < w].round()

    return scale, top_left

def accuracy(output, target):
    """Computes the precision@k for the specified values of k
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        correct_k = correct[0].view(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size)
        return res


def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, name + ".txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def make_print_to_file(path='./', txt_name=""):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''


    class Logger(object):
        def __init__(self, filename="log.txt", path="./"):
            # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            self.terminal = sys.stdout
            os.makedirs(path, exist_ok=True)
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = txt_name
    sys.stdout = Logger(fileName + '.txt', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))