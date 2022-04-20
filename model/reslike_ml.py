# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/2 16:27
@Auth ： Zhang Di
@File ：reslike_ml.py
@IDE ：PyCharm
"""
import pdb

import torch
import torch.nn as nn

from model.resnet.resnet import resnet50


def conv1x1(in_planes, out_planes):
    """1x1 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                         nn.ReLU6(True),
                         )


class MultiTask(nn.Module):
    def __init__(self, branch):
        # kps, fb, object, isHand, multi_cls, box
        super(MultiTask, self).__init__()
        self.branch = branch
        self.branch_pool = ["backbone", "kps", "fb", "object", "isHand", "multi_cls"]
        self.backbone = resnet50()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.7)
        # self.relu = nn.ReLU6(True)

        # kps head
        self.kps_fc = nn.Linear(768, 42)
        self.act = nn.Sigmoid()

        # fb head
        self.fb_conv1 = conv1x1(512, 64)
        self.fb_conv2 = conv1x1(64, 64)
        self.fb_cls = conv1x1(64, 2)

        # withObject head
        self.object_conv1 = conv1x1(512, 64)
        self.object_conv2 = conv1x1(64, 64)
        self.object_cls = conv1x1(64, 2)

        # isHand head
        self.isHand_conv1 = conv1x1(512, 64)
        self.isHand_conv2 = conv1x1(64, 64)
        self.isHand_cls = conv1x1(64, 2)

        # multiCls head
        self.multi_cls_conv1 = conv1x1(512, 64)
        self.multi_cls_conv2 = conv1x1(64, 64)
        self.multi_cls_cls = conv1x1(64, 7)

        # box head
        # self.box_conv1 = conv1x1(512, 64)
        # self.box_cls = conv1x1(64, 4)

        self._apple_freeze_branch(branch=self.branch)

    def forward(self, x):
        bs = x.shape[0]
        x2, x3 = self.backbone(x)
        o3 = x3
        x2 = torch.mean(x2, dim=[2, 3])
        x3 = torch.mean(x3, dim=[2, 3])

        kps_p = torch.cat((x2, x3), dim=1)
        kps_feat = self.act(self.kps_fc(kps_p))

        fb_feat = self._make_head(o3, self.fb_conv1, self.fb_conv2, self.fb_cls)
        withObject_feat = self._make_head(o3, self.object_conv1, self.object_conv2, self.object_cls)
        isHand_feat = self._make_head(o3, self.isHand_conv1, self.isHand_conv2, self.isHand_cls)
        multiCls_feat = self._make_head(o3, self.multi_cls_conv1, self.multi_cls_conv2, self.multi_cls_cls)

        # box_feat = self.box_conv1(o3)
        # box_feat = self.avg_pool(box_feat)
        # box_feat = self.dropout(box_feat)
        # box_feat = self.box_cls(box_feat).view(bs, -1)

        return kps_feat, fb_feat, withObject_feat, isHand_feat, multiCls_feat
        # return kps_feat

    def _make_head(self, x, conv1, conv2, cls):
        bs = x.shape[0]
        feat = conv1(x)
        feat = feat + conv2(feat)
        feat = self.avg_pool(feat)
        feat = self.dropout(feat)
        feat = cls(feat)
        feat = feat.view(bs, -1)
        return feat

    def _apple_freeze_branch(self, branch):

        if branch == "kps":
            train_branch = ["backbone", "kps"]
        else:
            train_branch = [branch]
        freeze_branch = list(set(self.branch_pool).difference(set(train_branch)))

        for k, v in self.named_parameters():
            if self._in_pool(k, freeze_branch):
                v.requires_grad=False

    def _in_pool(self, k, freeze_branch):
        for branch in freeze_branch:
            if branch in k:
                return True
        return False

def resnet_mt(branch):
    mt_model = MultiTask(branch)
    return mt_model

if __name__ == '__main__':
    x = torch.randn(1, 3, 160, 160)
    model = MultiTask("kps")
    torch.save(model.state_dict(), "raw_model.pth")
    out = model(x)





