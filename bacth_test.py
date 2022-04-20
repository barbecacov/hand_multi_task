from model.reslike_ml import resnet_mt
import torch
import os
import numpy as np
import torch.nn as nn
import argparse
import cv2
import math
import time
import pdb


from kps_dataset.dataset import kps_dataset
from kps_dataset.dataloader import ImageDataset
from utils import batch_pck, AverageMeter
from config import Config

parser = argparse.ArgumentParser(description="OKS benchmark")
parser.add_argument("--model-path", type=str, default="pretrained_model/pytorch_fp32/model_best_0.95408.pth")
args = parser.parse_args()

config = Config()

device = "cuda"

dataset = kps_dataset(root="/media/tcl2/facepro/HandData")

test_dataloader = torch.utils.data.DataLoader(
    ImageDataset(dataset=dataset.test, mode="test"),
    batch_size=config.batch_size, shuffle=False, num_workers=config.workers,
    pin_memory=True, drop_last=False
)

model = resnet_mt(config.branch)
check_point = torch.load(args.model_path, map_location="cpu")["net"]
model.load_state_dict(check_point, strict=True)
model.to(device)
model.eval()

pckes = AverageMeter()

with torch.no_grad():
    for i, (input, target, img_box, scale, top_size, left_size) in enumerate(test_dataloader):
        
        if device == "cuda":
            input = input.cuda()
            target = target.cuda()
        output = model(input)
        pred = output[0]
        pck = batch_pck(pred, target, [config.h, config.w], img_box, scale, top_size, left_size, device)
        pckes.update(pck, input.size(0))

print(pckes.avg)