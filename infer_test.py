# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/18 13:43
@Auth ： Zhang Di
@File ：test_tflite.py
@IDE ：PyCharm
"""

import tensorflow as tf
from config import Config
from kps_dataset.dataset import kps_dataset
from kps_dataset.transforms import init_test_data, Compose, padding_resize, ToTensor, Normalize
import torch
import numpy as np
from model.reslike_ml import resnet_mt
from kps_dataset.data_utils import draw_points
import pdb
import cv2


def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def torch_infer(input_tensor, model_path):  # n, c, h, w
    state_dict = torch.load(model_path)["net"]
    model = resnet_mt(config.branch)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        out = model(input_tensor)
    print(out[0])
    return out[0]


def tflite_infer(input_tensor, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    in_scale, in_zero_point = input_details[0]['quantization']
    scale, zero_point = output_details[0]['quantization']
    input_tensor = (input_tensor / in_scale + in_zero_point).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = output_data.astype(np.float32)
    output_data = (output_data - zero_point) * scale
    print(output_data[0])
    return output_data[0]


def post_points(output_tensor, scale, left_size, top_size):
    output_tensor *= config.h
    output_tensor = output_tensor.reshape(-1, 2)
    output_tensor[:, 0] = ((output_tensor[:, 0] - left_size) / scale).round()
    output_tensor[:, 1] = ((output_tensor[:, 1] - top_size) / scale).round()
    return output_tensor


config = Config()
dataset = kps_dataset(root=config.branch_dir[config.branch])
test_dataset = dataset.test

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

pad_reize = padding_resize(160., 160.)
to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

for img_path, img_size, img_box, img_kps in test_dataset:
    img = read_image(img_path)
    img, kpts, box = init_test_data(img, img_kps, img_box, img_size)

    draw_image = img
    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR)

    image, kpts, scale, top_size, left_size = pad_reize(img, kpts, box)
    ### for torch
    image, kpts = to_tensor(image, kpts)
    image, kpts = normalize(image, kpts)
    image_tensor_torch = torch.unsqueeze(image, 0)

    image_tensor_tflite = image_tensor_torch.permute(0, 2, 3, 1).numpy()

    pred_tflite = tflite_infer(image_tensor_tflite, "./out/qat_model.tflite")

    pred_torch = torch_infer(image_tensor_torch, "pretrained_model/pytorch_fp32/model_best_0.95408.pth")
    pdb.set_trace()
    coord = post_points(pred_torch.numpy(), scale, left_size, top_size)
    coord_tflite = post_points(pred_tflite, scale, left_size, top_size)
    # print(coord)
    # print(coord_tflite)
    res = draw_points(draw_image, coord_tflite)
    cv2.imwrite("torch.jpg", res)
    pdb.set_trace()










