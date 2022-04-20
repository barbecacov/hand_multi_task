import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
# from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet
from model.reslike_ml import resnet_mt
import pdb
import numpy as np
import random

from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer
from tinynn.util.train_util import DLContext, get_device, train
# from tinynn.util.cifar10 import get_dataloader, train_one_epoch, validate

from kps_dataset.dataset import kps_dataset
from kps_dataset.dataloader import ImageDataset
from utils import AverageMeter, adjust_learning_rate, batch_pck
from config_qat import Config


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def train_one_epoch(model, context: DLContext):
    avg_losses = AverageMeter()

    model.to(device=context.device)
    model.train()

    end = time.time()
    for i, (image, label) in enumerate(context.train_loader):

        if context.max_iteration is not None and context.iteration >= context.max_iteration:
            break

        image = image.to(device=context.device)

        output = model(image)

        label = label.to(device=context.device)
        # pdb.set_trace()
        loss = context.criterion(output[config.branch_index[config.branch]], label)
        context.optimizer.zero_grad()
        loss.backward()
        context.optimizer.step()
        avg_losses.update(loss.item(), image.size(0))
        if i % context.print_freq == 0:
            current_lr = 0.0
            for param_group in context.optimizer.param_groups:
                current_lr = param_group['lr']
                break
            print(
                f'Epoch:{context.epoch}\t'
                f'Iter:[{i}|{len(context.train_loader)}]\t'
                f'Lr:{current_lr:.5f}\t'

                f'Loss:{avg_losses.avg:.5f}\t'
            )

        if context.warmup_scheduler is not None and context.warmup_iteration > context.iteration:
            context.warmup_scheduler.step()

        context.iteration += 1

    if context.scheduler and context.warmup_iteration <= context.iteration:
        context.scheduler.step()


def validate(model, context: DLContext) -> float:
    model.to(device=context.device)
    model.eval()
    avg_acc = AverageMeter()
    with torch.no_grad():
        for i, (input, target, img_box, scale, top_size, left_size) in enumerate(context.val_loader):
            image = input.to(device=context.device)
            label = target.to(device=context.device)

            output = model(image)
            pck = batch_pck(output[config.branch_index[config.branch]], label, [config.h, config.w], img_box, scale, top_size, left_size, context.device)
            avg_acc.update(pck, image.size(0))
            # measure elapsed time
        print(f'Validation Acc@1 {avg_acc.avg:.5f}')
    return avg_acc.avg


def main(config):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seeds(999)
    dataset = kps_dataset(root=config.branch_dir[config.branch])

    train_dataloader = torch.utils.data.DataLoader(
        ImageDataset(dataset=dataset.train[3000:5000], mode="train"),
        batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
        pin_memory=True, drop_last=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        ImageDataset(dataset=dataset.test, mode="test"),
        batch_size=config.batch_size, shuffle=False, num_workers=config.workers,
        pin_memory=True, drop_last=False
    )

    with model_tracer():
        model = resnet_mt(config.branch)
        model.load_state_dict(torch.load(config.finetune)["net"], strict=True)

        dummy_input = torch.rand((1, 3, 160, 160))

        quantizer = QATQuantizer(model, dummy_input, work_dir='out',
                                 config={'per_tensor': True, 'asymmetric': True, 'disable_requantization_for_cat': True, 'backend': 'qnnpack'})
        qat_model = quantizer.quantize()
        pdb.set_trace()
    # print(qat_model)

    # Use DataParallel to speed up training when possible
    # if torch.cuda.device_count() > 1:
    #     qat_model = nn.DataParallel(qat_model)

    # if device == "cuda":
    #     qat_model = torch.nn.DataParallel(qat_model)

    # Move model to the appropriate device
    device = get_device()
    qat_model.to(device)

    context = DLContext()
    context.device = device
    context.train_loader = train_dataloader
    context.val_loader = test_dataloader
    context.max_epoch = config.max_epoch
    context.criterion = nn.L1Loss(reduction="mean")
    context.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, qat_model.parameters()), lr=config.lr, momentum=0.9, weight_decay=5e-4)
    context.scheduler = optim.lr_scheduler.CosineAnnealingLR(context.optimizer, T_max=context.max_epoch + 1, eta_min=0)

    # Quantization-aware training
    train(qat_model, context, train_one_epoch, validate, qat=True)

    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        qat_model = torch.quantization.convert(qat_model)

        # script_module = torch.jit.trace(qat_model, dummy_input, strict=False)

        # torch.jit.save(script_module, "model.pt")

        torch.save(qat_model.state_dict(), "./out/reslike_quant.pth")

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = 'qnnpack'


        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with a specific data type (e.g. int8)
        # you may specify `quantize_target_type='int8'` in the following line.
        # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
        # you may specify `strict_symmetric_check=True` in the following line.
        converter = TFLiteConverter(qat_model, dummy_input, tflite_path='out/qat_model.tflite',
                                    quantize_target_type='uint8', fuse_quant_dequant=True)
        converter.convert()


if __name__ == '__main__':
    config = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    main(config)