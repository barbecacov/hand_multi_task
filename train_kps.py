import torch
import os
import time
import pdb
import random
import numpy as np
from kps_dataset.dataset import kps_dataset
from kps_dataset.dataloader import ImageDataset
from model.reslike_ml import resnet_mt
from utils import AverageMeter, adjust_learning_rate, batch_pck, make_print_to_file
from config import Config
import datetime


best_acc = 0.0

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(config):

    ########################## environment ##########################
    set_random_seeds(999)
    start_epoch = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = kps_dataset(root=config.branch_dir[config.branch])
    train_dataloader = torch.utils.data.DataLoader(
        ImageDataset(dataset=dataset.train, mode="train"),
        batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
        pin_memory=True, drop_last=True, prefetch_factor=4
    )

    test_dataloader = torch.utils.data.DataLoader(
        ImageDataset(dataset=dataset.test, mode="test"),
        batch_size=config.batch_size, shuffle=False, num_workers=config.workers,
        pin_memory=True, drop_last=False, prefetch_factor=4
    )
    ################# data_loader #################

    ################ train config ####################

    model = resnet_mt(config.branch)
    print("training layers :")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    # model = network.creat_model("resnet18_1.0")

    if device == 'cuda':
        criterion = torch.nn.L1Loss(reduction='mean').cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        criterion = torch.nn.L1Loss(reduction='mean')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, eps=1e-8)

    if os.path.exists(config.resume_from):

        state = torch.load(config.resume_from)
        start_epoch = state["epoch"]
        net_state_dict = state["net"]
        optimizer_state_dict = state["optimizer"]
        best_acc = state["acc"]
        if device == "cuda":
            model.module.load_state_dict(net_state_dict, strict=True)
        else:
            model.load_state_dict(net_state_dict, strict=True)

        optimizer.load_state_dict(optimizer_state_dict)
        print("resuming from {} epoch and the best acc is {}".format(start_epoch, best_acc))

    if os.path.exists(config.finetune):
        state = torch.load(config.finetune)
        if "net" in state.keys():
            net_state_dict = state["net"]
        else:
            net_state_dict = state
        if device == "cuda":
            model.module.load_state_dict(net_state_dict, strict=False)
        else:
            model.load_state_dict(net_state_dict, strict=False)
        
        print("finetune with {} checkpoint ...".format(config.finetune))


    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    print("Start training ...")
    for epoch in range(start_epoch, config.max_epoch):
        lr = adjust_learning_rate(optimizer, epoch, config.lr_dec_epoch, config.gamma)
        train(train_dataloader, model, criterion, optimizer, epoch, device)
        if epoch > config.test_epoch:
            test(test_dataloader, model, criterion, optimizer, epoch, device, config)


def train(train_loader, model, criterion, optimizer, epoch, device):
    losses = AverageMeter()

    model.train()
    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        if device == 'cuda':
            input = input.cuda()
            target = target.cuda()

        output = model(input)

        kps_feat = output[0]
        loss = criterion(kps_feat, target)
        losses.update(loss.data.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 100 == 0 and i != 0):
            print('iteration {} | loss: {}, avg loss: {}'
                  .format(i, loss.data.item(), losses.avg))

    end_time = time.time()
    minute = (end_time - start_time) // 60
    second = (end_time - start_time) % 60

    print(
        "epoch: {} | lr : {} | loss: {} | time : {}m{:.3f}s".format(epoch, optimizer.param_groups[0]['lr'], losses.avg,
                                                                    minute, second))


def test(test_loader, model, criterion, optimizer, epoch, device, config):
    global best_acc
    model.eval()
    acc = AverageMeter()

    with torch.no_grad():
        for i, (input, target, img_box, scale, top_size, left_size) in enumerate(test_loader):
            if device == 'cuda':
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            pred = output[0]

            pck = batch_pck(pred, target, [config.h, config.w], img_box, scale, top_size, left_size, device)

            acc.update(pck, input.size(0))

        print("epoch {} | pck: {} | cur-best-pck: {}".format(epoch, acc.avg, best_acc))
        if acc.avg > best_acc:
            print("Saving ...")
            state = {}
            state["acc"] = acc.avg
            state["epoch"] = epoch
            state["optimizer"] = optimizer.state_dict()
            os.makedirs("./checkpoint/{}/{}".format(config.branch, runID), exist_ok=True)
            if device == "cuda":
                state["net"] = model.module.state_dict()
            else:
                state["net"] = model.state_dict()
            torch.save(state, "./checkpoint/{}/{}/model_best.pth".format(config.branch, runID))
            best_acc = acc.avg


if __name__ == '__main__':
    config = Config()
    runID = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    make_print_to_file(path='./log/{}'.format(config.branch), txt_name = runID)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    main(config)
    print(config.__dict__)














