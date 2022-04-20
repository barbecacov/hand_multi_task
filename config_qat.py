# -*- coding: utf-8 -*-
"""
@Time ： 2022/4/8 10:19
@Auth ： Zhang Di
@File ：config.py
@IDE ：PyCharm
"""

class Config:
    branch = "kps"

    branch_dir ={
        "kps" : "/media/tcl2/facepro/HandData",
        "fb" : "/media/tcl2/facepro/HandData",
        "isHand": "/media/tcl3/facepro/zm/tcl_gesture_data/isHand_multiCls",
        "multi_cls": "/media/tcl3/facepro/zm/tcl_gesture_data/isHand_multiCls",
        "object": "/media/tcl3/facepro/zm/tcl_gesture_data"
    }
    
    branch_index = {
        "kps" : 0,
        "fb" : 1,
        "isHand" : 3,
        "multi_cls" : 4,
        "object" : 2
    }
    batch_size = 256
    workers = 8
    lr = 2e-4
    gamma = 0.6
    max_epoch = 2
    test_epoch = 0
    gpus = "3"
    h = 160
    w = 160
    # lr_dec_epoch = list(range(10, 200, 10))
    lr_dec_epoch = [5]
    resume_from = ""
    finetune = "./pretrained_model/pytorch_fp32/model_best_0.95408.pth"

