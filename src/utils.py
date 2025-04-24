import os
import random
import time

import torch
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import _LRScheduler
# from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from PIL import Image
import csv

def stat_model_param(model, name=None):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if name is None:
        logger.info(f"total: {total_num:,}, trainable:{trainable_num:,}, frozen: {(total_num - trainable_num):,}")
    else:
        logger.info(f"[{name}] total: {total_num:,}, trainable:{trainable_num:,}, frozen: {(total_num - trainable_num):,}")
    return {'Total': total_num, 'Trainable': trainable_num}

def frozen_model(model):
    for param in model.parameters():
        param.requires_grad = False # Frozen 
    stat_model_param(model)
    
def activate_model(model):
    for param in model.parameters():
        param.requires_grad = True # trainable 
    stat_model_param(model)
    
def save_model(model, path):
    torch.save(model.state_dict(),path)
    logger.info(f"save {path} model parameters done")

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"load {path} model parameters done")
        
def setup_seed(seed):
    # https://zhuanlan.zhihu.com/p/462570775
    torch.use_deterministic_algorithms(True) # 检查pytorch中有哪些不确定性
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 大于CUDA 10.2 需要设置
    logger.info("seed: %d, random:%.4f, torch random:%.4f, np random:%.4f" %(seed, random.random(), torch.rand(1), np.random.rand(1)))
    
# def tensor_to_img(img_tensors):
#     imgs = []
#     toImg = transforms.ToPILImage()
#     for tensor in img_tensors:
#         img = toImg(tensor)
#         imgs.append(img)
#     return imgs

def img_norm(imgs):
    sigmoid = nn.Sigmoid()
    # compress [-1,1]  into  [0,1]
    imgs_norm =  sigmoid(imgs)
    return imgs_norm
 

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"create dir: {path}")
    else:
        logger.info(f"dir exists, {path}")