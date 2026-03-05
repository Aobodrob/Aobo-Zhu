import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

"""
Training script for U-Net based Semantic Segmentation.
For detailed dataset preparation (VOC format), please refer to the README.
"""

if __name__ == "__main__":
    # --- 1. Basic Configurations ---
    device_id = "0"  # 指定显卡 ID
    seed = 11  # 随机种子
    num_classes = 2  # 类别数 (目标类 + 背景)
    backbone = "vgg"  # 主干网络: vgg / resnet50
    input_shape = [512, 512]  # 输入尺寸

    # --- 2. Training Hyperparameters ---
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 2
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 2
    Freeze_Train = True

    Init_lr = 1e-4
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'

    # --- 3. Paths & Logging (De-sensitized) ---
    model_path = ""  # 预训练权重路径 (如有)
    dataset_path = 'datasets/synthetic dataset/IR_phase'  # 数据集根目录
    save_dir = 'checkpoints'  # 权重与日志保存目录
    save_period = 10  # 保存频率
    eval_flag = True  # 训练中开启验证
    eval_period = 10

    # --- 4. Advanced Settings ---
    Cuda = True
    distributed = False  # 是否分布式训练 (DDP)
    sync_bn = False  # 是否开启 SyncBatchNorm
    fp16 = False  # 是否混合精度
    num_workers = 4  # 数据读取线程数

    # 初始化
    # os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()

    # --- 5. Device Setup ---
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # --- 6. Model Initialization ---
    model = Unet(num_classes=num_classes, pretrained=False, backbone=backbone).train()
    if model_path == '':
        weights_init(model)
    else:
        # 这里的加载逻辑可以保留，但路径已改为 generic
        if local_rank == 0: print(f'Loading weights from: {model_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # --- 7. Loss and Callbacks ---
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "run_" + time_str)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    model_train = model.train()

    if Cuda:
        if distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank])
        else:
            model_train = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True

    # --- 8. Dataloader Preparation ---
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    num_train, num_val = len(train_lines), len(val_lines)

    if local_rank == 0:
        show_config(num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape)

    # --- 9. Training Loop ---
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 冻结/解冻逻辑切换
        if epoch >= Freeze_Epoch and Freeze_Train:
            model.unfreeze_backbone()
            Freeze_Train = False  # 防止重复触发

        batch_size = Freeze_batch_size if epoch < Freeze_Epoch else Unfreeze_batch_size

        # 自动调整学习率与优化器
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = Init_lr_fit * 0.01

        optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                               weight_decay=weight_decay) if optimizer_type == "adam" \
            else optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True)

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 构建 DataLoader
        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, dataset_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, dataset_path)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)

        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, dataset_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                      num_train // batch_size, num_val // batch_size, gen, gen_val, UnFreeze_Epoch,
                      Cuda, True, False, np.ones([num_classes], np.float32), num_classes, fp16, scaler, save_period,
                      save_dir, local_rank)

    if local_rank == 0:
        loss_history.writer.close()