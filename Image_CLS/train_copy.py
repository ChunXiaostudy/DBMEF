# 训练花朵分类模型的脚本
import argparse
import time
import os
import shutil
import pickle
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from datasets.flower_dataset import FlowerDataset
from models import build_model
from util.model_trainer import ModelTrainer
from config.config import cfg
from util.common import setup_seed, setup_logger, show_confMat, plot_line, plot_line_acc5


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, type=float, help='learning rate')
parser.add_argument('--bs', default=None, type=int, help='training batch size')
parser.add_argument('--max_epoch', type=int, default=None, help='number of epoch')
args = parser.parse_args()

cfg.lr0 = args.lr if args.lr else cfg.lr0
cfg.batch_size = args.bs if args.bs else cfg.batch_size
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch


if __name__ == "__main__":
    setup_seed(42)

    logger = setup_logger(cfg.log_path, 'w')
    # 参数配置
    os.makedirs(cfg.output_dir, exist_ok=True)
    transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                ])

    # 实例化dataset(train valid)
    # train_dataset = FlowerDataset(img_dir=cfg.train_dir, transform=cfg.train_transform)
    # # valid
    # valid_dataset = FlowerDataset(img_dir=cfg.valid_dir, transform=cfg.valid_transform)
    # # 组装dataloader
    # train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # valid_loader = DataLoader(valid_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # 实例化网络模型
    model = build_model("resnet18", cfg.num_cls, pretrained=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    
    # 优化器相关
    # loss函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器实例化
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr0, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
    # 学习率下降策略的实例化
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.decay_factor)

    logger.info(
        "cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
            cfg, loss_fn, lr_scheduler, optimizer, model
        )
    )

    # loop
    logger.info("start train...")
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    top5_acc = []
    best_acc, best_epoch = 0, 0
    t_start = time.time()
    for epoch in range(cfg.max_epoch):
        # 一次epoch的训练
        # 按batch形式取数据
        # 前向传播
        # 计算Loss
        # 反向传播计算梯度
        # 更新权重
        # 统计Loss 准确率
        loss_train, acc_train, conf_mat_train, path_error_train = ModelTrainer.train_one_epoch(
            train_loader, model, 
            loss_f=loss_fn, 
            optimizer=optimizer, 
            scheduler=lr_scheduler, 
            epoch_idx=epoch,
            device=device,
            log_interval=cfg.log_interval,
            max_epoch=cfg.max_epoch,
            logger=logger,
        )
        
        # 一次epoch验证
        # 按batch形式取数据
        # 前向传播
        # 计算Loss
        # 统计Loss 准确率
        loss_valid, acc_valid, conf_mat_valid, path_error_valid, acc_top5_valid = ModelTrainer.valid_one_epoch(
            valid_loader, 
            model, loss_fn, 
            device=device,
        )
        
        # 打印训练集和验证集上的指标
        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}". \
                    format(epoch + 1, cfg.max_epoch, acc_train, acc_valid, loss_train, loss_valid,
                           optimizer.param_groups[0]["lr"]))
        logger.info("TOP5ACC:" + str(acc_top5_valid))
        
        # 绘制混淆矩阵
        classes = list(range(cfg.num_cls))
        show_confMat(conf_mat_train, classes, "train", cfg.output_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        show_confMat(conf_mat_valid, classes, "valid", cfg.output_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        top5_acc.append(acc_top5_valid)
        # 绘制Loss和acc曲线
        plt_x = list(range(1, epoch + 2))
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=cfg.output_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=cfg.output_dir)
        plot_line_acc5(plt_x, top5_acc, mode="acc", out_dir=cfg.output_dir)

        # 保存模型
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, f"{cfg.output_dir}/last.pth")

        if best_acc < acc_valid:
            # 保存验证集上表现最好的模型
            best_acc, best_epoch = acc_valid, epoch
            shutil.copy(f"{cfg.output_dir}/last.pth", f"{cfg.output_dir}/best.pth")
    
            # 保存错误图片的路径
            err_imgs_out = os.path.join(cfg.output_dir, "error_imgs_best.pkl")
            error_info = {}
            error_info["train"] = path_error_train
            error_info["valid"] = path_error_valid
            with open(err_imgs_out, 'wb') as f:
                pickle.dump(error_info, f)

    t_use = (time.time() - t_start) / 3600
    logger.info(f"Train done, use time {t_use:.3f} hours, best acc: {best_acc:.3f} in :{best_epoch}")
