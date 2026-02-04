from typing import Any, Tuple
import torch
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt
import torchvision
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
import time
import pickle
import shutil
from torch import nn, optim
from util.model_trainer import ModelTrainer
from config.config import cfg
from models import build_model
from util.common import setup_seed, setup_logger, show_confMat, plot_line, plot_line_acc5
import timm

import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
model1 = timm.create_model('timm/tinynet_d.in1k', pretrained=True, num_classes=1000)
model1.eval()
config = resolve_data_config({}, model=model1)
transform = create_transform(**config)

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, type=float, help='learning rate')
parser.add_argument('--bs', default=None, type=int, help='training batch size')
parser.add_argument('--max_epoch', type=int, default=None, help='number of epoch')
args = parser.parse_args()

cfg.lr0 = args.lr if args.lr else cfg.lr0
cfg.batch_size = args.bs if args.bs else cfg.batch_size
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch


class NewImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(NewImageFolder, self).__init__(root, transform)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target, path
    
    def __len__(self) -> int:
        return super().__len__()


data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


# val_dataset = torchvision.datasets.ImageFolder(
#     root="D:\CV_project_class\data\ImageNet\ILSVRC2012_img_val",
#     transform=data_transform)
# val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)



if __name__ == "__main__":
    val_dataset = NewImageFolder(root="D:\LCX_DATA\ILSVRC2012_img_val_reorder\ILSVRC2012_img_val_reorder",
    transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    setup_seed(42)
    logger = setup_logger(cfg.log_path, 'w')
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    model = model1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 只进行验证，不需要定义优化器
    loss_fn = nn.CrossEntropyLoss()

    # loop
    logger.info("start valid...")
    loss_rec = {"valid": []}
    acc_rec = {"valid": []}
    top5_acc = []
    best_acc, best_epoch = 0, 0
    t_start = time.time()

    loss_valid, acc_valid, conf_mat_valid, path_error_valid, acc_top5_valid = ModelTrainer.valid_one_epoch(
            val_dataloader, 
            model, loss_fn, 
            device=device,
        )
    
    logger.info("Valid Acc:{:.2%}  Valid loss:{:.4f} ". \
                    format(acc_valid,  loss_valid))
    logger.info("TOP5ACC:" + str(acc_top5_valid))

    loss_rec["valid"].append(loss_valid)
    acc_rec["valid"].append(acc_valid)
    top5_acc.append(acc_top5_valid)

    checkpoint = {
            "model": model.state_dict(),
            "epoch": 1,
        }
    torch.save(checkpoint, f"{cfg.output_dir}/last.pth")

    shutil.copy(f"{cfg.output_dir}/last.pth", f"{cfg.output_dir}/best.pth")
    
            # 保存错误图片的路径
    err_imgs_out = os.path.join(cfg.output_dir, "error_imgs_best.pkl")
    error_info = {}
    error_info["valid"] = path_error_valid
    with open(err_imgs_out, 'wb') as f:
        pickle.dump(error_info, f)

    t_use = (time.time() - t_start) / 3600
    logger.info(f"Train done, use time {t_use:.3f} hours")