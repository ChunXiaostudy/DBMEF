import os
import logging
from matplotlib import pyplot as plt
import random
import numpy as np
import torch


def setup_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 卷积算法固定
        torch.backends.cudnn.benchmark = True      # 网络结构变化不大时使训练加速，为每个卷积层搜索最适合算法


def setup_logger(log_path, mode='w'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 配置文件Handler
    file_handler = logging.FileHandler(log_path, mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 配置屏幕Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def show_confMat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, figsize=None, perc=False):
    """绘制混淆矩阵并保存图片

    Args:
        confusion_mat (np.ndarray): 混淆矩阵二维数组
        classes (list): 类别名称
        set_name (str): 数据集名称 train or valid or test
        out_dir (str): 图片保存的文件夹
        epoch (int, optional): 第几个epoch. Defaults to 999.
        verbose (bool, optional): 是否打印详细信息. Defaults to False.
        figsize (optional): 绘制的图像大小. Defaults to None.
        perc (bool, optional): 是否采用百分比，图像分割时用，因分类数目过大. Defaults to False.
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()  # 按召回率

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num-10]
    plt.figure(figsize=(int(figsize), int(figsize*1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_tmp, cmap=cmap)
    plt.colorbar(fraction=0.03)

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 打印数字
    if perc:
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va='center', ha='center', color='red',
                         fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()

    if verbose:
        print(set_name)
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),  # 
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """绘制训练和验证集的loss曲线/acc曲线

    Args:
        train_x (list): x轴
        train_y (list): y轴
        valid_x (list): x轴
        valid_y (list): y轴
        mode (str): 'loss' or 'acc'
        out_dir (str): 图片保存的文件夹
    """

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title(str(mode))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()


def plot_line_acc5(valid_x, valid_y, mode, out_dir):
    """绘制训练和验证集的loss曲线/acc曲线

    Args:
        train_x (list): x轴
        train_y (list): y轴
        valid_x (list): x轴
        valid_y (list): y轴
        mode (str): 'loss' or 'acc'
        out_dir (str): 图片保存的文件夹
    """

    # plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title(str(mode))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()