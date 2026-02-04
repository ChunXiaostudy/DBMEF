# 划分花朵数据集
import os
import random
import shutil


def copy_file(img_list, target_dir, setname="train"):
    img_dir = os.path.join(target_dir, setname)
    os.makedirs(img_dir, exist_ok=True)
    for p in img_list:
        shutil.copy(p, img_dir)
    print(f"{setname} dataset: copy {len(img_list)} images to {img_dir}")


if __name__ == "__main__":
    # step1 根据图像路径得到所有图像的列表
    img_dir = r"E:\data\flowers_data\jpg"
    img_list = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
    
    random.seed(10086)
    random.shuffle(img_list)
    # step2 根据比例去获得训练集和验证集的列表  8：2
    train_ratio = 0.8
    valid_ratio = 0.2
    num_img = len(img_list)
    num_train = int(num_img * train_ratio)
    num_valid = num_img - num_train
    train_list = img_list[: num_train]
    valid_list = img_list[num_train: ]
    
    # step3 根据列表把图像复制到新的文件夹下
    target_dir = os.path.abspath(os.path.dirname(img_dir))
    copy_file(train_list, target_dir, "train")
    copy_file(valid_list, target_dir, "valid")
    

"""
train dataset: copy 6551 images to E:\data\flowers_data\train
valid dataset: copy 1638 images to E:\data\flowers_data\valid
"""