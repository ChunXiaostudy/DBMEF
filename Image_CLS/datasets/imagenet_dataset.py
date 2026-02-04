"""
将imagenet数据集加载到pytorch中
"""
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from collections import defaultdict, OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt


data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
# train_dataset = torchvision.datasets.ImageFolder(
#     root='./ImageNet/data/ImageNet2012/ILSVRC2012_img_train',
#     transform=data_transform)
# train_dataset_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

val_dataset = torchvision.datasets.ImageFolder(
    root="D:\CV_project_class\data\ImageNet\ILSVRC2012_img_val",
    transform=data_transform)

val_dataset_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)


if __name__ == "__main__":

    for i, (images, labels) in enumerate(val_dataset_loader):
        images = np.array(images)
        images = torch.from_numpy(images)
        for j in range(len(images)):
            image = images[j]
            #print(image)
            #print(image.size()) #torch.Size([3, 100, 100])
            #matplotlib.pyplot.imshow()需要数据是二维的数组或者第三维深度是3或4的三维数组，当第三维深度为1时，使用np.squeeze()压缩数据成为二维数组
            #https://jianzhuwang.blog.csdn.net/article/details/103723536
            plt.imshow((image).numpy().transpose(1, 2, 0))  # 显示图片
            plt.axis('off')  # 不显示坐标轴
            plt.title("$The label of the picture is {} $".format(labels[j]))
            plt.show()
            break

    print(val_dataset[0][0].size()) #第一张图片的图片矩阵
    print(val_dataset[0][1]) #第一张图片的标签
    print(val_dataset.class_to_idx) #查看子文件夹与标签的映射，注意：不是顺序映射
