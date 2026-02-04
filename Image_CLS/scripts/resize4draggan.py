from torchvision import transforms
import os
import shutil
from PIL import Image


def resize_and_copy(img_path):
    img = Image.open(img_path)
    resize = transforms.Resize([512, 512])
    img_resize = resize(img)
    img_resize.save("resized.jpg")


if __name__ == "__main__":
    img_path = r"D:\CV_project_class\data\ImageNet\Image_after_Drag\ILSVRC2012_val_00049425.JPEG"
    resize_and_copy(img_path)
