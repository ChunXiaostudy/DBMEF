# 通过训练好的模型预测图片
import os
from PIL import Image
from torch import nn
import torch
from torchvision import models, transforms
from easydict import EasyDict


if __name__ == "__main__":
    # config
    cfg = EasyDict()
    cfg.num_cls = 102
    cfg.transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 0-225 -> 0-1 float HWC-> CHW   BCHW
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 减去均值 除以方差
    ])
    
    # 初始化模型
    model = models.resnet18()  # 1000  fc
    in_features = model.fc.in_features
    fc = nn.Linear(in_features=in_features, out_features=cfg.num_cls)
    model.fc = fc
    # 载入训练好的权重
    model_weights = r"D:\CV_project_class\img_cls\ouputs\20231010-1518\best.pth"
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint["model"])
    

    # 读图-预处理
    data_dir = r"D:\CV_project_class\data\flowers_data\after_drag_valid"
    for img_name in os.listdir(data_dir):
        
        img_path = os.path.join(data_dir, img_name)
        img0 = Image.open(img_path).convert("RGB")

        img: torch.Tensor = cfg.transforms(img0)  # CHW 
        # 1CHW
        img = img.unsqueeze(dim=0)
        
        # 推理
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device=device)
        model.eval()  ##
        img = img.to(device=device)
        with torch.no_grad():  ##
            output = model(img)  # [1, 102]

        _, pred_label = torch.max(output, 1)
    
        # 展示
        print(f"path: {img_path}, pred label: {int(pred_label)}")
        print(torch.topk(output, k=5))