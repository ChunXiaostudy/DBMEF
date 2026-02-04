from typing import Any, Dict
import os
from PIL import Image
from torch.utils.data import Dataset


class FlowerDataset(Dataset):
    def __init__(self, img_dir, transform=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_infos = []  # path, label ...
        self._get_img_info()
        self.transform = transform
    
    def __getitem__(self, index) -> Any:
        img_info: Dict = self.img_infos[index]
        img_path, label_id = img_info["path"], img_info["label"]

        # PIL 优：适配torchvision.transform 劣：边缘端非py部署不支持PIL读取
        img = Image.open(img_path).convert("RGB")

        # opencv cv2 
        # img = cv2.imread(img_path) # BGR
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label_id, img_path

    def __len__(self):
        return len(self.img_infos)
    
    def _get_img_info(self):
        """根据图片文件夹路径获得所有图片的信息
        """
        # 获得mat文件路径
        label_file = os.path.join(os.path.dirname(self.img_dir), "imagelabels.mat")
        assert os.path.exists(label_file)
        # 读取mat文件
        from scipy.io import loadmat
        # [1, 8189] 0-1: label_id  1-2: label_id, ...
        label_array = loadmat(label_file)["labels"]
        # min_id:1 max_id: 102  1-102   pytorch: 0-101
        label_array -= 1  # from 0
        
        # 根据图像名得到对应的label_id
        for img_name in os.listdir(self.img_dir):
            path = os.path.join(self.img_dir, img_name)
            if not img_name[6:11].isdigit():
                continue
            img_id = int(img_name[6:11])
            col_id = img_id - 1
            cls_id = int(label_array[:, col_id])  # from 0
            self.img_infos.append({"path": path, "label": cls_id})


if __name__ == "__main__":   
    img_dir = r"E:\data\flowers_data\train"
    dataset = FlowerDataset(img_dir)
    img, label_id = dataset[1000]
    data_size = len(dataset)
    print(data_size)

    