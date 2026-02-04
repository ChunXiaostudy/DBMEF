# 将flower数据集按类别排放，便于分析
import os
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    root_dir = r"E:\data\flowers_data"
    path_mat = os.path.join(root_dir, "imagelabels.mat")
    reorder_dir = os.path.join(root_dir, "reorder")
    jpg_dir = os.path.join(root_dir, "jpg")

    from scipy.io import loadmat
    label_array = loadmat(path_mat)["labels"]

    names = os.listdir(jpg_dir)
    names = [p for p in names if p.endswith(".jpg")]
    for img_name in tqdm(os.listdir(jpg_dir)):
        path = os.path.join(jpg_dir, img_name)
        if not img_name[6:11].isdigit():
            continue
        img_id = int(img_name[6:11])
        col_id = img_id - 1
        cls_id = int(label_array[:, col_id]) - 1  # from 0

        out_dir = os.path.join(reorder_dir, str(cls_id))
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(path, out_dir)      # 复制文件
        
    print("Done")