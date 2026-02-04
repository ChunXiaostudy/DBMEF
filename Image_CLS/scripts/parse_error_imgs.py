# 将错误分类的图片挑出来，进行观察
import os
import pickle
import shutil


def load_pickle(path_file):
    with open(path_file, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    path_pkl = r"D:\LCX_IMG_CLS\ouputs\20240109-1704\error_imgs_best.pkl"
    data_root_dir = r"D:\LCX_DIFFUSION_classifer\dataset\ILSVRC2012_img_val"
    out_dir = path_pkl[:-4]  # 输出文件目录
    error_info = load_pickle(path_pkl)

    for setname, info in error_info.items():  # [train, valid]
        for imgs_data in info:
            label, pred, path_img_rel = imgs_data
            path_img = os.path.join(data_root_dir, os.path.basename(path_img_rel))
            img_dir = os.path.join(out_dir, setname, str(label), str(pred))     # 图片文件夹
            os.makedirs(img_dir, exist_ok=True)
            shutil.copy(path_img, img_dir)      # 复制文件

    print("Done")