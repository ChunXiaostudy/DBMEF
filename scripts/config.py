import time
from easydict import EasyDict
from torchvision import transforms
import torch

cfg = EasyDict()
cfg.model_id = "D:\LCX_DIFFUSION_classifer\stable-diffusion-v1-5"
cfg.device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg.device2 = "cuda:1" if torch.cuda.is_available() else "cpu"
cfg.prompt_path = r"D:\LCX_DIFFUSION_classifer\diffusion-classifier-master\prompts\imagenet_prompts.csv"
cfg.data_dir = r"D:\LCX_DIFFUSION_classifer\dataset\ImageNet_Valid_resnet18"

cfg.data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

time_str = time.strftime("%Y%m%d-%H%M")
cfg.output_dir = f"ouputs/{time_str}"
cfg.log_path = cfg.output_dir + "/log.txt"
cfg.prompt_path = r"D:\LCX_DIFFUSION_classifer\diffusion-classifier-master\prompts\imagenet_prompts.csv"
cfg.max_n_samples = 60
cfg.guidance_scale = 1.0
cfg.batch_size = 30
cfg.img_size = 512