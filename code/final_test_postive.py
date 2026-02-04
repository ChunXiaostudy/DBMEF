import os
from PIL import Image
import torch
from torchvision import transforms
from easydict import EasyDict
from torchvision.models import resnet50, resnet18
import pandas as pd
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, \
    EulerDiscreteScheduler
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import numpy as np
import random
import json
from diffusion.models import get_sd_model, get_scheduler_config
import tqdm
import torch.nn.functional as F
import time
from scripts.common import setup_logger, seed_torch, scheduler_config, imagenet_classes
from scripts.config import cfg
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


device = cfg.device2
model_id = cfg.model_id

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler).to(device)
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
vae = vae.to(device)
unet = unet.to(device)
text_encoder = text_encoder.to(device)

T = scheduler_config['num_train_timesteps']
prompt_path = cfg.prompt_path
data = pd.DataFrame(pd.read_csv(prompt_path))
nerative_data = pd.DataFrame(pd.read_csv(prompt_path))



def get_prompt(idx):
    return data['prompt'][idx]


def get_prompt_negative_1(idxs):
    neg = "the features of "
    for idx in idxs:
        neg += str(nerative_data['classname'][idx]) + ","
    
    neg = neg[:-1]
    neg += "."

    return neg

def transforms_4_sd(image, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)

@torch.no_grad()
def eval_error(unet, scheduler, latent, noise_all, text_emb_positive, t_to_eval, guidance_scale, ts, noise_idxs, text_emb_idxs):

    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=True):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = noise_all[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            t_input = batch_ts.to(device)
            text_input_p = text_emb_positive[text_emb_idxs[idx: idx + batch_size]]
            # text_input_n = text_emb_negative[text_emb_idxs[idx: idx + batch_size]]
            noise_pred_p = unet(noised_latent, t_input, encoder_hidden_states=text_input_p).sample
            # noise_pred_n = unet(noised_latent, t_input, encoder_hidden_states=text_input_n).sample

            # noise_pred = noise_pred_n + guidance_scale * (noise_pred_p - noise_pred_n)
            noise_pred = noise_pred_p
            error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors


if __name__ == "__main__":
    seed_torch(1024)
    logger = setup_logger(cfg.log_path, 'w')
    os.makedirs(cfg.output_dir, exist_ok=True)
    start_time = time.time()
    logger.info("start train...")
    model = resnet18(pretrained=True)
    model = model.to(device)

    max_n_samples = cfg.max_n_samples
    guidance_scale = cfg.guidance_scale

    data_dir = cfg.data_dir

    correct = 0
    total_num = 0
    batch_size = cfg.batch_size
    # msp_5 = 0.4998440146446228  # 0.01
    msp_5 = 0.37922099232673645 # resnet18
    msp_1 = 0.28802409768104553
    num_do_again = 0 # 有多少张图片需要再分类一次
    original_t_2_f = 0
    original_f_2_t = 0
    original_f_drop = 0
    img_size = cfg.img_size  # 这个地方可以改成256

    for img_name in os.listdir(data_dir):
        flag = True
        img_path = os.path.join(data_dir, img_name)

        img0 = Image.open(img_path).convert("RGB")
        image_sd = transforms_4_sd(image=img0, img_size=img_size) # 将图片转化为512 512 用于stablediffusion
        
        image_sd = image_sd.to(device).unsqueeze(0)  # [3, 512, 512] -> [1, 3, 512, 512]


        x0 = vae.encode(image_sd).latent_dist.mean # [1, 4, 64, 64]
        x0 = 0.18215 * x0

        img = cfg.data_transform(img0)
        img = img.unsqueeze(dim=0)
        model.eval()
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
            prob = F.softmax(output)
            max_prob = torch.max(prob)

        if max_prob > msp_5:
            flag = False
        _, pred_label = torch.max(output, 1)
        _, pred5 = torch.topk(output, 5)
        logger.info(f"img_name: {img_name}, pred label: {int(pred_label)}, top5: {pred5}")
        # get 5 promots
        pred5 = pred5.squeeze(dim=0)
        pred5 = pred5.cpu().detach().numpy()
        pred5 = list(pred5)

        if flag: # 小于msp 需要再分类 这里可能有对的分错 也可能有错的分对
            logger.info(str(img_name) + " need cls again")
            num_do_again += 1
            
            start = T // max_n_samples // 2
            t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]  # len: 100 [5, 15, ..., 995]

            noise_all = torch.randn((max_n_samples, 4, img_size//8, img_size//8), device=device)

 
            error_list = []
            pred5_extend = []
  
            ts = t_to_eval * 5  # 5个label, 每个label都是相同的100条

            noise_idxs = [i for i in range(max_n_samples)] * 5 # each label 0-99 total 500
            text_emb_idxs = [0] * max_n_samples + [1] * max_n_samples + [2] * max_n_samples + [3] * max_n_samples + [4] * max_n_samples  


            for i in range(len(pred5)):
                pred5_extend.append(pred5)

            text_emb_positive = []
            text_emb_negative = []
            for i, index in enumerate(pred5_extend):
                positive_prompt = get_prompt(index[i])
                # positive_prompt = get_prompt(index[i])
                # print(positive_prompt)
                # negative_index = index[i-1:i] + index[i+1:i+2] # 4——2
                # negative_index = index[:i]

                # negative_prompt = get_prompt_negative_1(negative_index)
                # 将prompt通过text_embeddings
                text_input_positive = tokenizer([positive_prompt], padding="max_length",
                            max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt") # ["input_ids"]
                text_input_positive = text_input_positive.to(device)
                text_embeddings_positive = text_encoder(text_input_positive.input_ids[:].to(device), )[0]
                text_emb_positive.append(text_embeddings_positive)
                # print(text_embeddings_positive.shape)
                # text_input_negative = tokenizer([negative_prompt], padding="max_length",
                #             max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt") # ["input_ids"]
                # text_input_negative = text_input_negative.to(device)
                # text_embeddings_negative = text_encoder(text_input_negative.input_ids.to(device), )[0]
                # text_emb_negative.append(text_embeddings_negative)
                # print(text_embeddings_negative.shape)
            text_emb_positive = torch.cat(text_emb_positive, dim=0)
            # text_emb_negative = torch.cat(text_emb_negative, dim=0)
                
            error_1_image = eval_error(unet, scheduler, x0, noise_all, text_emb_positive, t_to_eval, guidance_scale, ts, noise_idxs, text_emb_idxs)
            loss1 = error_1_image[:max_n_samples]  
            loss2 = error_1_image[max_n_samples: 2 * max_n_samples]
            loss3 = error_1_image[2 * max_n_samples : 3 * max_n_samples]
            loss4 = error_1_image[3 * max_n_samples: 4 * max_n_samples]
            loss5 = error_1_image[4 * max_n_samples: ]
            
            loss = []
            loss.append(sum(loss1)/max_n_samples)
            loss.append(sum(loss2)/max_n_samples)
            loss.append(sum(loss3)/max_n_samples)
            loss.append(sum(loss4)/max_n_samples)
            loss.append(sum(loss5)/max_n_samples)

            error_list = np.array(loss)
            # new_pred = pred5[np.argmax(error_list)]
            new_pred = pred5[np.argmin(error_list)]
            logger.info("pred_label_new = " + str(new_pred))

            total_num += 1
            img_name = img_name.split("_")
            img_truelabel = list(img_name)[0]


            if str(new_pred) == img_truelabel:
                correct += 1
                if list(img_name)[0] != list(img_name)[1]:
                    original_f_2_t += 1
            else:
                if list(img_name)[0] == list(img_name)[1]:
                    original_t_2_f += 1
        else:
            logger.info(str(img_name) + " do not need cls again")
            new_pred = pred_label
            new_pred = torch.squeeze(new_pred, dim=0).cpu().numpy()
            total_num += 1
            img_name = img_name.split("_")
            img_truelabel = list(img_name)[0]
            if str(new_pred) == img_truelabel:
                correct += 1
            else:
                if list(img_name)[0] != list(img_name)[1]:
                    original_f_drop += 1
        logger.info("ACC_Now: " + str(correct / total_num))

    end_time = time.time()
    acc_final = correct / total_num
    t_use = (end_time - start_time) / 3600
    print("Running_time: ", (end_time - start_time) / 3600)
    logger.info(f"Train done, use time {t_use:.3f} hours, acc: {acc_final:.3f}")
    logger.info("original_t_2_f: " + str(original_t_2_f))
    logger.info("original_f_2_t: " + str(original_f_2_t))
    logger.info(str(num_do_again) + " images do diffusion classifier")
    logger.info("original_f_drop: " + str(original_f_drop))

    
    