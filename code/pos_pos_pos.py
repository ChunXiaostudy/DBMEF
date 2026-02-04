import os
from PIL import Image
import torch
from torchvision import transforms
from easydict import EasyDict
from torchvision.models import resnet50
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


T = scheduler_config['num_train_timesteps'] # T == 1000
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
def eval_error_neg(unet, scheduler, latent, noise_all_neg, text_emb_negative, t_to_eval, guidance_scale, ts, noise_idxs, text_emb_idxs):

    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            print("###START Btach Evaluate###")
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = noise_all_neg[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            t_input = batch_ts.to(device)
            text_input_n = text_emb_negative[text_emb_idxs[idx: idx + batch_size]]
            noise_pred_n = unet(noised_latent, t_input, encoder_hidden_states=text_input_n).sample

            noise_pred = noise_pred_n
            error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors


@torch.no_grad()
def eval_error_pos(unet, scheduler, latent, noise_all_pos, ts, noise_idxs, text_emb_idxs):

    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            print("###START Btach Evaluate###")
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = noise_all_pos[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            t_input = batch_ts.to(device)
            text_input_p = text_emb_positive[text_emb_idxs[idx: idx + batch_size]]
            noise_pred_p = unet(noised_latent, t_input, encoder_hidden_states=text_input_p).sample
            noise_pred = noise_pred_p
            error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors


def find_new_pred(pred_all, times_p, error_list_p_1, error_list_p_2, error_list_p_3):   # 看看平票的情况多不
    for item in pred_all:
        times_p[item] += 1
    max_time = 0
    idx = 0
    for key, values in times_p.items():
        if values > max_time:
            idx = key
            max_time = values
    if max_time != 1:
        return idx
    else:
        return pred_all[2]

         

        
        

if __name__ == "__main__":
    seed_torch(888)
    logger = setup_logger(cfg.log_path, 'w')
    os.makedirs(cfg.output_dir, exist_ok=True)
    start_time = time.time()
    logger.info("start train...")
    model = resnet50(pretrained=True)
    model = model.to(device)

    max_n_samples = cfg.max_n_samples
    guidance_scale = cfg.guidance_scale
    data_dir = cfg.data_dir



    correct = 0
    total_num = 0
    batch_size = cfg.batch_size
    msp_5 = 0.4998440146446228  # 0.01
    num_do_again = 0 # 有多少张图片需要再分类一次
    original_t_2_f = 0
    original_f_2_t = 0 
    original_f_drop = 0
    img_size = cfg.img_size  # 这个地方可以改成256
    img_size_s = 256

    for img_name in os.listdir(data_dir):
        flag = True
        img_path = os.path.join(data_dir, img_name)

        img0 = Image.open(img_path).convert("RGB")
        image_sd = transforms_4_sd(image=img0, img_size=img_size) # 将图片转化为512 512 用于stablediffusion
        image_sd_s = transforms_4_sd(image=img0, img_size=256)
        image_sd = image_sd.to(device).unsqueeze(0)  # [3, 512, 512] -> [1, 3, 512, 512]
        image_sd_s = image_sd_s.to(device).unsqueeze(0)
        
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
        times_p = {key:0 for key in pred5}

        if flag: # 小于msp 需要再分类 这里可能有对的分错 也可能有错的分对
            logger.info(str(img_name) + " need cls again")
            num_do_again += 1
            
            start = T // max_n_samples // 2
            t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]  # len: 100 [5, 15, ..., 995] 全部的【0-1000】内均匀的
            # t_to_eval_f = list(range(start, T // 2, T // max_n_samples // 2))[:max_n_samples] # front 500之前的
            # t_to_eval_b = list(range(T // 2, T, T // max_n_samples // 2))[:max_n_samples]  # back 500之后的
            t_to_eval_f = [i-1 for i in t_to_eval ]
            t_to_eval_f2 = [i-2 for i in t_to_eval ]
            # t_to_eval_b = list(range(T // 2, T, T // max_n_samples // 2))[:max_n_samples]  # back 500之后的
            t_to_eval_b = [i+1 for i in t_to_eval ]
            t_to_eval_b2 = [i+2 for i in t_to_eval ]

            noise_all_pos_1 = torch.randn((max_n_samples, 4, img_size//8, img_size//8), device=device)
            noise_all_pos_2 = torch.randn((max_n_samples, 4, img_size//8, img_size//8), device=device)
            noise_all_pos_3 = torch.randn((max_n_samples, 4, img_size//8, img_size//8), device=device)
            noise_all_pos_4 = torch.randn((max_n_samples, 4, img_size//8, img_size//8), device=device)
            noise_all_pos_5 = torch.randn((max_n_samples, 4, img_size//8, img_size//8), device=device)
            
            error_list_n, error_list_p_1, error_list_p_2 = [], [], []
            pred5_extend = []
  
            ts = t_to_eval * 5  # 5个label, 每个label都是相同的100条
            ts_f = t_to_eval_f * 5
            ts_b = t_to_eval_b * 5
            ts_f2 = t_to_eval_f2 * 5
            ts_b2 = t_to_eval_b2 * 5

            noise_idxs = [i for i in range(max_n_samples)] * 5 # each label 0-99 total 500
            text_emb_idxs = [0] * max_n_samples + [1] * max_n_samples + [2] * max_n_samples + [3] * max_n_samples + [4] * max_n_samples  


            for i in range(len(pred5)):
                pred5_extend.append(pred5)

            text_emb_positive = []
            # text_emb_negative = []
            for i, index in enumerate(pred5_extend):
                positive_prompt = get_prompt(index[i])

                # negative_index = index[:i] + index[i+1:]

                # negative_prompt = get_prompt_negative_2(negative_index)
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
                
            # error_1_image_neg = eval_error_neg(unet, scheduler, x0, noise_all_neg, text_emb_negative, t_to_eval, guidance_scale, ts, noise_idxs, text_emb_idxs)
            error_1_image_pos_1 = eval_error_pos(unet, scheduler, x0, noise_all_pos = noise_all_pos_1, ts=ts_b2, noise_idxs=noise_idxs, text_emb_idxs = text_emb_idxs)
            error_1_image_pos_2 = eval_error_pos(unet, scheduler, x0, noise_all_pos = noise_all_pos_2, ts=ts_f2, noise_idxs=noise_idxs, text_emb_idxs = text_emb_idxs) #256分辨率的
            error_1_image_pos_3 = eval_error_pos(unet, scheduler, x0, noise_all_pos = noise_all_pos_3, ts=ts, noise_idxs=noise_idxs, text_emb_idxs = text_emb_idxs) #256分辨率的
            error_1_image_pos_4 = eval_error_pos(unet, scheduler, x0, noise_all_pos = noise_all_pos_3, ts=ts_b, noise_idxs=noise_idxs, text_emb_idxs = text_emb_idxs) #256分辨率的
            error_1_image_pos_5 = eval_error_pos(unet, scheduler, x0, noise_all_pos = noise_all_pos_3, ts=ts_f, noise_idxs=noise_idxs, text_emb_idxs = text_emb_idxs) #256分辨率的

            # lossn1 = error_1_image_neg[:max_n_samples]  
            # lossn2 = error_1_image_neg[max_n_samples: 2 * max_n_samples]
            # lossn3 = error_1_image_neg[2 * max_n_samples : 3 * max_n_samples]
            # lossn4 = error_1_image_neg[3 * max_n_samples: 4 * max_n_samples]
            # lossn5 = error_1_image_neg[4 * max_n_samples: ]
            
            loss1p1 = error_1_image_pos_1[:max_n_samples]  
            loss1p2 = error_1_image_pos_1[max_n_samples: 2 * max_n_samples]
            loss1p3 = error_1_image_pos_1[2 * max_n_samples : 3 * max_n_samples]
            loss1p4 = error_1_image_pos_1[3 * max_n_samples: 4 * max_n_samples]
            loss1p5 = error_1_image_pos_1[4 * max_n_samples: ]

            loss2p1 = error_1_image_pos_2[:max_n_samples]  
            loss2p2 = error_1_image_pos_2[max_n_samples: 2 * max_n_samples]
            loss2p3 = error_1_image_pos_2[2 * max_n_samples : 3 * max_n_samples]
            loss2p4 = error_1_image_pos_2[3 * max_n_samples: 4 * max_n_samples]
            loss2p5 = error_1_image_pos_2[4 * max_n_samples: ]

            loss3p1 = error_1_image_pos_3[:max_n_samples]  
            loss3p2 = error_1_image_pos_3[max_n_samples: 2 * max_n_samples]
            loss3p3 = error_1_image_pos_3[2 * max_n_samples : 3 * max_n_samples]
            loss3p4 = error_1_image_pos_3[3 * max_n_samples: 4 * max_n_samples]
            loss3p5 = error_1_image_pos_3[4 * max_n_samples: ]

            loss4p1 = error_1_image_pos_4[:max_n_samples]  
            loss4p2 = error_1_image_pos_4[max_n_samples: 2 * max_n_samples]
            loss4p3 = error_1_image_pos_4[2 * max_n_samples : 3 * max_n_samples]
            loss4p4 = error_1_image_pos_4[3 * max_n_samples: 4 * max_n_samples]
            loss4p5 = error_1_image_pos_4[4 * max_n_samples: ]

            loss5p1 = error_1_image_pos_5[:max_n_samples]  
            loss5p2 = error_1_image_pos_5[max_n_samples: 2 * max_n_samples]
            loss5p3 = error_1_image_pos_5[2 * max_n_samples : 3 * max_n_samples]
            loss5p4 = error_1_image_pos_5[3 * max_n_samples: 4 * max_n_samples]
            loss5p5 = error_1_image_pos_5[4 * max_n_samples: ]

            lossp1, lossp2, lossp3, lossp4, lossp5 = [], [], [], [], []

            lossp1.append(sum(loss1p1)/max_n_samples)
            lossp1.append(sum(loss1p2)/max_n_samples)
            lossp1.append(sum(loss1p3)/max_n_samples)
            lossp1.append(sum(loss1p4)/max_n_samples)
            lossp1.append(sum(loss1p5)/max_n_samples)

            lossp2.append(sum(loss2p1)/max_n_samples)
            lossp2.append(sum(loss2p2)/max_n_samples)
            lossp2.append(sum(loss2p3)/max_n_samples)
            lossp2.append(sum(loss2p4)/max_n_samples)
            lossp2.append(sum(loss2p5)/max_n_samples)

            lossp3.append(sum(loss3p1)/max_n_samples)
            lossp3.append(sum(loss3p2)/max_n_samples)
            lossp3.append(sum(loss3p3)/max_n_samples)
            lossp3.append(sum(loss3p4)/max_n_samples)
            lossp3.append(sum(loss3p5)/max_n_samples)

            lossp4.append(sum(loss4p1)/max_n_samples)
            lossp4.append(sum(loss4p2)/max_n_samples)
            lossp4.append(sum(loss4p3)/max_n_samples)
            lossp4.append(sum(loss4p4)/max_n_samples)
            lossp4.append(sum(loss4p5)/max_n_samples)

            lossp5.append(sum(loss5p1)/max_n_samples)
            lossp5.append(sum(loss5p2)/max_n_samples)
            lossp5.append(sum(loss5p3)/max_n_samples)
            lossp5.append(sum(loss5p4)/max_n_samples)
            lossp5.append(sum(loss5p5)/max_n_samples)

            error_list_p_1 = np.array(lossp1)
            error_list_p_2 = np.array(lossp2)
            error_list_p_3 = np.array(lossp3)
            error_list_p_4 = np.array(lossp4)
            error_list_p_5 = np.array(lossp5)
            
            
            new_pred_p_1 = pred5[np.argmin(error_list_p_1)]
            new_pred_p_2 = pred5[np.argmin(error_list_p_2)]
            new_pred_p_3 = pred5[np.argmin(error_list_p_3)]
            new_pred_p_4 = pred5[np.argmin(error_list_p_4)]
            new_pred_p_5 = pred5[np.argmin(error_list_p_5)]

            pred_all = []
            pred_all.append(new_pred_p_1)
            pred_all.append(new_pred_p_2)
            pred_all.append(new_pred_p_3)
            pred_all.append(new_pred_p_4)
            pred_all.append(new_pred_p_5)
            logger.info("before vote: " + str(pred_all))
            new_pred = find_new_pred(pred_all, times_p, error_list_p_1, error_list_p_2, error_list_p_3)
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
    logger.info(f"Train done, use time {t_use:.3f} hours, acc: {acc_final:.3f}")
    logger.info("original_t_2_f: " + str(original_t_2_f))
    logger.info("original_f_2_t: " + str(original_f_2_t))
    logger.info(str(num_do_again) + " images do diffusion classifier")
    logger.info("original_f_drop: " + str(original_f_drop))
    