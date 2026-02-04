# 统计模型的参数量大小
# import sys
# sys.path.append('.')
# from models import build_model


# for model_name in 'vgg16_bn', 'resnet18', 'se_resnet50':
#     model = build_model(model_name, num_cls=102)
#     total_params = sum(p.numel() for p in model.parameters()) / 1e6
#     print(f"{model_name:15s} Number of parameters: {total_params:.3f}M")

import timm
 
if __name__ == '__main__':
    model_vit = timm.list_models('*vit*')
    print(len(model_vit),model_vit[:])