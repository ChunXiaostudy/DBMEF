from torch import nn
import torchvision.models as models
import torch
import timm


def build_model(model_name, num_cls, pretrained=True):
    if model_name == 'resnet18':
        from .resnet_tv import resnet18
        
        model = resnet18(pretrained=True)
        in_features = model.fc.in_features
        fc = nn.Linear(in_features=in_features, out_features=num_cls)
        model.fc = fc
    elif model_name == 'vgg16':
        from timm import create_model as creat
        model = creat('vgg16.tv_in1k', pretrained=True, num_classes=1000)
        # in_features = model.classifier[-1].in_features
        # fc = nn.Linear(in_features=in_features, out_features=num_cls)
        # model.classifier[-1] = fc

    elif model_name == 'eva02':
        from timm import create_model as creat
        model = creat('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True, num_classes=1000)

    elif model_name == 'vit_small':
        from timm import create_model as creat
        model = creat('vit_small_patch32_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=1000)
    
    elif model_name == 'mobilenet':
        from timm import create_model as creat
        model = creat('mobilenetv3_small_100.lamb_in1k', pretrained=True, num_classes=1000)

    elif model_name == 'se_resnet50':
        from .se_resnet import se_resnet50
        
        model = se_resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        fc = nn.Linear(in_features=in_features, out_features=num_cls)
        model.fc = fc
    elif model_name == "resnet50":    # 在imagenet数据集上需要把下面的替换删除掉
        from .resnet_tv import resnet50

        model = resnet50(pretrained=True)
        in_features = model.fc.in_features
        fc = nn.Linear(in_features=in_features, out_features=num_cls)
        model.fc = fc
    # elif model_name == "resnet18":    # 在imagenet数据集上需要把下面的替换删除掉
    #     from .resnet_tv import resnet18

    #     model = resnet18(pretrained=True)
    #     in_features = model.fc.in_features
    #     fc = nn.Linear(in_features=in_features, out_features=num_cls)
    #     model.fc = fc
    # elif model_name == "resnet50":    # 在imagenet数据集上需要把下面的替换删除掉
    #     from .resnet_tv import resnet50

    #     model = resnet50(pretrained=True)
    #     # in_features = model.fc.in_features
    #     # fc = nn.Linear(in_features=in_features, out_features=num_cls)
    #     # model.fc = fc
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=False)
        num_fc = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=num_cls)
        
    elif model_name == "vit":
        from timm import create_model as creat
        model = creat('vit_base_patch16_224', pretrained=True, num_classes=1000)

    elif model_name == "vgg16_tv1k":
        from timm import create_model as creat
        model = timm.create_model('vgg16.tv_in1k', pretrained=True)
        

    else:
        raise Exception(f'only support resnet18, vgg16_bn, resnet50 and se_resnet50, but got {model_name}')
        
    return model