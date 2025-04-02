import torch.nn as nn
import torchvision.models as models

def get_resnet(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'resnet':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    resnet_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    model_fn = resnet_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'resnet18'
    model_family = 'resnet'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_resnet(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)