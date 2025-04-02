import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b1

def get_model(model_name, model_family, pretrained, num_classes, transfer_learning):
    effnet_dict = {
        'efficientnet-b0': efficientnet_b0,
        'efficientnet-b1': efficientnet_b1,
    }
    if model_family != 'efficientnet':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    model_fn = effnet_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'efficientnet-b0'
    model_family = 'efficientnet'
    pretrained = True
    num_classes = 10
    transfer_learning = True
    
    model = get_model(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)