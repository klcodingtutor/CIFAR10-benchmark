import torch.nn as nn
import torchvision.models as models

def get_resnet(model_name, pretrained=False, num_classes=10):
    resnet_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }
    model_fn = resnet_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_resnet('resnet18', pretrained=True)
    print(model)