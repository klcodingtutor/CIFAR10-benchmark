import torch.nn as nn
import torchvision.models as models

def get_resnet(model_name='resnet18', num_classes=10, pretrained=False):
    resnet_variants = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
    }
    
    if model_name not in resnet_variants:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    # Get the corresponding ResNet model
    model = resnet_variants[model_name](pretrained=pretrained)
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
