import torch.nn as nn
import torchvision.models as models

def get_resnet(model_name='resnet18', num_classes=10, pretrained=False):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model