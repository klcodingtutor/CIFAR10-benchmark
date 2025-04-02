import torch.nn as nn
import torchvision.models as models

def get_resnet18(model_name='resnet18', num_classes=10, pretrained=False):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet34(model_name='resnet34', num_classes=10, pretrained=False):
    if model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet50(model_name='resnet50', num_classes=10, pretrained=False):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet101(model_name='resnet101', num_classes=10, pretrained=False):
    if model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet152(model_name='resnet152', num_classes=10, pretrained=False):
    if model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

