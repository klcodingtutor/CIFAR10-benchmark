import torch.nn as nn
import torchvision.models as models

def get_mobilenet(model_name, num_classes, pretrained, transfer_learning):
    """
    Example:
        get_mobilenet(model_name='mobilenet_v2', num_classes=10, pretrained=False, transfer_learning=False):
    """
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported MobileNet variant: {model_name}")
    
    # Freeze feature extractor layers for transfer learning
    if transfer_learning:
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Modify the classifier for CIFAR-10
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model