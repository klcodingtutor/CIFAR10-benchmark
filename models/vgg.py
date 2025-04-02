import torch.nn as nn
import torchvision.models as models

def get_vgg(model_name, num_classes, pretrained, transfer_learning):
    """
    Example:
         get_vgg(model_name='vgg16', num_classes=10, pretrained=False, transfer_learning=False):
    """
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported VGG variant: {model_name}")
    
    if transfer_learning:
        # Freeze feature extractor layers
        for param in model.features.parameters():
            param.requires_grad = False
    
    # Modify the classifier for CIFAR-10
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model