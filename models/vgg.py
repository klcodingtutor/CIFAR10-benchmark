import torch.nn as nn
import torchvision.models as models

def get_vgg(model_name='vgg16', num_classes=10, pretrained=True):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported VGG variant: {model_name}")
    
    # Modify the classifier for CIFAR-10
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model