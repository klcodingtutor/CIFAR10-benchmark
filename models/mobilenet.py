import torch.nn as nn
import torchvision.models as models

def get_mobilenet(model_name='mobilenet_v2', num_classes=10, pretrained=False):
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported MobileNet variant: {model_name}")
    
    # Modify the classifier for CIFAR-10
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model