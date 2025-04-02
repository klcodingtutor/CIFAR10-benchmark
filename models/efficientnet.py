from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def get_efficientnet_b0(model_name='efficientnet-b0', num_classes=10, pretrained=True):
    if pretrained:
        model = EfficientNet.from_pretrained(model_name)
    else:
        model = EfficientNet.from_name(model_name)
    
    # Modify the classifier for CIFAR-10
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    return model