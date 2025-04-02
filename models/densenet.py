import torch.nn as nn
import torchvision.models as models

def get_densenet(model_name='densenet121', num_classes=10, pretrained=False):
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported DenseNet variant: {model_name}")
    
    # Modify the classifier for CIFAR-10
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model