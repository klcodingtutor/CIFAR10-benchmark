import torch.nn as nn
import torchvision.models as models

def get_resnet(model_name='resnet18', num_classes=10, pretrained=False, transfer_learning=False):
    # Dictionary mapping model names to torchvision model functions
    resnet_models = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
    }
    
    if model_name not in resnet_models:
        raise ValueError(f"Unsupported ResNet variant: {model_name}")
    
    # Load the specified ResNet model
    model = resnet_models[model_name](pretrained=pretrained)
    
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if transfer_learning:
        # Freeze all layers except the final fully connected layer
        for param in model.parameters():
            param.requires_grad = False
        # Ensure the final layer is trainable
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model
