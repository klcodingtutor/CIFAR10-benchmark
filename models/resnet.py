import torch.nn as nn
import torchvision.models as models

def get_resnet(model_name, num_classes, pretrained):
    """
    Creates a ResNet model with a modified fully connected layer for a specified number of classes.
    Args:
        model_name (str): The name of the ResNet variant to use. Supported variants are 
                          'resnet18', 'resnet34', 'resnet50', 'resnet101', and 'resnet152'.
        num_classes (int): The number of output classes for the final fully connected layer.
        pretrained (bool): Whether to load a model pre-trained on ImageNet.
    Returns:
        torch.nn.Module: The ResNet model with the modified fully connected layer.
    Raises:
        ValueError: If an unsupported ResNet variant is specified.
    """
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
    
    # Modify the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
