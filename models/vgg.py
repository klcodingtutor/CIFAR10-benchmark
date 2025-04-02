import torch.nn as nn
import torchvision.models as models

def get_vgg(model_name, num_classes, pretrained):
    """
    Creates and returns a VGG model with a modified classifier for a specified number of classes.
    Args:
        model_name (str): The name of the VGG variant to use ('vgg16' or 'vgg11').
        num_classes (int): The number of output classes for the classifier.
        pretrained (bool): Whether to load pretrained weights for the model.
    Returns:
        torch.nn.Module: The VGG model with the modified classifier.
    Raises:
        ValueError: If an unsupported VGG variant is specified.
    """
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported VGG variant: {model_name}")
    
    # Modify the classifier for CIFAR-10
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model