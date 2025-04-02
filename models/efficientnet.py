from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def get_efficientnet(model_name, num_classes, pretrained):
    """
    Creates and returns an EfficientNet model customized for a specific number of classes.
    Args:
        model_name (str): The name of the EfficientNet model to use (e.g., 'efficientnet-b0').
        num_classes (int): The number of output classes for the classifier.
        pretrained (bool): Whether to load pretrained weights for the model.
    Returns:
        torch.nn.Module: An EfficientNet model with a modified classifier for the specified number of classes.
    """
    if pretrained:
        model = EfficientNet.from_pretrained(model_name)
    else:
        model = EfficientNet.from_name(model_name)
    
    # Modify the classifier for CIFAR-10
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    return model