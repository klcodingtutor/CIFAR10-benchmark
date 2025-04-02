import torch.nn as nn
import torchvision.models as models

def get_densenet(model_name='densenet121', num_classes=10, pretrained=False):
    """
    Creates and returns a DenseNet model with a modified classifier for a specified number of classes.
    Args:
        model_name (str): The name of the DenseNet variant to use. Supported values are 'densenet121' and 'densenet169'.
        num_classes (int): The number of output classes for the classifier.
        pretrained (bool): Whether to load a model pre-trained on ImageNet.
    Returns:
        torch.nn.Module: The DenseNet model with the modified classifier.
    Raises:
        ValueError: If an unsupported DenseNet variant is specified.
    """
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported DenseNet variant: {model_name}")
    
    # Modify the classifier for CIFAR-10
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model