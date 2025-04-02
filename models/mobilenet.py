import torch.nn as nn
import torchvision.models as models

def get_mobilenet(model_name='mobilenet_v2', num_classes=10, pretrained=False):
    def get_mobilenet(model_name, num_classes, pretrained):
        """
        Creates and returns a MobileNet model customized for a specific number of classes.
        Args:
            model_name (str): The variant of MobileNet to use. Currently supports 'mobilenet_v2'.
            num_classes (int): The number of output classes for the model.
            pretrained (bool): Whether to load a model pre-trained on ImageNet.
        Returns:
            torch.nn.Module: A MobileNet model with the classifier modified for the specified number of classes.
        Raises:
            ValueError: If an unsupported MobileNet variant is specified.
        """
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported MobileNet variant: {model_name}")
    
    # Modify the classifier for CIFAR-10
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model