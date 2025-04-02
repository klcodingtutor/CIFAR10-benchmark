def get_model(model_name, model_family, pretrained, num_classes):
    """Return the specified model architecture."""
    if model_family == 'resnet':
        from torchvision.models import resnet18
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_family == 'efficientnet':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(model_name)
        model._fc = nn.Linear(model._fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Model family '{model_family}' is not supported.")