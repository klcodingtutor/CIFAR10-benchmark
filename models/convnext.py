import torch.nn as nn
import torchvision.models as models

def get_convnext(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'convnext':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    convnext_dict = {
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
        'convnext_large': models.convnext_large,
    }
    model_fn = convnext_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported ConvNeXt variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classification head
    in_features = model.classifier[2].in_features  # Last Linear layer in ConvNeXt
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'convnext_tiny'
    model_family = 'convnext'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_convnext(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)