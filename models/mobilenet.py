import torch.nn as nn
import torchvision.models as models

def get_mobilenet(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'mobilenet':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    mobilenet_dict = {
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
    }
    model_fn = mobilenet_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported MobileNet variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classifier layer
    if model_name == 'mobilenet_v2':
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:  # MobileNetV3
        in_features = model.classifier[3].in_features  # Last Linear layer in V3
        model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'mobilenet_v2'
    model_family = 'mobilenet'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_mobilenet(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)