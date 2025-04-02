import torch.nn as nn
import torchvision.models as models

def get_densenet(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'densenet':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    densenet_dict = {
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'densenet161': models.densenet161,
    }
    model_fn = densenet_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported DenseNet variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classifier layer
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'densenet121'
    model_family = 'densenet'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_densenet(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)