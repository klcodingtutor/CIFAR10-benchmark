import torch.nn as nn
import torchvision.models as models

def get_vgg(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'vgg':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    vgg_dict = {
        'vgg11': models.vgg11,
        'vgg13': models.vgg13,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'vgg11_bn': models.vgg11_bn,
        'vgg13_bn': models.vgg13_bn,
        'vgg16_bn': models.vgg16_bn,
        'vgg19_bn': models.vgg19_bn,
    }
    model_fn = vgg_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported VGG variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final classifier layer
    in_features = model.classifier[6].in_features  # Last Linear layer in VGG
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'vgg16'
    model_family = 'vgg'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_vgg(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)