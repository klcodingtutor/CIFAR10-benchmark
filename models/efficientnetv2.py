import torch.nn as nn
import torchvision.models as models

def get_efficientnetv2(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'efficientnetv2':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    effnetv2_dict = {
        'efficientnet_v2_s': models.efficientnet_v2_s,  # Small
        'efficientnet_v2_m': models.efficientnet_v2_m,  # Medium
        'efficientnet_v2_l': models.efficientnet_v2_l,  # Large
    }
    model_fn = effnetv2_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported EfficientNetV2 variant: {model_name}")
    
    if pretrained:
        print("Using pretrained weights.")
        model = model_fn(pretrained=True)
    else:
        print("Not using pretrained weights.")
        model = model_fn(pretrained=False)
    
    if transfer_learning:
        print("Freezing model parameters for transfer learning.")
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'efficientnet_v2_s'
    model_family = 'efficientnetv2'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_efficientnetv2(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)