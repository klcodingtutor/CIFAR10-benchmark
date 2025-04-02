import torch.nn as nn
import torchvision.models as models

def get_swin(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'swin':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    swin_dict = {
        'swin_t': models.swin_t,    # Tiny model
        'swin_s': models.swin_s,    # Small model
        'swin_b': models.swin_b,    # Base model
        'swin_v2_t': models.swin_v2_t,  # Tiny, version 2
        'swin_v2_s': models.swin_v2_s,  # Small, version 2
        'swin_v2_b': models.swin_v2_b,  # Base, version 2
    }
    model_fn = swin_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported Swin Transformer variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classification head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'swin_t'
    model_family = 'swin'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_swin(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)