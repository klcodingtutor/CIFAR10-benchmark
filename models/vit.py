import torch.nn as nn
import torchvision.models as models

def get_vit(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'vit':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    vit_dict = {
        'vit_b_16': models.vit_b_16,  # Base model, patch size 16
        'vit_b_32': models.vit_b_32,  # Base model, patch size 32
        'vit_l_16': models.vit_l_16,  # Large model, patch size 16
        'vit_l_32': models.vit_l_32,  # Large model, patch size 32
    }
    model_fn = vit_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported Vision Transformer variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classification head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'vit_b_16'
    model_family = 'vit'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_vit(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)