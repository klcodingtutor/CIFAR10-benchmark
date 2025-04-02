import torch.nn as nn
import torchvision.models as models

def get_shufflenet(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family != 'shufflenet':
        raise ValueError(f"Unsupported model family: {model_family}")
    
    shufflenet_dict = {
        'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,  # 0.5x width
        'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,  # 1.0x width
        'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,  # 1.5x width
        'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,  # 2.0x width
    }
    model_fn = shufflenet_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported ShuffleNet V2 variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model_name = 'shufflenet_v2_x1_0'
    model_family = 'shufflenet'
    pretrained = True
    num_classes = 10
    transfer_learning = True

    model = get_shufflenet(model_name, model_family, pretrained, num_classes, transfer_learning)
    print(model)