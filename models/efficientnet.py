import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b1

def get_efficientnet(model_name, pretrained=False, num_classes=10):
    effnet_dict = {
        'efficientnet-b0': efficientnet_b0,
        'efficientnet-b1': efficientnet_b1,
    }
    model_fn = effnet_dict.get(model_name)
    if not model_fn:
        raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
    
    model = model_fn(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_efficientnet('efficientnet-b0', pretrained=True)
    print(model)