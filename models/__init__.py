from .resnet import get_resnet
from .efficientnet import get_efficientnet

def get_model(model_name, model_family, pretrained=False, num_classes=10):
    if model_family == 'resnet':
        return get_resnet(model_name, pretrained, num_classes)
    elif model_family == 'efficientnet':
        return get_efficientnet(model_name, pretrained, num_classes)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

if __name__ == "__main__":
    model = get_model('resnet18', 'resnet', pretrained=True)
    print(model)
    model = get_model('efficientnet-b0', 'efficientnet', pretrained=True)
    print(model)