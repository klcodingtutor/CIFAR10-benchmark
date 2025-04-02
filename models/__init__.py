from .resnet import get_resnet
from .efficientnet import get_efficientnet

def print_args(func):
    def wrapper(*args, **kwargs):
        print(f"Arguments: args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper

@print_args
def get_model(model_name, model_family, pretrained, num_classes, transfer_learning):
    print("DEBUG")
    print(model_name, model_family, pretrained, num_classes, transfer_learning)
    print("DEBUG")
    if model_family == 'resnet':
        return get_resnet(model_name, pretrained, num_classes, transfer_learning)
    elif model_family == 'efficientnet':
        return get_efficientnet(model_name, pretrained, num_classes, transfer_learning)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

if __name__ == "__main__":
    model = get_model('resnet18', 'resnet', pretrained=True, transfer_learning=True)
    print(model)
    model = get_model('efficientnet-b0', 'efficientnet', pretrained=True, transfer_learning=True)
    print(model)