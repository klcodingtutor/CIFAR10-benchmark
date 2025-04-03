from .resnet import get_resnet
from .efficientnet import get_efficientnet
from .vgg import get_vgg
from .densenet import get_densenet
from .mobilenet import get_mobilenet
from .vit import get_vit
from .swin import get_swin
from .convnext import get_convnext
from .efficientnetv2 import get_efficientnetv2
from .shufflenet import get_shufflenet
from .resnet_32_input import get_resnet_32_input
from .efficientnet_32_input import get_efficientnet_32_input
from .mobilenetv2_32_input import get_mobilenetv2_32_input
from utils.decorator import print_args

@print_args
def get_model(model_name, model_family, pretrained, num_classes, transfer_learning):
    if model_family == 'resnet':
        model = get_resnet(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'resnet_32_input':
        model = get_resnet_32_input(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'efficientnet':
        model = get_efficientnet(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'efficientnet_32_input':
        model = get_efficientnet_32_input(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'vgg':
        model = get_vgg(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'densenet':
        model = get_densenet(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'mobilenet':
        model = get_mobilenet(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'mobilenetv2_32_input':
        model = get_mobilenetv2_32_input(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'vit':
        model = get_vit(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'swin':
        model = get_swin(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'convnext':
        model = get_convnext(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'efficientnetv2':
        model = get_efficientnetv2(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'squeezenet':
        model = get_squeezenet(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    elif model_family == 'shufflenet':
        model = get_shufflenet(
            model_name=model_name,
            model_family=model_family,
            pretrained=pretrained,
            num_classes=num_classes,
            transfer_learning=transfer_learning
        )
    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    print("Model Summary:")
    print(model)
    return model

if __name__ == "__main__":
    # Test various models
    model = get_model('resnet18', 'resnet', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('efficientnet-b0', 'efficientnet', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('vgg16', 'vgg', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('densenet121', 'densenet', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('mobilenet_v2', 'mobilenet', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('vit_b_16', 'vit', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('swin_t', 'swin', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('convnext_tiny', 'convnext', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('efficientnet_v2_s', 'efficientnetv2', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)
    model = get_model('shufflenet_v2_x1_0', 'shufflenet', pretrained=True, num_classes=10, transfer_learning=True)
    print(model)