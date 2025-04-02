from .resnet import get_resnet18, get_resnet34, get_resnet50, get_resnet101, get_resnet152, get_resnet18_transfer_learning
from .efficientnet import get_efficientnet_b0
from .vgg import get_vgg
from .densenet import get_densenet
from .mobilenet import get_mobilenet
from .custom_models import get_simple_cnn_transfer_learning


model_mapping = {
    "get_resnet18": get_resnet18,
    "get_resnet34": get_resnet34,
    "get_resnet50": get_resnet50,
    "get_resnet101": get_resnet101,
    "get_resnet152": get_resnet152,
    "get_efficientnet_b0": get_efficientnet_b0,
    "get_vgg": get_vgg,
    "get_densenet": get_densenet,
    "get_mobilenet": get_mobilenet,
    "get_resnet18_transfer_learning": get_resnet18_transfer_learning,
    "get_simple_cnn_transfer_learning": get_simple_cnn_transfer_learning,
}