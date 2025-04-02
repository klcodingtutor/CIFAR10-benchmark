from .resnet import get_resnet
from .efficientnet import get_efficientnet
from .vgg import get_vgg
from .densenet import get_densenet
from .mobilenet import get_mobilenet

__all__ = [
    'get_resnet',
    'get_efficientnet',
    'get_vgg',
    'get_densenet',
    'get_mobilenet'
]

# a decorator wrapper to wrap all of the model above, and will print out the args
def model_wrapper(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args} and kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

# Wrap all model functions with the decorator
get_resnet = model_wrapper(get_resnet)
get_efficientnet = model_wrapper(get_efficientnet)
get_vgg = model_wrapper(get_vgg)
get_densenet = model_wrapper(get_densenet)
get_mobilenet = model_wrapper(get_mobilenet)