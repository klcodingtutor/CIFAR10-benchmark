import torch.nn

def print_args(func):
    def wrapper(*args, **kwargs):
        filtered_args = [
            arg if not isinstance(arg, torch.nn.Module) else "<PyTorch Model>"
            for arg in args
        ]
        filtered_kwargs = {
            k: (v if not isinstance(v, torch.nn.Module) else "<PyTorch Model>")
            for k, v in kwargs.items()
        }
        print(f"Function: {func.__name__}, Arguments: args={filtered_args}, kwargs={filtered_kwargs}")
        return func(*args, **kwargs)
    return wrapper