def print_args(func):
    def wrapper(*args, **kwargs):
        print(f"Function: {func.__name__}, Arguments: args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper