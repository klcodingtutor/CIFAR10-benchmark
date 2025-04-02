import yaml


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Loaded configuration from {config_path}: {config}")
    return config

if __name__ == "__main__":
    # Test config loading
    config = load_config("../configs/efficientnet_b0_cifar10.yaml")
    print(config)