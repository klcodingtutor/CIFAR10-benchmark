import yaml
import os

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("-----------------------------------------------------------------------------------")
    print("Config loaded:")
    print(config)
    print("-----------------------------------------------------------------------------------")

    return config