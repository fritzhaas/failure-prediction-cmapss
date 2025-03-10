import yaml
import os

# Get absolute path to config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yml")

def load_config():
    """
    Loads the configuration from config.yml
    """
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config

# Load configuration once
CONFIG = load_config()