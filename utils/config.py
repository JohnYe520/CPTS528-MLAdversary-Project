import yaml

# Loads the configuration from a file
def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
