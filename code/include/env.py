import json
def loadConfig(cfg_path):
    with open(cfg_path, 'r') as f:
        config = json.load(f)
    return config
