import os
from yaml import safe_load


def get_config(key: str = None, config_path: str = None):
    """Function for retrieving base configurations from base_config.yml file"""

    if config_path is None:
        config_path = os.path.join("core", "settings", "base_config.yml")

    with open(config_path, "r") as conf:
        settings = safe_load(conf)
        if key and (key in settings.keys()):
            return settings[key]
        return settings
