import os
from yaml import safe_load


BASE_CONFIG_PATH = os.path.join("core", "settings", "base_config.yml")


def get_config(key: str = None):
    """Function for retrieving base configurations from base_config.yml file"""

    with open(BASE_CONFIG_PATH, "r") as conf:
        settings = safe_load(conf)
        if key and (key in settings.keys()):
            return settings[key]
        return settings
