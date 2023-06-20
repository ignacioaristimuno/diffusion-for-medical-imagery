from io import BytesIO
import json
from PIL import Image
import requests
from typing import Dict


def download_image(url: str) -> Image:
    """Function for downloading an image from an url"""

    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_json(path: str) -> Dict:
    """Function for reading a JSON file from its path"""

    with open(path) as file:
        return json.load(file)
