from io import BytesIO
import json
import os
from PIL import Image
import requests
from typing import Dict, List


CONCEPTS_FOLDER = "data/concepts"


def download_image(url: str) -> Image:
    """Function for downloading an image from an url"""

    try:
        response = requests.get(url)
    except Exception:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def store_images_from_urls(urls: List[str], concept_name: str) -> List[str]:
    """Function for retrieving and storing a list of images from their urls"""

    images = list(filter(None, [download_image(url) for url in urls]))
    save_path = f"{CONCEPTS_FOLDER}/{concept_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    [image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]
    return [f"{save_path}/{i}.jpeg" for i, _ in enumerate(images)]


def show_image_grid(images: List[Image.Image], cols: int = 4, rows: int = 4) -> None:
    """Function for showing a grid of images"""

    width, height = images[0].size
    grid = Image.new("RGB", size=(width * cols, height * rows))
    for i, image in enumerate(images):
        grid.paste(image, box=(width * (i % cols), height * (i // cols)))
    return grid


def load_json(path: str) -> Dict:
    """Function for reading a JSON file from its path"""

    with open(path) as file:
        return json.load(file)
