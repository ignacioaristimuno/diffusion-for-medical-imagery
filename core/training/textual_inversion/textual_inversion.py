import os
from typing import List

from core.utils import download_image, load_json


CONCEPTS_FOLDER = "data/concepts"


# Params
what_to_teach = "object"  # ["object", "style"]
placeholder_token = "<cat-toy>"
initializer_token = "toy"  # A word that can summarise what your new concept is, to be used as a starting point


# Images
finetuning_images = [
    "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
]


class TextualInversionTrainer:
    """
    Class for handling the finetuning the Stable Diffusion model using
    Textual Inversion for teaching the model a specific concept.

    https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb#scrollTo=u4c1vbVfnmLf
    """

    def __init__(self, concept_name: str) -> None:
        self.images_folder = f"{CONCEPTS_FOLDER}/{concept_name}"
        self.prompt_templates = load_json(f"{os.getcwd()}/prompt_templates.json")

    def finetune_model(self, urls: List[str]):
        pass

    def _save_images_locally(self, urls: List[str]) -> None:
        images = list(filter(None, [download_image(url) for url in urls]))
        if not os.path.exists(self.images_folder):
            os.mkdir(self.images_folder)
        for i, image in enumerate(images):
            image.save(f"{self.images_folder}/{str(i).rjust(5,'0')}.jpg")
