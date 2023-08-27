import json
import os
from PIL import Image

from core.base_objects import ConceptType
from core.settings import custom_logger
from core.utils import (
    show_image_grid,
    store_images_from_urls,
    CONCEPTS_FOLDER,
)


logger = custom_logger(" ".join(__name__.split("_")).title())


if __name__ == "__main__":
    # Parameters
    concept_name = "cat_toy"
    what_to_teach = ConceptType.OBJECT  # ["object", "style"]
    placeholder_token = "<cat-toy>"
    initializer_token = "toy"  # A word that can summarise what your new concept is, to be used as a starting point
    placeholder_prompt = f"A picture of {placeholder_token}"

    # Store images
    finetuning_images = [
        "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
    ]

    logger.info(f"Storing {concept_name} images...")
    images_paths = store_images_from_urls(finetuning_images, concept_name=concept_name)

    # Store metadata json lines
    concept_path = f"{CONCEPTS_FOLDER}/{concept_name}"
    logger.info(f"Storing {concept_name} metadata...")
    with open(f"{concept_path}/metadata.jsonl", "w") as f:
        for image_path in images_paths:
            f.write(
                json.dumps(
                    {
                        "image": image_path,
                        "text": placeholder_prompt,
                    }
                )
                + "\n"
            )

    # Check images
    images = []
    for file_path in os.listdir(concept_path):
        try:
            image_path = os.path.join(concept_path, file_path)
            images.append(Image.open(image_path).resize((512, 512)))
        except Exception:
            print(
                f"{image_path} is not a valid image, please make sure to remove this file from the directory otherwise the training could fail."
            )

    show_image_grid(images, 1, len(images))
