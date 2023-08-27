from abc import ABC, abstractmethod
import os
import torch

from core.settings.logger import custom_logger
from core.utils import show_image_grid


class BaseDiffusionModel(ABC):
    """
    Class for centralizing the main methods for the collection of Diffusion Models.
    """

    def __init__(self, class_name: str) -> None:
        self.BASE_IMAGES_FOLDER = "data/output"
        self.logger = custom_logger(class_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Running the {class_name} model on {device.upper()}!")
        self.device = torch.device(device)

        self.load_model()
        self.pipe.to(self.device)

    @abstractmethod
    def load_model(self) -> None:
        """Method to load the specified pipeline from the Diffusers library."""
        pass

    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        n_images: int = 1,
        height: int = 512,
        width: int = 512,
        n_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        label: str = "general",
        save: bool = False,
    ):
        """Method for generating images with the loaded Diffusion model"""

        configs = {
            "prompt": prompt,
            "num_images_per_prompt": n_images,
            "height": height,
            "width": width,
            "num_inference_steps": n_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt:
            configs["negative_prompt"] = negative_prompt

        with torch.inference_mode():
            images = self.pipe(**configs).images

        images = [img for img in images if self.check_nsfw_image(img)]
        if save:
            base_path = f"{self.BASE_IMAGES_FOLDER}/{self.__class__.__name__}/{label}"
            n_images = len(
                [img for img in os.listdir(base_path) if img.endswith(".png")]
            )
            for image in images:
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                image.save(f"{base_path}/image_{label}_{n_images}.png")
                n_images += 1
            return images

    @staticmethod
    def check_nsfw_image(image):
        """Evaluates if an image was labeled as NSFW by the model (completely black image)."""
        return image.getpixel((1, 1)) == (0, 0, 0)
