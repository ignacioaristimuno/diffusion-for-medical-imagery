from abc import ABC, abstractmethod
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
        n_images: int = 1,
        n_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        n_rows: int = 1,
        show_grid: bool = True,
    ):
        """
        Method for generating images with the loaded Diffusion model.

        The placeholder concept should be included in the prompt.
        E.g.: "A <cat-toy> inside ramen-bowl."

        """

        all_images = []
        for _ in range(n_rows):
            images = self.pipe(
                [prompt] * n_images,
                num_inference_steps=n_inference_steps,
                guidance_scale=guidance_scale,
            ).images
            all_images.extend(images)

        if show_grid:
            all_images = [img for img in all_images if not self.check_nsfw_image(img)]
            show_image_grid(all_images, n_rows, n_images)

        return all_images

    def check_nsfw_image(self, image):
        """Evaluates if an image was labeled as NSFW by the model (completely black image)."""
        return image.getpixel((1, 1)) == (0, 0, 0)
