from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import torch

from core.models import BaseDiffusionModel
from core.settings.config import get_config
from core.utils import show_image_grid


class StableDiffusionWithTextualInversion(BaseDiffusionModel):
    """Class for loading and using the Stable Diffusion model with Textual Inversion"""

    def __init__(self, concept_name: str) -> None:
        super().__init__(self.__class__.__name__)
        self.concept_name = concept_name
        self.model_path = get_config("TextualInversion")["output_dir"]

    def load_model(self) -> None:
        """Method to load the specified pipeline from the pretrained model"""

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(
                self.model_path, subfolder="scheduler"
            ),
            torch_dtype=torch.float16,
        )

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
            show_image_grid(all_images, n_rows, n_images)
