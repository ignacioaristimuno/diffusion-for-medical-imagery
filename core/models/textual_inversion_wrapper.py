from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import torch

from core.models import BaseDiffusionModel
from core.settings.config import get_config


class TextualInversionWrapper(BaseDiffusionModel):
    """Class for loading and using the Stable Diffusion model with Textual Inversion."""

    def __init__(self, concept_name: str) -> None:
        self.model_path = get_config("TextualInversion")["output_dir"]
        super().__init__(self.__class__.__name__)
        self.concept_name = concept_name

    def load_model(self) -> None:
        """Method to load the specified pipeline from the pretrained model."""

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(
                self.model_path, subfolder="scheduler"
            ),
            torch_dtype=torch.float16,
        )
