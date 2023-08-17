from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import os
import torch

from core.models import BaseDiffusionModel


class StableDiffusionModel(BaseDiffusionModel):
    """Class for loading and using the standard Stable Diffusion model"""

    def __init__(self, model_id: str) -> None:
        super().__init__(self.__class__.__name__)
        self.model_id = model_id

    def load_model(self) -> None:
        """Method to load the specified pipeline from the Diffusers library"""

        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, scheduler=scheduler, torch_dtype=torch.float16
        )
        self.pipe.enable_xformers_memory_efficient_attention()
