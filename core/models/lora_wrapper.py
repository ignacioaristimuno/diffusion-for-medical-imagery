from diffusers import DiffusionPipeline
import torch

from core.models import BaseDiffusionModel
from core.settings.config import get_config


class LoRAWrapper(BaseDiffusionModel):
    """Class for loading and using the Stable Diffusion model with LoRA."""

    def __init__(self, concept_name: str) -> None:
        self.configs = get_config("LoRA")
        self.model_path = self.configs["output_dir"]
        super().__init__(self.__class__.__name__)
        self.concept_name = concept_name

    def load_model(self) -> None:
        """Method to load the specified pipeline from the pretrained model."""

        # Load previous pipeline
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        ).to(self.device)

        # Add LoRA layers
        self.pipe.unet.load_attn_procs(self.model_path)
