from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import os
import torch

from core.models import BaseDiffusionModel


class StableDiffusionModel(BaseDiffusionModel):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id, self.__class__.__name__)

    def load_model(self) -> None:
        """Method to load the specified pipeline from the Diffusers library"""

        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, scheduler=scheduler, torch_dtype=torch.float16
        )
        self.pipe.enable_xformers_memory_efficient_attention()

    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        n_images: int = 1,
        height: int = 512,
        width: int = 512,
        n_steps: int = 50,
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
            "num_inference_steps": n_steps,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt:
            configs["negative_prompt"] = negative_prompt

        with torch.inference_mode():
            images = self.pipe(**configs).images

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
