from abc import ABCMeta, abstractmethod
import torch

from core.settings.logger import custom_logger


class BaseDiffusionModel(ABCMeta):
    """
    Class for centralizing the main methods for the collection of
    Diffusion Models
    """

    def __init__(self, model_id: str, class_name: str) -> None:
        self.model_id = model_id
        self.BASE_IMAGES_FOLDER = "data/output"
        self.logger = custom_logger(class_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Running model on {device.upper()}!")
        self.device = torch.device(device)

        self.load_model()
        self.pipe.to(self.device)

    @abstractmethod
    def load_model(self) -> None:
        """Method to load the specified pipeline from the Diffusers library"""
        pass

    @abstractmethod
    def generate_images(self):
        """Method for generating images with the loaded Diffusion model"""
        pass
