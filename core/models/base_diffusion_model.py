from abc import ABC, abstractmethod
import torch

from core.settings.logger import custom_logger


class BaseDiffusionModel(ABC):
    """
    Class for centralizing the main methods for the collection of
    Diffusion Models
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
        """Method to load the specified pipeline from the Diffusers library"""
        pass

    @abstractmethod
    def generate_images(self):
        """Method for generating images with the loaded Diffusion model"""
        pass
