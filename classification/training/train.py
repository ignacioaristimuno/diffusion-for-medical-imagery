import json
import numpy as np
import os
import shutil
import sys
from typing import Any, Dict, List
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

sys.path.append(os.getcwd())

from core.settings import custom_logger, get_config
from classification.training.model import SkinLesionCNN, get_cnn_settings


BASE_IMAGES_FOLDER = "data/classification"
BASE_MODELS_FOLDER = "models/classification"
BASE_METRICS_FLODER = "metrics/classification"


def get_training_settings() -> Dict[str, Any]:
    """Function for retrieving the configs for the class SkinLesionCNN class"""

    configs = get_config(
        "TrainCNN", config_path="classification/settings/base_config.yml"
    )
    settings = {
        "images_shape": configs["ImagesShape"],
        "batch_size": configs["BatchSize"],
        "double_finetuning": configs["DoubleFinetuning"],
        "horizontal_flip": configs["DataAugmentation"]["HorizontalFlip"],
        "vertical_flip": configs["DataAugmentation"]["VerticalFlip"],
        "zoom_range": configs["DataAugmentation"]["ZoomRange"],
        "rotation_range": configs["DataAugmentation"]["RotationRange"],
        "width_shift_range": configs["DataAugmentation"]["WidthShiftRange"],
        "height_shift_range": configs["DataAugmentation"]["HeightShiftRange"],
    }
    return settings


class SkinLesionCNNTrainer:
    """Class for wrapping the training process of the CNN for skin lesion classification"""

    def __init__(
        self,
        n_classes: int,
        images_shape: List[int],
        batch_size: int,
        double_finetuning: bool,
        horizontal_flip: bool,
        vertical_flip: bool,
        zoom_range: float,
        rotation_range: float,
        width_shift_range: float,
        height_shift_range: float,
    ) -> None:
        self.logger = custom_logger(self.__class__.__name__)

        # Configs
        self.n_classes = n_classes
        self.images_shape = images_shape
        self.batch_size = batch_size
        self.double_finetuning = double_finetuning
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

    def train(self, first_folder: str, second_folder: str, save_folder: str) -> None:
        """Function for training the CNN model for skin lesion classification"""

        # Remove previous models and metrics
        if os.path.exists(f"{BASE_MODELS_FOLDER}/{save_folder}"):
            shutil.rmtree(f"{BASE_MODELS_FOLDER}/{save_folder}")
        if os.path.exists(f"{BASE_METRICS_FLODER}/{save_folder}"):
            shutil.rmtree(f"{BASE_METRICS_FLODER}/{save_folder}")

        # Data Loaders
        first_train_dataloader = None
        first_val_dataloader = None
        if self.double_finetuning:
            first_train_dataloader = self._get_train_dataloader(folder=first_folder)
            first_val_dataloader = self._get_val_dataloader(folder=first_folder)
        second_train_dataloader = self._get_train_dataloader(folder=second_folder)
        second_val_dataloader = self._get_val_dataloader(folder=second_folder)
        test_dataloader = self._get_test_dataloader(folder=second_folder)

        # Training
        skin_lesion_cnn = SkinLesionCNN(**get_cnn_settings(), n_classes=self.n_classes)
        model, metrics = skin_lesion_cnn.train_model(
            first_train_dataloader,
            first_val_dataloader,
            second_train_dataloader,
            second_val_dataloader,
            test_dataloader,
        )

        # Save artifacts
        self._save_model(model, save_folder)
        self._save_metrics(metrics, save_folder)

    def _get_train_dataloader(self, folder: str):
        self.logger.info("Fetching Train DataLoader...")
        train_image_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            zoom_range=self.zoom_range,
            rotation_range=self.rotation_range,
            width_shift_range=self.rotation_range,
            height_shift_range=self.height_shift_range,
        )
        train_data_gen = train_image_gen.flow_from_directory(
            batch_size=self.batch_size,
            directory=f"{BASE_IMAGES_FOLDER}/{folder}/train",
            shuffle=True,
            target_size=(self.images_shape[0], self.images_shape[1]),
            class_mode="categorical",
            interpolation="bilinear",
        )
        self.logger.info(f"Train indices: {train_data_gen.class_indices}")
        return train_data_gen

    def _get_val_dataloader(self, folder: str):
        self.logger.info("Fetching Validation DataLoader...")
        val_image_dataloader = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        val_data_gen = val_image_dataloader.flow_from_directory(
            batch_size=self.batch_size,
            directory=f"{BASE_IMAGES_FOLDER}/{folder}/val",
            shuffle=True,
            target_size=(self.images_shape[0], self.images_shape[1]),
            class_mode="categorical",
            interpolation="bilinear",
        )
        self.logger.info(f"Validation indices: {val_data_gen.class_indices}")
        return val_data_gen

    def _get_test_dataloader(self, folder: str, shuffle: bool = True):
        self.logger.info("Fetching Test DataLoader...")
        test_image_dataloader = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        test_data_gen = test_image_dataloader.flow_from_directory(
            batch_size=self.batch_size,
            directory=f"{BASE_IMAGES_FOLDER}/{folder}/test",
            shuffle=shuffle,
            target_size=(self.images_shape[0], self.images_shape[1]),
            class_mode="categorical",
            interpolation="bilinear",
        )
        self.logger.info(f"Test indices: {test_data_gen.class_indices}")
        return test_data_gen

    def _save_model(self, model: Model, save_folder: str):
        self.logger.info("Saving model...")
        model_path = f"{BASE_MODELS_FOLDER}/{save_folder}/cnn_model_original.h5"
        model.save(model_path)
        self.logger.info(f"Model saved at path {model_path}!")

    def _save_metrics(self, metrics: Dict[str, Any], save_folder: str):
        self.logger.info("Saving metrics...")
        if not os.path.exists(f"{BASE_METRICS_FLODER}/{save_folder}"):
            os.makedirs(f"{BASE_METRICS_FLODER}/{save_folder}")
        metrics_path = f"{BASE_METRICS_FLODER}/{save_folder}/cnn_model_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        self.logger.info(f"Metrics saved at path {metrics_path}!")


if __name__ == "__main__":
    first_folder = "baseline"  # "mix_60_original_40_generated"
    second_folder = "augmentation"
    save_folder = "augmentation"
    n_classes = 5
    trainer = SkinLesionCNNTrainer(**get_training_settings(), n_classes=n_classes)
    trainer.train(
        first_folder=first_folder, second_folder=second_folder, save_folder=save_folder
    )
