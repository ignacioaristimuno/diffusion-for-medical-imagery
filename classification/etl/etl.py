import math
import os
import pandas as pd
import random
import shutil
import sys
from typing import List

sys.path.append(os.getcwd())

from core.settings import custom_logger


class ClassificationETL:
    """Class for ETL operations for training the classification models"""

    def __init__(
        self, folder: str, class_labels: List[str], test_size: float, val_size=float
    ) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.folder = folder
        self.class_labels = class_labels
        self.BASE_IMAGES_FOLDERS = "data/classification"
        self.test_size = test_size
        self.val_size = val_size
        self.labels_dict = {
            "dermatofibroma": "df",
            "melanoma": "mel",
            "vascular": "vasc",
            "basal_cell_carcinoma": "bcc",
            "nevus": "nv",
            "pigmented_benign_keratosis": "bkl",
            "pigmented_bowens": "akiec",
        }

    def run(self) -> None:
        """Run the ETL process for classification models"""

        random.seed(42)
        self.logger.info("Starting ETL process...")
        min_images = self._get_min_images_in_dataset()
        self._copy_images_to_training_folders(min_images)
        self.logger.info("ETL process completed!")

    def prepare_raw_data(self) -> None:
        """Prepare the raw data for the classification models"""

        # Read metadata csv
        df = pd.read_csv("../dataset/HAM10000_metadata.csv")

        # Folder structure
        self.logger.info("Creating raw folders...")
        os.makedirs(f"{self.BASE_IMAGES_FOLDERS}/raw")
        for label in self.class_labels:
            os.makedirs(f"{self.BASE_IMAGES_FOLDERS}/raw/{label}")

            df_label = df[df["dx"] == self.labels_dict[label]]
            self.logger.info(f"{df_label.shape[0]}, {df_label.shape[1]}")
            for image_id in df_label["image_id"].to_list():
                shutil.copy(
                    self._get_image_path(image_id),
                    f"{self.BASE_IMAGES_FOLDERS}/raw/{label}/{image_id}.jpg",
                )

        self.logger.info("Raw data prepared!")

    def _get_image_path(self, image_id: str) -> str:
        """Get the image path from the image id"""

        if int(image_id.split(".")[0].split("_")[1]) > 29305:
            return f"../dataset/HAM10000_images_part_2/{image_id}.jpg"
        else:
            return f"../dataset/HAM10000_images_part_1/{image_id}.jpg"

    def _get_min_images_in_dataset(self):
        """Get the minimum number of images in the dataset"""

        min_images = 0
        for label in os.listdir(f"{self.BASE_IMAGES_FOLDERS}/raw"):
            if not label.startswith("."):
                images_count = len(
                    [
                        file
                        for file in os.listdir(
                            f"{self.BASE_IMAGES_FOLDERS}/raw/{label}"
                        )
                        if file.endswith(".jpg")
                        or file.endswith(".png")
                        or file.endswith(".jpeg")
                    ]
                )
                self.logger.info(f"Found {images_count} images for label {label}")
                if min_images == 0 or images_count < min_images:
                    min_images = images_count

        self.logger.info(f"Mimimum number of images per class in dataset: {min_images}")
        return min_images

    def _copy_images_to_training_folders(self, min_images: int) -> None:
        """Copy the images to the training folders"""

        self.logger.info("Copying images to training folders...")
        for label in os.listdir(f"{self.BASE_IMAGES_FOLDERS}/raw"):
            if not label.startswith("."):
                images = [
                    file
                    for file in os.listdir(f"{self.BASE_IMAGES_FOLDERS}/raw/{label}")
                    if file.endswith(".jpg")
                    or file.endswith(".png")
                    or file.endswith(".jpeg")
                ]
                images = random.sample(
                    images, max(min_images, min(len(images), int(min_images * 1.2)))
                )

                # Train, val, test split
                test_images = random.sample(
                    images, math.ceil(len(images) * self.test_size)
                )
                val_images = random.sample(
                    [img for img in images if img not in test_images],
                    math.ceil(len(images) * self.val_size),
                )
                train_images = [
                    img
                    for img in images
                    if img not in test_images and img not in val_images
                ]

                # Copy test images
                os.makedirs(f"{self.BASE_IMAGES_FOLDERS}/{self.folder}/test/{label}")
                for image in test_images:
                    shutil.copy(
                        f"{self.BASE_IMAGES_FOLDERS}/raw/{label}/{image}",
                        f"{self.BASE_IMAGES_FOLDERS}/{self.folder}/test/{label}/{image}",
                    )
                self.logger.info(
                    f"Copied {len(test_images)} images for label {label} to the test folder"
                )

                # Copy validation images
                os.makedirs(f"{self.BASE_IMAGES_FOLDERS}/{self.folder}/val/{label}")
                for image in val_images:
                    shutil.copy(
                        f"{self.BASE_IMAGES_FOLDERS}/raw/{label}/{image}",
                        f"{self.BASE_IMAGES_FOLDERS}/{self.folder}/val/{label}/{image}",
                    )
                self.logger.info(
                    f"Copied {len(val_images)} images for label {label} to the validation folder"
                )

                # Copy training images
                os.makedirs(f"{self.BASE_IMAGES_FOLDERS}/{self.folder}/train/{label}")
                for image in train_images:
                    shutil.copy(
                        f"{self.BASE_IMAGES_FOLDERS}/raw/{label}/{image}",
                        f"{self.BASE_IMAGES_FOLDERS}/{self.folder}/train/{label}/{image}",
                    )
                self.logger.info(
                    f"Copied {len(train_images)} images for label {label} to the training folder"
                )

        self.logger.info("Images copied to training folders!")


if __name__ == "__main__":
    folder = "baseline"
    class_labels = ["dermatofibroma", "melanoma", "vascular"]
    test_size = 0.25
    val_size = 0.2
    etl = ClassificationETL(
        folder=folder, class_labels=class_labels, test_size=test_size, val_size=val_size
    )
    if not os.path.exists("data/classification/raw"):
        etl.prepare_raw_data()
    etl.run()
