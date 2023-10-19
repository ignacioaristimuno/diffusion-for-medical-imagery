import os
import random
import shutil
from typing import Dict


class ImagesMixer:
    """Class for mixing images from different folders"""

    def __init__(self) -> None:
        self.BASE_IMAGES_FOLDERS = (
            "data/classification"  # Replace with your folder path
        )
        self.TARGET_FOLDER = (
            "data/classification/raw_mix"  # Folder to copy mixed images to
        )
        self.CLASS_LABELS = [
            "basal_cell_carcinoma",
            "dermatofibroma",
            "melanoma",
            "pigmented_benign_keratosis",
            "vascular",
        ]
        self.PROPORTIONS = {"raw": 0.8, "raw_generated": 0.2}  # Proportions for mixing

    def run(self) -> None:
        """Method for mixing images from different folders"""

        if os.path.exists(self.TARGET_FOLDER):
            shutil.rmtree(self.TARGET_FOLDER)
        os.makedirs(self.TARGET_FOLDER, exist_ok=True)
        n_images = self._get_folders_images_number()
        self._copy_mixed_images(n_images)
        print(n_images)

    def _get_class_image_count(self, folder: str, class_label: str) -> int:
        """Method for getting number of images in a class label folder"""

        folder_path = os.path.join(self.BASE_IMAGES_FOLDERS, folder, class_label)
        images = len([img for img in os.listdir(folder_path) if img.endswith(".jpg")])
        return images

    def _get_folders_images_number(self) -> Dict[str, int]:
        """Method for getting number of images in class label folders"""

        n_images = {}
        total_images = 0

        # Calculate the number of images in each class label folder
        for folder, mix_ratio in self.PROPORTIONS.items():
            images = 0
            for class_label in self.CLASS_LABELS:
                images += self._get_class_image_count(folder, class_label)
            total_images += images

            # Calculate the number of images to mix for this folder
            target_images = int(total_images * mix_ratio)
            n_images[folder] = target_images

        # Adjust the number of images for folders with lower proportions
        for folder, mix_ratio in self.PROPORTIONS.items():
            if n_images[folder] < sum(
                self._get_class_image_count(folder, class_label)
                for class_label in self.CLASS_LABELS
            ):
                n_images[folder] = sum(
                    self._get_class_image_count(folder, class_label)
                    for class_label in self.CLASS_LABELS
                )

        return n_images

    def _copy_mixed_images(self, n_images: Dict[str, int]) -> None:
        """Method for copying mixed images to target folder"""

        for class_label in self.CLASS_LABELS:
            if not os.path.exists(f"{self.TARGET_FOLDER}/{class_label}"):
                os.makedirs(f"{self.TARGET_FOLDER}/{class_label}")

        for folder, image_count in n_images.items():
            for class_label in self.CLASS_LABELS:
                source_folder = os.path.join(
                    self.BASE_IMAGES_FOLDERS, folder, class_label
                )
                image_files = [
                    img for img in os.listdir(source_folder) if img.endswith(".jpg")
                ]
                num_images_available = len(image_files)

                num_images_to_copy = min(image_count, num_images_available)

                if num_images_to_copy == num_images_available:
                    selected_images = image_files  # Copy all available images
                else:
                    selected_images = random.sample(image_files, num_images_to_copy)

                for img_file in selected_images:
                    source_path = os.path.join(source_folder, img_file)
                    target_path = os.path.join(
                        self.TARGET_FOLDER, class_label, img_file
                    )
                    shutil.copy(source_path, target_path)


if __name__ == "__main__":
    mixer = ImagesMixer()
    mixer.run()
