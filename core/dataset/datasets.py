import numpy as np
import os
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from core.base_objects import ConceptType
from core.utils import load_json


class TextualInversionDataset(Dataset):
    """Class for handling the Textual Inversion dataset"""

    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property=ConceptType.OBJECT,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set_split="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set_split == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": Image.LINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]

        prompt_templates = load_json(
            "core/training/textual_inversion/prompt_templates.json"
        )
        self.templates = (
            prompt_templates["style"]
            if ConceptType.STYLE
            else prompt_templates["object"]
        )
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i % self.num_images])

        if image.mode != "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example = {
            "input_ids": self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        }
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
            ]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


class SDXLDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        class_data_root=None,
        class_num=None,
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        instance_image = exif_transpose(instance_image)

        if instance_image.mode != "RGB":
            instance_image = instance_image.convert("RGB")

        example = {"instance_images": self.image_transforms(instance_image)}

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            class_image = exif_transpose(class_image)

            if class_image.mode != "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

        return example
