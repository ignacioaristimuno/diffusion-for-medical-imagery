from efficientnet_pytorch import EfficientNet
import numpy as np
import os
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from core.settings.logger import custom_logger
from core.settings.config import get_config


DATA_FOLDER = "data/raw/skin_lesions"


class CustomDataset(Dataset):
    """Class for creating a custom dataset for clustering images with a pre-trained model."""

    def __init__(self, data_folder, use_transforms: bool = True):
        self.logger = custom_logger(self.__class__.__name__)

        self.logger.info(f"Retrieving images from folder {data_folder}")
        self.data_folder = data_folder
        self.image_paths = [
            os.path.join(data_folder, filename)
            for filename in os.listdir(data_folder)
            if filename.split(".")[-1] in ["png", "jpg", "jpeg"]
        ]

        # Define transformation for image preprocessing
        self.use_transforms = use_transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.use_transforms:
            image = self.transform(image)

        return image


class EmbeddingsGenerator:
    """Class for generating embeddings from images using a pre-trained model."""

    def __init__(self) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.configs = get_config("Clustering")

        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using {device.upper()} as device!")
        self.device = torch.device(device)

        # Load model
        self.model = EfficientNet.from_pretrained(self.configs["model_id"])
        self.model.to(self.device)
        self.model.eval()

    def generate_embeddings(self, dataloader: DataLoader) -> torch.Tensor:
        """Generate embeddings from images using a pre-trained model."""

        self.logger.info(
            f"Generating embeddings for {len(dataloader)} batches of size: {dataloader.batch_size}"
        )
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                features = self.model.extract_features(batch)
                embeddings.append(features)
        embeddings = torch.cat(embeddings, dim=0)
        self.logger.info(f"Embeddings shape (before mean pooling): {embeddings.shape}")
        embeddings = torch.mean(embeddings, dim=(2, 3))
        self.logger.info(f"Embeddings shape (after mean pooling): {embeddings.shape}")
        return embeddings


class Clusterer:
    """Class for clustering images using the embeddings of a pre-trained model."""

    def __init__(self, epsilon: float = None, min_samples: int = None) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.configs = get_config("Clustering")
        self.dbscan = DBSCAN(
            eps=self.configs["epsilon"] if epsilon is None else epsilon,
            min_samples=self.configs["min_samples"]
            if min_samples is None
            else min_samples,
            metric=self.configs["metric"],
        )

    def get_clusters(self, embeddings: torch.Tensor) -> np.ndarray:
        """Get the clusters from using a DBSCAN model."""

        # Generate clusters
        self.logger.info(f"Clustering {embeddings.shape[0]} embeddings using DBSCAN")
        clusters = self.dbscan.fit_predict(embeddings.cpu().numpy())

        # Get labels and clusters
        n_clustered_images = len([img for img in clusters if img != -1])
        self.logger.info(f"Number of clusters: {len(set(clusters))}")
        self.logger.info(
            f"Number of images within a cluster: {n_clustered_images} -> {100*n_clustered_images/len(clusters):.1f}%"
        )
        return clusters


if __name__ == "__main__":
    # Create a DataLoader for the custom dataset
    clustering_configs = get_config("Clustering")
    dataset = CustomDataset(DATA_FOLDER)
    dataloader = DataLoader(
        dataset, batch_size=clustering_configs["batch_size"], shuffle=False
    )

    # Create embeddings for the images
    embeddings_gen = EmbeddingsGenerator()
    embeddings = embeddings_gen.generate_embeddings(dataloader)

    # Create clusters
    clusterer = Clusterer()
    clusters = clusterer.get_clusters(embeddings)

    # Save clusters' data
    filename_to_label = {
        filename: label for filename, label in zip(dataset.image_paths, clusters)
    }
