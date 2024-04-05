import torch
import random
import PIL.Image
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Tuple
from torchvision import transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple, Union
from sklearn.metrics.pairwise import pairwise_distances
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
    verify_str_arg,
)


class Flowers102Triplet(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset modified for Triplet Loss.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"``, ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a
            transformed version. E.g., ``transforms.RandomCrop``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        from scipy.io import loadmat

        # Load the dataset splits, labels, and construct mappings
        set_ids = loadmat(
            self._base_folder / self._file_dict["setid"][0], squeeze_me=True
        )
        self.image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(
            self._base_folder / self._file_dict["label"][0], squeeze_me=True
        )["labels"]
        self.image_id_to_label = {
            image_id: label
            for image_id, label in zip(range(1, len(labels) + 1), labels)
        }

        # Group image ids by label
        self.label_to_image_ids = {}
        for image_id, label in self.image_id_to_label.items():
            if label not in self.label_to_image_ids:
                self.label_to_image_ids[label] = []
            self.label_to_image_ids[label].append(image_id)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        anchor_id = self.image_ids[index]
        anchor_label = self.image_id_to_label[anchor_id]

        positive_id = random.choice(self.label_to_image_ids[anchor_label])
        while positive_id == anchor_id:
            positive_id = random.choice(self.label_to_image_ids[anchor_label])

        negative_label = random.choice(list(self.label_to_image_ids.keys()))
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.label_to_image_ids.keys()))
        negative_id = random.choice(self.label_to_image_ids[negative_label])

        anchor_img = PIL.Image.open(
            self._images_folder / f"image_{anchor_id:05d}.jpg"
        ).convert("RGB")
        positive_img = PIL.Image.open(
            self._images_folder / f"image_{positive_id:05d}.jpg"
        ).convert("RGB")
        negative_img = PIL.Image.open(
            self._images_folder / f"image_{negative_id:05d}.jpg"
        ).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def __len__(self) -> int:
        return len(self.image_ids)

    def _check_integrity(self) -> bool:
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False
        for _, md5 in self._file_dict.values():
            if not check_integrity(str(self._base_folder / _), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            self._download_url_prefix + self._file_dict["image"][0],
            self._base_folder,
            md5=self._file_dict["image"][1],
        )
        for filename, md5 in self._file_dict.values():
            download_url(
                self._download_url_prefix + filename, self._base_folder, md5=md5
            )


# Define transforms for data augmentation and normalization
data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_val_test_loader(triplet: bool = False):
    # Load datasets
    if not triplet:
        train_dataset = Flowers102(
            root="./data", split="train", download=True, transform=data_transforms
        )
        test_dataset = Flowers102(
            root="./data", split="test", download=True, transform=data_transforms
        )
    else:
        train_dataset = Flowers102Triplet(
            root="./data", split="train", transform=data_transforms
        )
        test_dataset = Flowers102Triplet(
            root="./data", split="test", transform=data_transforms
        )
    # Split train_dataset into training and validation datasets
    train_size = int(0.8 * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = random_split(
        train_dataset, [train_size, validation_size]
    )
    print(f"TRAINING SIZE: {len(train_dataset)}")
    print(f"VALIDATION SIZE: {len(validation_dataset)}")
    print(f"TRAINING SIZE: {len(test_dataset)}")

    # Create DataLoader instances for each dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, validation_loader, test_loader


def train_and_evaluate(
    model, train_loader, validation_loader, criterion, optimizer, num_epochs=10
):
    print("START TRAINING")
    # Initialize lists to record the training and validation metrics
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_loss = running_loss / total
        validation_accuracy = correct / total
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {validation_loss:.4f}, Val Acc: {validation_accuracy:.4f}"
        )

    return train_losses, train_accuracies, validation_losses, validation_accuracies


def train_and_evaluate_triple(
    model, train_loader, validation_loader, criterion, optimizer, num_epochs=10
):
    print("START TRAINING")
    train_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for anchors, positives, negatives in train_loader:
            anchors, positives, negatives = (
                anchors.to(device),
                positives.to(device),
                negatives.to(device),
            )

            optimizer.zero_grad()

            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for anchors, positives, negatives in validation_loader:
                anchors, positives, negatives = (
                    anchors.to(device),
                    positives.to(device),
                    negatives.to(device),
                )

                anchor_embeddings = model(anchors)
                positive_embeddings = model(positives)
                negative_embeddings = model(negatives)

                val_loss = criterion(
                    anchor_embeddings, positive_embeddings, negative_embeddings
                )

                running_loss += val_loss.item()
            validation_loss = running_loss / len(validation_loader)
            validation_losses.append(validation_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {validation_loss:.4f}"
        )
    return train_losses, validation_losses


# Test phase
def test_model(model, test_loader, criterion):
    print("START TESTING")
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / total
    test_accuracy = correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def get_embeddings_and_labels(model, test_loader):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb)
            labels.append(targets)

    embeddings = torch.cat(embeddings).to(device)
    labels = torch.cat(labels).to(device)
    return embeddings, labels


def test_model_triplet(embeddings, labels):
    distances = pairwise_distances(embeddings, embeddings, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.argmin(distances, axis=1)
    correct = labels[nearest_neighbors] == labels
    test_accuracy = correct.sum() / correct.size
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy


def plot_figure(
    epochs, train_losses, train_accuracies, validation_losses, validation_accuracies
):
    # Plotting
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
