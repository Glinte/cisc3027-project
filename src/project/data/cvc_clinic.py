import os
from pathlib import Path
from typing import Callable, Tuple, Literal, Mapping

import numpy as np
import torch
from skimage import io
from supervision import mask_to_xyxy
from torch import Tensor
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from torchvision.tv_tensors import BoundingBoxFormat


class ClinicDB(VisionDataset):
    """
    ClinicDB Dataset.

    ClinicDB is a dataset for the evaluation of image-based colonoscopy analysis.
    """
    base_folder = "CVC-ClinicDB"
    url = "https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=1"
    train_percentage = 0.7
    validation_percentage = 0.15
    test_percentage = 0.15

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "validation", "test"] = "train",
        transforms: Callable | None = None,
        transform: Callable[..., Tensor] | None = None,
        target_transform: Callable[..., Tensor] | None = None
    ):
        """
        Initializes the dataset.

        Args:
            root: Root directory of dataset.
            split: The dataset split, supports "train", "validation", and "test".
            transform: A function/transform that takes in a PIL image and returns a transformed version.
            target_transform: A function/transform that takes in the target and transforms it.

        Notes:
            Unlike the torchvision datasets, this dataset does not download the data for you and does not have integrity checks.
            `transforms` and the combination of `transform` and `target_transform` are mutually exclusive.
        """
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)

        self.split = split
        self.data = []
        self.targets = []

        folder_path = os.path.join(self.root, self.base_folder)
        if not os.path.exists(folder_path):
            raise RuntimeError(
                f"Dataset not found. Please download it manually from {self.url}"
            )

        def _parse_file_index(file):
            return int(file.split(".")[0])

        images = sorted(os.listdir(os.path.join(folder_path, "Original")), key=_parse_file_index)
        masks = sorted(os.listdir(os.path.join(folder_path, "Ground Truth")), key=_parse_file_index)

        all_data = [
            (io.imread(os.path.join(folder_path, "Original", img)),
             io.imread(os.path.join(folder_path, "Ground Truth", mask)))
            for img, mask in zip(images, masks)
        ]

        total_samples = len(all_data)
        train_end = int(total_samples * self.train_percentage)
        valid_end = train_end + int(total_samples * self.validation_percentage)

        # Split data
        if self.split == 'train':
            self.data, self.targets = zip(*all_data[:train_end])
        elif self.split == 'validation':
            self.data, self.targets = zip(*all_data[train_end:valid_end])
        elif self.split == 'test':
            self.data, self.targets = zip(*all_data[valid_end:])
        else:
            raise ValueError("unknown split type")

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)  # (N, H, W)

    def __getitem__(self, index: int) -> Tuple[Tensor, Mapping[str, Tensor]]:
        """
        Args:
            index: Index

        Returns:
            TODO
        """
        img, target = self.data[index], self.targets[index]
        img = tv_tensors.Image(img)
        # Add channel dimension to target
        target = np.expand_dims(target, axis=0)
        target = tv_tensors.Mask(target)
        # target = {
        #     "masks": tv_tensors.Mask(target),
        #     # "bbox": tv_tensors.BoundingBoxes(mask_to_xyxy(torch.tensor(target)), format=BoundingBoxFormat.XYXY, canvas_size=img.size)
        # }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def get_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader[tuple[Tensor, Mapping[str, Tensor]]]:
        """
        Returns a DataLoader for the dataset.

        Args:
            batch_size: How many samples per batch to load.
            shuffle: Set to True to have the data reshuffled at every epoch.
            num_workers: How many subprocesses to use for data loading.

        Returns:
            DataLoader
        """

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
