import os
from typing import Callable, Any

from pathlib import Path
import logging

import numpy as np
import torch.utils.data
import SimpleITK as sitk
import torchvision.tv_tensors
from PIL import Image
from jaxtyping import Num
from matplotlib import pyplot as plt
from torchvision import tv_tensors

from project.config import PROJECT_ROOT
from project.data.preprocess_luna import generate_masks
from project.types import Array


logger = logging.getLogger(__name__)


class Luna16Dataset(torch.utils.data.Dataset):
    """
    LUNA16 Dataset for image segmentation.

    This dataset class loads CT images and their corresponding segmentation masks.

    Attributes:
        images (dict): Dictionary mapping image IDs to image file paths.
        masks (dict): Dictionary mapping mask IDs to mask file paths.
        ids (list): List of image IDs.
        transforms (callable): Transformations to apply to images and masks.
    """

    combined_images_path = ["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7", "subset8", "subset9"]
    train_subset = ["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7"]
    test_subset = ["subset8", "subset9"]
    masks_path = "mask"

    def __init__(self, root: str | Path, transforms: Callable | None = None, train: bool = True):
        """
        Initializes the dataset.

        Args:
            root: Root directory of dataset.
            transforms: Transformations to apply to images and masks.
        """
        self.root = Path(root)
        self.transforms = transforms
        self.train = train

        self.images = {}
        self.masks = {}
        self.ids = []

        if not os.path.exists(self.root / self.masks_path):
            logger.warning(f"Mask directory not found at {self.root / self.masks_path}, generating masks.")
            generate_masks(self.root, self.root / self.masks_path)

        images_dir = self.train_subset if self.train else self.test_subset
        for dir in images_dir:
            file: Path
            for file in (self.root / dir).iterdir():
                if file.suffix == ".mhd":
                    self.images[file.stem] = file

        for dirpath, _dirnames, filenames in (self.root / self.masks_path).walk():
            filepaths = [dirpath / filename for filename in filenames]
            for file in filepaths:
                uid = file.stem.rstrip("_segmentation")
                if file.suffix == ".mhd" and uid in self.images:
                    self.masks[uid] = file

        self.ids = list(self.images.keys())
        assert len(self.images) == len(self.masks), "Number of images and masks do not match."

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[Num[Array, "1 depth height width"], Num[Array, "1 depth height width"]]:
        """
        Returns the image and mask at the given index.

        Args:
            idx (int): Index of the image and mask.

        Returns:
            tuple: Tuple containing the image and mask.
        """
        image_id = self.ids[idx]
        image, mask = sitk.ReadImage(self.images[image_id]), sitk.ReadImage(self.masks[image_id])
        image, mask = sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(mask)
        image, mask = image[np.newaxis], mask[np.newaxis]
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


def main():
    dataset = Luna16Dataset(root=PROJECT_ROOT / "data" / "LUNA16", transforms=None)
    print(len(dataset))
    image, mask = dataset[0]
    print(type(image), type(mask))
    print(image.shape, mask.shape)
    print(image.dtype, mask.dtype)
    image = Image.fromarray(image[:, 70, :, :].squeeze().astype("uint8") * 255)
    mask = Image.fromarray(mask[:, 70, :, :].squeeze().astype("uint8") * 255)
    image.show()
    mask.show()


if __name__ == "__main__":
    main()
