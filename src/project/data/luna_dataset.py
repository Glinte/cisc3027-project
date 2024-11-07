"""
---
title: LUNA16 dataset for the U-Net experiment
summary: >
  LUNA16 dataset for the U-Net experiment.
---

# LUNA16 Dataset for the [U-Net](index.html) [experiment](experiment.html)

Ensure you have the LUNA16 dataset downloaded and structured correctly.
Save the training images inside `luna16/images` folder and the masks in `luna16/masks` folder.
"""

from torch import nn
from pathlib import Path
import torch.utils.data
import torchvision
from PIL import Image


class Luna16Dataset(torch.utils.data.Dataset):
    """
    LUNA16 Dataset for image segmentation.

    This dataset class loads CT images and their corresponding segmentation masks.

    Attributes:
        images (dict): Dictionary mapping image IDs to image file paths.
        masks (dict): Dictionary mapping mask IDs to mask file paths.
        ids (list): List of image IDs.
        transforms (torchvision.transforms.Compose): Transformations to apply to images and masks.
    """

    def __init__(self, image_path: Path, mask_path: Path):
        """
        Initializes the dataset.

        Args:
            image_path (Path): Path to the directory containing images.
            mask_path (Path): Path to the directory containing masks.
        """
        # Get a dictionary of images by id
        self.images = {p.stem: p for p in image_path.iterdir() if p.suffix in ['.png', '.jpg']}
        # Get a dictionary of masks by id
        self.masks = {p.stem: p for p in mask_path.iterdir() if p.suffix in ['.png', '.jpg']}

        # Image ids list
        self.ids = list(self.images.keys())

        # Transformations
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),  # Adjust size as necessary
            torchvision.transforms.ToTensor(),
        ])

    def __getitem__(self, idx: int):
        """
        Retrieves an image and its corresponding mask.

        Args:
            idx (int): Index of the image.

        Returns:
            tuple: A tuple containing the transformed image and the transformed mask.
        """
        # Get image id
        id_ = self.ids[idx]
        # Load image
        image = Image.open(self.images[id_]).convert('L')  # Load as grayscale
        # Transform image and convert it to a PyTorch tensor
        image = self.transforms(image)

        # Load mask
        mask = Image.open(self.masks[id_]).convert('L')  # Load as grayscale
        # Transform mask and convert it to a PyTorch tensor
        mask = self.transforms(mask)

        # Scale the mask appropriately if necessary (assuming binary masks)
        mask = mask / mask.max() if mask.max() > 0 else mask

        # Return the image and the mask
        return image, mask

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.ids)


# Testing code
if __name__ == '__main__':
    ds = Luna16Dataset(Path('luna16/images'), Path('luna16/masks'))
    print(f"Dataset size: {len(ds)}")
    sample_image, sample_mask = ds[0]
    print(f"Image shape: {sample_image.shape}, Mask shape: {sample_mask.shape}")
