from typing import Callable, Any

from pathlib import Path
import torch.utils.data
import SimpleITK as sitk
from PIL import Image
from jaxtyping import Num
from matplotlib import pyplot as plt

from project.types import Array


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

    images_path = ["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7", "subset8", "subset9"]
    masks_path = ["seg-lungs-LUNA16"]

    def __init__(self, root: str | Path, transforms: Callable | None = None):
        """
        Initializes the dataset.

        Args:
            root: Root directory of dataset.
            transforms: Transformations to apply to images and masks.
        """
        self.root = Path(root)
        self.transforms = transforms

        self.images = {}
        self.masks = {}
        self.ids = []

        for path in self.images_path:
            file: Path
            for file in (self.root / path).iterdir():
                if file.suffix == ".mhd":
                    self.images[file.stem] = file

        for path in self.masks_path:
            file: Path
            for file in (self.root / path).iterdir():
                if file.suffix == ".mhd":
                    self.masks[file.stem] = file

        self.ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[Num[Array, "..."], Num[Array, "..."]]:
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

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


def main():
    dataset = Luna16Dataset(root=PROJEECT_ROOT / "data" / "LUNA16", transforms=None)
    print(len(dataset))
    image, mask = dataset[0]
    print(type(image), type(mask))
    print(image.shape, mask.shape)
    print(image.dtype, mask.dtype)
    print(image.tolist())
    image = Image.fromarray(mask[70, :, :].astype("uint8") * 255)
    image.show()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
