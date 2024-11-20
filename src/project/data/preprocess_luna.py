import logging
import os
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm

from project.config import PROJECT_ROOT


logger = logging.getLogger(__name__)


def make_mask(mask: np.ndarray, v_center: np.ndarray, v_diam: float, spacing: np.ndarray) -> None:
    """Update mask with nodule region."""
    v_diam_z = int(v_diam / spacing[2] + 1)
    v_diam_y = int(v_diam / spacing[1] + 1)
    v_diam_x = int(v_diam / spacing[0] + 1)
    v_diam_z = np.rint(v_diam_z / 2)
    v_diam_y = np.rint(v_diam_y / 2)
    v_diam_x = np.rint(v_diam_x / 2)
    z_min = int(v_center[0] - v_diam_z)
    z_max = int(v_center[0] + v_diam_z + 1)
    x_min = int(v_center[1] - v_diam_x)
    x_max = int(v_center[1] + v_diam_x + 1)
    y_min = int(v_center[2] - v_diam_y)
    y_max = int(v_center[2] + v_diam_y + 1)
    mask[z_min:z_max, x_min:x_max, y_min:y_max] = 1.0

    logger.info(f"Mask set: {z_max} - {z_min}, {x_max} - {x_min}, {y_max} - {y_min}")


def main():
    # Getting list of image files and save mask image files
    for subset_index in range(10):
        luna_path = PROJECT_ROOT / "data" / "LUNA16"
        output_path = PROJECT_ROOT / "data" / "LUNA16" / "mask"

        luna_subset_path = luna_path / ("subset" + str(subset_index))
        luna_subset_mask_path = output_path / ("subset" + str(subset_index))

        if not os.path.exists(luna_subset_mask_path):
            os.makedirs(luna_subset_mask_path)
        subset_file_list = glob(f"{luna_subset_path}/*.mhd")

        # The locations of the nodes
        df_node = pd.read_csv(luna_path / "annotations.csv")

        # Looping over the image files
        for img_file in tqdm(subset_file_list):
            file_uid = Path(img_file).stem
            mini_df = df_node[df_node["seriesuid"] == file_uid]
            itk_img = sitk.ReadImage(img_file)
            # indexes are z, y, x (notice the ordering)
            img_array = sitk.GetArrayFromImage(itk_img)

            # num_z height width constitute the transverse plane
            num_z, height, width = img_array.shape
            # x,y,z  Origin in world coordinates (mm)
            origin = np.array(itk_img.GetOrigin())
            # spacing of voxels in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())

            # some files may not have a nodule--skipping those
            if mini_df.shape[0] == 0:
                mask_itk = np.zeros(shape=(num_z, height, width), dtype=np.float32)
            if mini_df.shape[0] > 0:
                mask_itk = np.zeros(shape=(num_z, height, width), dtype=np.float32)
                # go through all nodes in one series image
                for node_idx, cur_row in mini_df.iterrows():
                    node_x, node_y, node_z = cur_row["coordX"], cur_row["coordY"], cur_row["coordZ"]
                    center = np.array([node_x, node_y, node_z])

                    v_center = np.rint((center - origin) / spacing)
                    v_diam = cur_row["diameter_mm"]

                    # convert x,y,z to z,y,x
                    v_center = v_center[[2, 1, 0]]
                    make_mask(mask_itk, v_center, v_diam, spacing)

            mask_itk = np.uint8(mask_itk * 255.)
            mask_itk = np.clip(mask_itk, 0, 255).astype('uint8')
            sitk_maskimg = sitk.GetImageFromArray(mask_itk)
            sitk_maskimg.SetSpacing(spacing)
            sitk_maskimg.SetOrigin(origin)
            sitk.WriteImage(sitk_maskimg, luna_subset_mask_path / f"{file_uid}_segmentation.mhd")


if __name__ == "__main__":
    main()
