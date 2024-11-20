import SimpleITK as sitk

import numpy as np
import os
import glob

def generate_spherical_mask(shape, center, radius):
    """
    生成球形掩码
    """
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    return (dist_from_center <= radius).astype(np.uint8)

def preprocess_and_save(mhd_folder, save_folder, annotations_csv):
    """
    预处理 LUNA16 数据并保存为 .npy 文件
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 加载标注
    annotations = np.genfromtxt(annotations_csv, delimiter=',', skip_header=1, dtype=str)

    # 处理每个 .mhd 文件
    mhd_files = glob.glob(os.path.join(mhd_folder, "*.mhd"))
    for mhd_file in mhd_files:
        # 加载 .mhd 文件
        image = sitk.ReadImage(mhd_file)
        image_array = sitk.GetArrayFromImage(image)  # 转换为 numpy 数组
        seriesuid = os.path.basename(mhd_file).replace(".mhd", "")

        # 保存图像数据
        np.save(os.path.join(save_folder, f"{seriesuid}_image.npy"), image_array)

        # 提取对应标注信息并保存掩码
        mask = np.zeros_like(image_array, dtype=np.uint8)
        for annotation in annotations:
            if annotation[0] == seriesuid:
                x, y, z, diameter = map(float, annotation[1:])
                center = [z, y, x]
                radius = diameter / 2
                mask += generate_spherical_mask(mask.shape, center, radius)
        np.save(os.path.join(save_folder, f"{seriesuid}_mask.npy"), mask)

if __name__ == "__main__":
    preprocess_and_save(
        mhd_folder="raw_data/subset0",  # 原始数据路径
        save_folder="processed_data/train",  # 保存路径
        annotations_csv="annotations.csv"  # 标注文件路径
    )
