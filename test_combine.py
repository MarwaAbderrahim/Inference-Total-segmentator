import io
import os
import contextlib
import sys
import time
import shutil
import zipfile
from pathlib import Path

import requests
import numpy as np
import nibabel as nib

from totalsegmentator.map_to_binary import class_map
from totalsegmentator.map_to_binary import class_map_5_parts

def combine_masks_to_multilabel_file(masks_dir, multilabel_file, class_type):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """


    if class_type == "ribs":
        masks = list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "vertebrae":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values())
    elif class_type == "vertebrae_ribs":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values()) + list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "lung":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "lung_left":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
    elif class_type == "lung_right":
        masks = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "heart":
        masks = ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
                 "heart_atrium_right", "heart_ventricle_right"]
    elif class_type == "pelvis":
        masks = ["femur_left", "femur_right", "hip_left", "hip_right"]
    elif class_type == "body":
        masks = ["body_trunc", "body_extremities"]


    masks_dir = Path(masks_dir)
    ref_img = nib.load(masks_dir / "liver.nii.gz")
    # masks = class_map["total"].values()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)
    print(masks)
    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
            print(f"{masks_dir}/{mask}.nii.gz")
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx+1

    return(img_out)



masks_dir=r"D:\test_total_segmentator\image1\image"
ref_img = nib.load(r"D:\test_total_segmentator\image1\image\liver.nii.gz")
multilabel_file=r"D:\test_total_segmentator\newwww.nii"
img_out=combine_masks_to_multilabel_file(masks_dir, multilabel_file, "pelvis")
print('done done')
new_img=nib.Nifti1Image(img_out, None)
print("done 1")
nib.save(new_img, multilabel_file)
print('done')