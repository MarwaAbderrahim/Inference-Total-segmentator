import sys
import os
from pathlib import Path
import time

import numpy as np
import nibabel as nib
import torch

from totalsegmentator.libs import setup_nnunet, download_pretrained_weights

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def totalsegmentator(input, output, ml=False, task="total", roi_subset=None,
                     crop_path=None, body_seg=False, force_split=False, output_type="nifti"):
    """
    Run TotalSegmentator from within python. 

    For explanation of the arguments see description of command line 
    arguments in bin/TotalSegmentator.
    """

    if not torch.cuda.is_available():
        print("No GPU detected. Running on CPU. This can be very slow. The '--fast' option can help to some extend.")

    setup_nnunet()

    from totalsegmentator.nnunet import nnUNet_predict_image  # this has to be after setting new env vars

    crop_addon = [3, 3, 3]  # default value

<<<<<<< HEAD
    task == "total"
    task_id = [251, 252, 253, 254, 255]
    resample = 1.5
    trainer = "nnUNetTrainerV2_ep4000_nomirror"
    crop = None
    model = "3d_fullres"
    folds = [0]
=======
    if task == "total":
        task_id = [251, 252, 253, 254, 255]
        resample = 1.5
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop = None
        model = "3d_fullres"
        folds = [0]
>>>>>>> origin/main
    crop_path = output if crop_path is None else crop_path
    download_pretrained_weights(task_id)

    # Generate rough body segmentation (speedup for big images; not useful in combination with --fast option)
    if task == "total" and body_seg:
        download_pretrained_weights(269)
        st = time.time()
        body_seg = nnUNet_predict_image(input, None, 269, model="3d_fullres", folds=[0],
                            trainer="nnUNetTrainerV2", tta=False, multilabel_image=True, resample=6.0,
                            crop=None, crop_path=None, task_name="body", save_binary=True, 
                            crop_addon=crop_addon,  test=0)
        crop = body_seg
 
    folds = [0]  # None
    seg = nnUNet_predict_image(input, output, task_id, model=model, folds=folds,
                         trainer=trainer, tta=False, multilabel_image=ml, resample=resample,
                         crop=crop, crop_path=crop_path, task_name=task,
                         force_split=force_split, crop_addon=crop_addon, roi_subset=roi_subset,
                         output_type=output_type )
    seg = seg.get_fdata().astype(np.uint8)

<<<<<<< HEAD
=======

>>>>>>> origin/main
    
