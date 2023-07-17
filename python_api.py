import sys
import os
from pathlib import Path
import time

import numpy as np
import nibabel as nib
import torch

from totalsegmentator.libs import setup_nnunet, download_pretrained_weights

def totalsegmentator(input, output, ml=False, nr_thr_resamp=1, nr_thr_saving=6,
                     fast=False, nora_tag="None", preview=False, task="total", roi_subset=None,
                     statistics=False, radiomics=False, crop_path=None, body_seg=False,
                     force_split=False, output_type="nifti", quiet=False, verbose=False, test=0):
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

    if task == "total":
        task_id = [251, 252, 253, 254, 255]
        resample = 1.5
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop = None
        model = "3d_fullres"
        folds = [0]
        
    crop_path = output if crop_path is None else crop_path
    download_pretrained_weights(task_id)
    folds = [0]  # None
    print("Bonjour bonjour bonjour")
    seg = nnUNet_predict_image(input, output, task_id, model=model, folds=folds,
                         trainer=trainer, tta=False, multilabel_image=ml, resample=resample,
                         crop=crop, crop_path=crop_path, task_name=task, nora_tag=nora_tag, preview=preview, 
                         nr_threads_resampling=nr_thr_resamp, nr_threads_saving=nr_thr_saving, 
                         force_split=force_split, crop_addon=crop_addon, roi_subset=roi_subset,
                         output_type=output_type, quiet=quiet, verbose=verbose, test=test)
    seg = seg.get_fdata().astype(np.uint8)