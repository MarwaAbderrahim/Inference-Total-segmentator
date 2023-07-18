import sys
import os
import itertools
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from fury import window, actor, ui, io, utils

from totalsegmentator.vtk_utils import contour_from_roi_smooth, plot_mask
from totalsegmentator.map_to_binary import class_map


np.random.seed(1234)
random_colors = np.random.rand(100, 4)

roi_groups = {
    "total": [
        ['humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 'clavicula_left',
         'clavicula_right', 'femur_left', 'femur_right', 'hip_left', 'hip_right', 'sacrum',
         'colon', 'trachea'],
        ['spleen', 'kidney_right', 'kidney_left', 'gallbladder',
         'adrenal_gland_right', 'adrenal_gland_left',
         'gluteus_medius_left', 'gluteus_medius_right',
         'heart_atrium_left', 'heart_atrium_right', 'heart_myocardium'
         ],
        ['iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right',
         'aorta', 'inferior_vena_cava',
         'portal_vein_and_splenic_vein', 'esophagus'],
        ['small_bowel', 'stomach', 'lung_upper_lobe_left',
         'lung_upper_lobe_right', 'face'],
        ['lung_lower_lobe_left', 'lung_middle_lobe_right', 'lung_lower_lobe_right',
         'pancreas', 'brain'],
        ['vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2',
         'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9',
         'vertebrae_T8', 'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4',
         'vertebrae_T3', 'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6',
         'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2', 'vertebrae_C1',
         'gluteus_maximus_left', 'gluteus_maximus_right'],
        ['rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6',
         'rib_left_7', 'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12',
         'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6',
         'rib_right_7', 'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11',
         'rib_right_12', 'urinary_bladder', 'duodenum',
         'gluteus_minimus_left', 'gluteus_minimus_right'],
        ['liver', 'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right',
         'heart_ventricle_left', 'heart_ventricle_right', 'pulmonary_artery']
    ],
}
