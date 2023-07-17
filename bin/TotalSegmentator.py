#!/usr/bin/env python
import os
import io
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from contextlib import contextmanager

import warnings
sys.stdout =io.StringIO()
import argparse
from pkg_resources import require
from pathlib import Path
import numpy as np
import nibabel as nib 

from totalsegmentator.python_api import totalsegmentator
from DicomRTTool import DicomReaderWriter
import nibabel
import SimpleITK as sitk
import os
import time
from totalsegmentator.libs import combine_masks_to_multilabel_file
# from totalsegmentator.libs import combine_roi_to_multilabel_file

def main():

    output = io.StringIO()
    sys.stdout = output

    parser = argparse.ArgumentParser(description="Segment 104 anatomical structures in CT images.")

    parser.add_argument("-i", metavar="filepath", dest="input",
                        help="CT nifti image or folder of dicom slices", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                        help="Output directory for segmentation masks", 
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-ot", "--output_type", choices=["nifti", "dicom"],
                    help="Select if segmentations shall be saved as Nifti or as Dicom RT Struct image.",
                    default="nifti")
                    
    parser.add_argument("-ml", "--ml", action="store_true", help="Save one multilabel image for all classes",
                        default=False)

    parser.add_argument("-nr", "--nr_thr_resamp", type=int, help="Nr of threads for resampling", default=1)

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations", 
                        default=6)

    parser.add_argument("-f", "--fast", action="store_true", help="Run faster lower resolution model",
                        default=False)

    parser.add_argument("-t", "--nora_tag", type=str, 
                        help="tag in nora as mask. Pass nora project id as argument.",
                        default="None")

    parser.add_argument("-p", "--preview", action="store_true", 
                        help="Generate a png preview of segmentation",
                        default=False)

    parser.add_argument("-ta", "--task", choices=["total", "lung_vessels", "cerebral_bleed", 
                        "hip_implant", "coronary_arteries", "body", "pleural_pericard_effusion", 
                        "liver_vessels", "bones_extremities", "tissue_types",
                        "heartchambers_highres", "head", "aortic_branches", "heartchambers_test", 
                        "bones_tissue_test", "test"],
                        help="Select which model to use. This determines what is predicted.",
                        default="total")

    parser.add_argument("-rs", "--roi_subset", type=str, nargs="+", default="None",
                        help="Define a subset of classes to save (space separated list of class names). If running 1.5mm model, will only run the appropriate models for these rois.")

    parser.add_argument("-cp", "--crop_path", help="Custom path to masks used for cropping. If not set will use output directory.", 
                        type=lambda p: Path(p).absolute(), default=None)

    parser.add_argument("-bs", "--body_seg", action="store_true", 
                        help="Do initial rough body segmentation and crop image to body region",
                        default=False)
    
    parser.add_argument("-fs", "--force_split", action="store_true", help="Process image in 3 chunks for less memory consumption",
                        default=False)

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    parser.add_argument("--test", metavar="0|1|2|3", choices=[0, 1, 2, 3], type=int,
                        help="Only needed for unittesting.",
                        default=0)

    parser.add_argument('--version', action='version', version=require("TotalSegmentator")[0].version)

    parser.add_argument("-mlabels", "--mlabels", action="store_true", help="Save one multilabel image for all classes",
                         default=True)
    
    # parser.add_argument("-roi", "--roi", help="Save one roi", choices=["ribs", "vertebrae", "vertebrae_ribs",
    #                              "lung", "heart", "pelvis"], default=False)

    args = parser.parse_args()

    #-------------------------------------- Convert dicom to nifti----------------------------------------------

    cwd = os.getcwd() 
    in_path_dicom_nifti=args.input
    image=cwd + "/" + "raw_data_Dicom2Nii.nii.gz"
    output_nifti = cwd + "/" + "All_calsses"
    #Initialize the reader
    reader = DicomReaderWriter()

    # Provide a path through which the reader should search for DICOM
    reader.walk_through_folders(in_path_dicom_nifti)

    #  Load the images
    reader.get_images()

    # Write .nii images
    sitk.WriteImage(reader.dicom_handle,  image)

    # path of the .nii image
    input=image 
    output_dcm=args.output

    sys.stdout = sys.__stdout__


    #-------------------------------------- Générer le masque ----------------------------------------------

    totalsegmentator(input, output_dcm, args.ml, args.nora_tag, args.preview, args.task, args.roi_subset,
                    args.output_type, args.verbose)
    
    #-------------------------------------- Convert nitfti to dicom ----------------------------------------------

    # output_value = output.getvalue()
    # print(output_value)
    
    sys.stdout = sys.__stdout__

    # sys.stdout = io.StringIO()
    
    multilabel_file = cwd + "/" + "multilabel.nii.gz"
    roi_file = cwd + "/" + "ROI_mask.nii.gz"


    if args.mlabels : 
         combine_masks_to_multilabel_file(output_dcm, multilabel_file)

    # if args.roi : 
    #     combine_roi_to_multilabel_file(output_nifti, roi_file, args.roi)

    # def writeSlices(series_tag_values, new_img, i, out_dir):
    #         image_slice = new_img[:,:,i]
    #         writer = sitk.ImageFileWriter()
    #         writer.KeepOriginalImageUIDOn()

    #         # Tags shared by the series.
    #         list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    #         # Slice specific tags.
    #         image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    #         image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

    #         # Setting the type to CT preserves the slice location.
    #         image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

    #         # (0020, 0032) image position patient determines the 3D spacing between slices.
    #         image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    #         image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

    #         # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    #         writer.SetFileName(os.path.join(out_dir,'slice' + str(i).zfill(4) + '.dcm'))
    #         writer.Execute(image_slice)
    # sys.stdout = open(os.devnull, 'w')

    # def convert_nifti_to_dicom(nifti_dir, out_dir):
    #     pixel_data = sitk.sitkUInt16
    #     new_img = sitk.ReadImage(nifti_dir, pixel_data) 
    #     modification_time = time.strftime("%H%M%S")
    #     modification_date = time.strftime("%Y%m%d")
    #     direction = new_img.GetDirection()
    #     series_tag_values = [("0008|0031",modification_time), # Series Time
    #                     ("0008|0021",modification_date), # Series Date
    #                     ("0008|0008","DERIVED\\SECONDARY"), # Image Type
    #                     ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
    #                     ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
    #                                                         direction[1],direction[4],direction[7])))),
    #                     ("0008|103e", "Created-SimpleITK")] # Series Description

    #     # Write slices to output directory
    #     list(map(lambda i: writeSlices(series_tag_values, new_img, i, out_dir), range(new_img.GetDepth())))
    
    # output_dcm=args.output
    # os.mkdir(output_dcm)

    # # log_file = 'dicomrttool.log'
    # # sys.stdout = open(log_file, 'w')

    # convert_nifti_to_dicom(multilabel_file, output_dcm)

    # sys.stdout = sys.__stdout__

    # Dice similarity function
    # def dice(pred, true, k = 1):
    #     # intersection = np.sum(pred[true==k]) * 2.0
    #     pred=nib.load(pred).get_fdata()
    #     true=nib.load(true).get_fdata()
    #     intersection = np.sum(pred[true==k]==k) * 2.0
    #     # dice = intersection / (np.sum(pred) + np.sum(true))
        
        
    #     dice = intersection /(np.sum(pred[pred==k]==k) + np.sum(true[true==k]==k))
    #     return dice
    
    # y_true=os.path.dirname(in_path_dicom_nifti)+ "/mask.nii.gz"
    # dice_score = dice(multilabel_file, y_true, k = 104) 
    
    # print ("Dice Similarity: {}".format(dice_score))

if __name__ == "__main__":
    main()


