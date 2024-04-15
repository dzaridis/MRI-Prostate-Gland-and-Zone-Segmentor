import os
from Utils import InputCheck
from Utils import ImageProcessor
import torch
import helpers
import SimpleITK as sitk
from MedProIO import Coregistrator
import numpy as np
import json
import warnings
import multiprocessing
import nnUnet_call
warnings.filterwarnings('ignore')
import logging
nnUNet_raw = os.path.join("nnUnet_paths", "nnUNet_raw")

def outputs_saving(wg_dict_original:dict, 
                zones_original:dict,
                wg_dict_resampled:dict,
                zones_resampled:dict):
    """Saves the original and resampled to the original wg, tz and pz masks 

    Args:
        wg_dict_original (dict): dictionary with the paths
        zones_original (dict): dictionary with the paths
        wg_dict_resampled (dict): dictionary with the paths
        zones_resampled (dict): dictionary with the paths
    """
    for k,v in wg_dict_original.items():
        v.update(zones_original[k])
    for k,v in wg_dict_resampled.items():
        v.update(zones_resampled[k])

    with open(os.path.join("Outputs","ResampledToOriginalSegmentationPaths.json"), "w") as file:
        json.dump(wg_dict_resampled, file, indent=4)
    with open(os.path.join("Outputs","nnOutputSegmentationPaths.json"), "w") as file:
        json.dump(wg_dict_original, file, indent=4)

def initial_processing(pats:dict):
    """Performs image processing operations to prepare patients

    Args:
        pats (dict): Initial dict with patients

    Returns:
        dict: path to the processed patient to be ready for nnU-Net whole gland model
    """
    pats_for_wg = {}
    for key,val in pats.items():
        processed = ImageProcessor.ImageProcessing(val)
        pats_for_wg.update({key:processed})

    for k,v in pats_for_wg.items():
        sitk.WriteImage(v, os.path.join(nnUNet_raw, os.path.join(os.path.join('Dataset016_WgSegmentationPNetAndPicai', 'ImagesTs'), f"ProstateWG_{k}_0000.nii.gz")))   
    return pats_for_wg

def run():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    path = "Pats" # load the volume here
    pats = InputCheck.load_nii_gz_files(path) # loads the NIfTI files from the volume or path

    # prepares patients for the first nnUnet which is responsible for Whole Gland segmentation
    pats_for_wg = initial_processing(pats)

    # Performes Whole gland segmentation
    wg_nn = nnUnet_call.WGNNUnet(input_path="Dataset016_WgSegmentationPNetAndPicai", output_path="OutcomesWG")
    wg_nn.prediction()
    pats_for_wg_inference = wg_nn.return_paths(pats_for_wg=pats_for_wg)

    # Saves the Whole Gland and returns the saved paths to NIfTI files
    file_handling = helpers.ImageProcessorClass(base_output_path='Outputs', nnUNet_raw = nnUNet_raw)
    file_handling.process_images(pats_for_wg_inference, pats_for_wg)
    wg_dict_original, wg_dict_resampled = file_handling.get_paths()

    # Performes Transition & peripheral zonal segmentation
    zones_nn = nnUnet_call.ZonesNNUnet(input_path="Dataset019_ProstateZonesSegmentationWgFilteredLessDilated", output_path="OutcomesZones")
    zones_nn.prediction()
    pats_for_zones = zones_nn.return_paths(pats_for_wg_inference=pats_for_wg_inference)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Saves the Zonal segmentations and returns the saved paths to NIfTI files
    zone_handling = helpers.ZoneProcessor('Outputs')
    zone_handling.process_zones(pats_for_zones, pats)
    zones_original, zones_resampled = zone_handling.get_paths()

    # saves the dictionaries indexing the saved paths
    outputs_saving(wg_dict_original, zones_original, wg_dict_resampled, zones_resampled)

    # Deletes the renduntant files from the workspace
    renduntant = helpers.DeleteRedundantfiles()
    renduntant.clean_workspace_wg(pats_for_wg_inference)
    renduntant.clean_workspace_zones(pats_for_zones)
    renduntant.clean_patients_directory(os.path.join(nnUNet_raw, os.path.join("Dataset016_WgSegmentationPNetAndPicai", "ImagesTs")),
                                        os.path.join(nnUNet_raw, os.path.join("Dataset019_ProstateZonesSegmentationWgFilteredLessDilated", "ImagesTs")))

if __name__ == '__main__':
    # Initialize and start your multiprocessing objects here
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()