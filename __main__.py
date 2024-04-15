import os
from Utils import InputCheck
from Utils import ImageProcessor
import torch
from Utils import helpers, nnUnet_call
import SimpleITK as sitk
from MedProIO import Coregistrator
import numpy as np
import json
import warnings
import multiprocessing
warnings.filterwarnings('ignore')
import logging
nnUNet_raw = os.path.join("nnUnet_paths", "nnUNet_raw")

def run():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Loads the files
    path = "Pats" # load the volume here
    pats = InputCheck.load_nii_gz_files(path) # loads the NIfTI files from the volume or path

    # prepares patients for the first nnUnet which is responsible for Whole Gland segmentation
    pats_for_wg = helpers.initial_processing(pats)

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
    helpers.outputs_saving(wg_dict_original, zones_original, wg_dict_resampled, zones_resampled)

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