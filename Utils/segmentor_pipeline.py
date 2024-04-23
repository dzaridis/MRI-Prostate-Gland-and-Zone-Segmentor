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

def segmentor_pipeline_operation(output_volume:str, pats:dict):
    segmentor = Segmentor()
    segmentor.wg_model(pats)
    segmentor.preparation_zones(input_patients=pats)
    segmentor.zones_model()
    segmentor.post_process_zones(output_patient_folder=output_volume, pats=pats)
    segmentor.saving()
    segmentor.clean_workspace()

class Segmentor:
    def __init__(self):
        self.pats_for_wg_inference = None
        self.pats_for_wg = None
        self.wg_dict_original = None
        self.wg_dict_resampled = None
        self.pats_for_zones = None
        self.zones_original = None
        self.zones_resampled = None

    @staticmethod
    def preparation_wg(input_patients:dict):
        pats_for_wg = helpers.initial_processing(input_patients)
        return pats_for_wg
    
    def wg_model(self, input_patients:dict):
        self.pats_for_wg = self.preparation_wg(input_patients = input_patients)
        wg_nn = nnUnet_call.WGNNUnet(input_path="Dataset016_WgSegmentationPNetAndPicai", output_path="OutcomesWG")
        wg_nn.prediction()
        self.pats_for_wg_inference = wg_nn.return_paths(pats_for_wg=self.pats_for_wg)

    def preparation_zones(self, input_patients:dict):
        file_handling = helpers.ImageProcessorClass(base_output_path='Outputs', nnUNet_raw = nnUNet_raw)
        file_handling.process_images(self.pats_for_wg_inference, self.pats_for_wg, pats=input_patients)
        self.wg_dict_original, self.wg_dict_resampled = file_handling.get_paths()

    def zones_model(self):
        zones_nn = nnUnet_call.ZonesNNUnet(input_path="Dataset019_ProstateZonesSegmentationWgFilteredLessDilated", output_path="OutcomesZones")
        zones_nn.prediction()
        self.pats_for_zones = zones_nn.return_paths(pats_for_wg_inference=self.pats_for_wg_inference)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def post_process_zones(self, output_patient_folder:str, pats:dict):
        zone_handling = helpers.ZoneProcessor(output_patient_folder)
        zone_handling.process_zones(self.pats_for_zones, pats)
        self.zones_original, self.zones_resampled = zone_handling.get_paths()
    
    def saving(self):
        helpers.outputs_saving(self.wg_dict_original, self.zones_original, self.wg_dict_resampled, self.zones_resampled)

    def clean_workspace(self):
        renduntant = helpers.DeleteRedundantfiles()
        renduntant.clean_workspace_wg(self.pats_for_wg_inference)
        renduntant.clean_workspace_zones(self.pats_for_zones)
        renduntant.clean_patients_directory(os.path.join(nnUNet_raw, os.path.join("Dataset016_WgSegmentationPNetAndPicai", "ImagesTs")),
                                            os.path.join(nnUNet_raw, os.path.join("Dataset019_ProstateZonesSegmentationWgFilteredLessDilated", "ImagesTs")))
