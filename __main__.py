import os
from Utils import InputCheck
from Utils import ImageProcessor
import torch
from Utils import helpers, nnUnet_call, segmentor_pipeline
import SimpleITK as sitk
from MedProIO import Coregistrator
import numpy as np
import json
import warnings
import multiprocessing
warnings.filterwarnings('ignore')
import logging

def run():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    INPUT_VOLUME = "Pats"
    OUTPUT_VOLUME = "Outputs"
    # Loads the files
    pats = InputCheck.load_nii_gz_files(INPUT_VOLUME) # loads the NIfTI files from the volume or path
    segmentor_pipeline.segmentor_pipeline_operation(output_volume=OUTPUT_VOLUME, pats=pats)

if __name__ == '__main__':
    # Initialize and start your multiprocessing objects here
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()