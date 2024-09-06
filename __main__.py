'''
Main component for prostate zone (WG,PZ,TZ) segmentation, data transformation(dcm->nii, nii->dcm),
and uploading results to an orthanc server + ohif viewer server.
'''
import os
import shutil
import warnings
import multiprocessing
import logging
from Utils import helpers, segmentor_pipeline, InputCheck
from Utils.get_images import get_images
from Utils.nifti2dicom_convert import converter
from Utils.ImportDicomFiles import upload
warnings.filterwarnings('ignore')

INPUT_VOLUME = "Pats"
OUTPUT_VOLUME = "Outputs"

def run_process(): #input_folder, output_folder
    ''' Creates zone segmentation for the given data via trained NNUnet '''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    pats = InputCheck.load_nii_gz_files(INPUT_VOLUME) # loads files

    try:
        # perform segmentation operations
        segmentor_pipeline.segmentor_pipeline_operation(
            output_volume=OUTPUT_VOLUME, pats=pats
        )

    except Exception as e:
         # Open file in append mode
        with open(os.path.join(OUTPUT_VOLUME,'error_log.txt'), 'a') as f: 
            f.write(f"An error occurred: {str(e)}\n")

    try:
        # post process masks, filling holes for WG and TZ when necessary
        helpers.process_masks(out_volume=OUTPUT_VOLUME) 

    except Exception as e:
        print ("ERROR IN THE POST PROCESS OF WG MASK, CHECK SEGMENTATION",e)

if __name__ == '__main__':

    process = multiprocessing.Process(
        target=run_process
    )
    process.start()
    process.join()