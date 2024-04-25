import os
from Utils import InputCheck
#from flask import Flask, request, jsonify, render_template, jsonify, redirect, url_for
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
import tkinter as tk
from tkinter import filedialog

def select_folder(prompt):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_path

def run_process(input_folder, output_folder): #
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    INPUT_VOLUME = input_folder #"Pats"
    OUTPUT_VOLUME = output_folder # "Outputs"
    pats = InputCheck.load_nii_gz_files(INPUT_VOLUME) # loads files
    try:
        segmentor_pipeline.segmentor_pipeline_operation(output_volume=OUTPUT_VOLUME, pats=pats) # perform segmentation operations
    except Exception as e:
        with open(os.path.join(OUTPUT_VOLUME,'error_log.txt'), 'a') as f:  # Open file in append mode
            f.write(f"An error occurred: {str(e)}\n")
            pass
    try:
        helpers.process_masks(out_volume=OUTPUT_VOLUME) # post process masks, filling holes for WG and TZ when necessary
    except Exception as e:
        print ("ERROR IN THE POST PROCESS OF WG MASK, CHECK SEGMENTATION",e)
        pass
    print("Processing completed successfully!")

if __name__ == '__main__':
    input_folder = select_folder("Select Input Folder")
    output_folder = select_folder("Select Output Folder")

    if input_folder and output_folder:
        # Here you might want to confirm the selections or start processing immediately
        print(f"Selected input folder: {input_folder}")
        print(f"Selected output folder: {output_folder}")
        
        # Start processing
        process = multiprocessing.Process(target=run_process, args=(input_folder, output_folder)) #, 
        process.start()
        process.join()
    else:
        print("No folders selected. Exiting application.")