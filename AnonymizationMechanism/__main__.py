import os
from helpers.anonymization import find_dicom_files, anonymize_dicom

if __name__ == '__main__':
    INPUT_DIR = os.environ.get('INPUT_DIR', '/app/data/input')
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/app/data/output')
    
    # Create anonymized directory inside input directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dicom_files = find_dicom_files(INPUT_DIR)
    anonymize_dicom(dicom_files, OUTPUT_DIR)