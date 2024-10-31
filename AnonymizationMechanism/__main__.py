import os
from helpers.anonymization import find_dicom_files, anonymize_dicom

if __name__ == '__main__':
    data_dir = os.environ.get('DATA_DIR', 'Pats')
    
    # Create anonymized directory inside input directory
    anonymized_dir = os.path.join(data_dir, 'anonymized')
    
    dicom_files = find_dicom_files(data_dir)
    anonymize_dicom(dicom_files, anonymized_dir)