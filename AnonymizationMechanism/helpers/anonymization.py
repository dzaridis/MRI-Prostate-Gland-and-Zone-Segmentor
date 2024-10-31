import os
from pydicom import dcmread
import os
from typing import List
from pydicom.uid import generate_uid
import uuid

def generate_patient_id():
    # Generate a UUID and get its hex representation without dashes
    hash_part = str(uuid.uuid4()).replace('-', '')[:8]  # Using first 8 characters
    return f"PCa-{hash_part}"

def find_dicom_files(root_dir):
    dicom_files = []
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.dcm'):
                full_path = os.path.join(dirpath, file)
                dicom_files.append(full_path)
    return dicom_files

def anonymize_dicom(dicom_paths: List[str], output_dir: str = "anonymized") -> None:
    """Anonymizes DICOM files according to the provided specification."""
    
    os.makedirs(output_dir, exist_ok=True)
    new_id = generate_patient_id()#str(generate_uid())
    sop_id = generate_uid()
    stud_id = generate_uid() 
    ser_id = generate_uid() 

    for idx, path in enumerate(dicom_paths):
        ds = dcmread(path)
        
        # Generate unique identifiers
        #new_id = str(generate_uid())
        
        # Critical anonymization - use string values, not PersonName
        if hasattr(ds, 'PatientName'):
            ds.PatientName = new_id
            
        if hasattr(ds, 'PatientID'):
            ds.PatientID = new_id  # Use same ID, but as string
            
        # Remove identifying elements
        meta_to_remove = {
            "StudyDate": "",
            "SeriesDate": "",
            "AcquisitionDate": "",
            "ContentDate": "",
            "AccessionNumber": "",
            "InstitutionName": "",
            "StudyDescription": "",
            "PatientBirthDate": "",
            "ProtocolName": "",
        }
        
        # Apply the changes
        for tag, value in meta_to_remove.items():
            if hasattr(ds, tag):
                setattr(ds, tag, value)
        
        # Remove private tags
        ds.remove_private_tags()
        
        # Update necessary UIDs
        ds.SOPInstanceUID = sop_id
        ds.SeriesInstanceUID = ser_id
        ds.StudyInstanceUID = stud_id
        
        # Set Patient Identity Removed
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "ProCAncer-I Whitelist"
        
        # Create directory structure based on DICOM tags
        structure_path = os.path.join(
            output_dir,
            str(ds.PatientID),
            str(ds.StudyInstanceUID),
            str(ds.SeriesInstanceUID)
        )
        os.makedirs(structure_path, exist_ok=True)
        
        # Save anonymized file with numbered name
        output_path = os.path.join(structure_path, f"image_{idx+1:03d}.dcm")
        ds.save_as(output_path)
# def anonymize_dicom(dicom_paths: List[str], output_dir: str = "anonymized") -> None:
#     """Anonymizes DICOM files according to the provided specification."""
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     for idx, path in enumerate(dicom_paths):
#         ds = dcmread(path)
        
#         # Critical anonymization
#         if hasattr(ds, 'PatientName'):
#             ds.PatientName = f"{generate_uid()}"
        
#         try:
#             if hasattr(ds, 'PatientID'):
#                 ds.PatientID = ds.PatientName
#         except:
#             ds.PatientID = f"{generate_uid()}"
            
#         # Remove identifying elements
#         meta_to_remove = {"study_date":ds.StudyDate, 
#         "series_date":ds.SeriesDate,
#         "acquisiton_date":ds.AcquisitionDate,
#         "content_date":ds.ContentDate,
#         "accession_number":ds.AccessionNumber,
#         "institution_name":ds.InstitutionName,
#         "study_description":ds.StudyDescription,
#         "patient_id":ds.PatientID,
#         "patient_name":ds.PatientName,
#         "patient_birth_date":ds.PatientBirthDate,
#         "protocol":ds.ProtocolName,
#         }

#         # Remove private tags
#         ds.remove_private_tags()
        
#         # Remove specified tags
#         for k,v in meta_to_remove.items():
#             if k=="patient_birth_date":
#                 v = v[:4]
#             else:
#                 v = ""

#         # Update necessary UIDs
#         ds.SOPInstanceUID = generate_uid()
#         ds.SeriesInstanceUID = generate_uid() 
#         ds.StudyInstanceUID = generate_uid()
        
#         # Set Patient Identity Removed
#         ds.PatientIdentityRemoved = "YES"
#         ds.DeidentificationMethod = "ProCAncer-I Whitelist"
        
#         # Create directory structure based on DICOM tags
#         structure_path = os.path.join(
#             output_dir,
#             str(ds.PatientID),
#             str(ds.StudyInstanceUID),
#             str(ds.SeriesInstanceUID)
#         )
#         os.makedirs(structure_path, exist_ok=True)
        
#         # Save anonymized file with numbered name
#         output_path = os.path.join(structure_path, f"image_{idx+1:03d}.dcm")
#         ds.save_as(output_path)