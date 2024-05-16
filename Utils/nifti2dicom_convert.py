'''Convert all nifti images (except if source dicom exists) to dicom'''
import os
import shutil
import yaml
from pydicom import dcmread
from Utils.nifti2dicomseg import nifti2dicomseg
from Utils.nifti2dicom import nifti2dicom

with open("Pats/patient_dict.yaml","r",encoding="utf-8") as yfile:
    PATIENT_DICT = yaml.safe_load( yfile )

SEG_OUTPUT = "Outputs"
CATEGORY = "Resampled"


def converter():
    '''nifti to dicom'''
    for key,value in PATIENT_DICT.items():

        if value["source_type"] == "nii.gz":

            nii_path = value["destination_nifti"]
            t2_path = nifti2dicom(nii_path)

            out_location = os.path.join( *nii_path.split('/')[1::])
            out_location = os.path.join( out_location.split('.nii.gz')[0] )

            seg_path = os.path.join(
                SEG_OUTPUT,
                out_location,
                CATEGORY
            )

            nifti2dicomseg(seg_path, copy_t2)
            nifti2dicomseg(seg_path, copy_t2,'wg')
            nifti2dicomseg(seg_path, copy_t2,'pz')
            nifti2dicomseg(seg_path, copy_t2,'tz')

        if value["source_type"] == "dcm":

            dcm_path = key
            temp_list = [x for x in os.listdir(key) if x.endswith(".dcm")]

            temp_dcm = dcmread(
                os.path.join(key,temp_list[0]
                ), stop_before_pixels= True
            )

            pat_id = temp_dcm.PatientID
            study_id = temp_dcm.StudyInstanceUID

            copy_t2 = os.path.join(
                SEG_OUTPUT,
                pat_id,
                study_id,
                "t2w"
            )

            os.makedirs(copy_t2, exist_ok=True)

            shutil.copytree(key, copy_t2, dirs_exist_ok=True)

            out_location = os.path.join( *dcm_path.split('/')[1::])
            out_location = os.path.join( out_location.split('.')[0] ).replace(os.sep, "_")

            seg_path = os.path.join(
                SEG_OUTPUT,
                out_location,
                CATEGORY
            )

            nifti2dicomseg(seg_path, copy_t2)
            nifti2dicomseg(seg_path, copy_t2,'wg')
            nifti2dicomseg(seg_path, copy_t2,'pz')
            nifti2dicomseg(seg_path, copy_t2,'tz')
