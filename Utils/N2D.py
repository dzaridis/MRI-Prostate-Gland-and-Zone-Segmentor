import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom import Dataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, generate_uid, MRImageStorage
import os
import json
from Utils.ImportDicomFiles import upload

def post_orthanc(file):
    
    try:
        upload(filename=file)
    except:
        pass


def convert_nifti_to_dicom(nifti_path, output_folder, parent_folder, patient_id, study_instance_uid, series_instance_uid, accession_number, series_descriptor):
    # Read the NIfTI file
    # Load NIfTI file
    nii = sitk.ReadImage(nifti_path)
    data = sitk.GetArrayFromImage(nii)
    data = np.transpose(data, (1, 2, 0))  # Adjust based on your specific data orientation

    # Generate DICOM file for each slice
    for i in range(data.shape[2]):
        ds = Dataset()
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.InstanceNumber = str(i + 1)
        
        # File meta information
        ds.file_meta = Dataset()
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.ImplementationClassUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # Essential DICOM tags
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.StudyDate = '20240101'  # Use appropriate date format
        ds.SeriesDate = '20240101'  # Use appropriate date format
        ds.Modality = 'SEG'
        if "segm" not in series_descriptor:
            ds.Modality = 'MR'
        ds.Manufacturer = 'Manufacturer'  # Optional: specify if known

        ds.SeriesDescription = series_descriptor
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.Rows, ds.Columns = data.shape[0], data.shape[1]
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelSpacing = [nii.GetSpacing()[0], nii.GetSpacing()[1]]  # Adjust as necessary
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SliceThickness = nii.GetSpacing()[2]

        # Pixel data
        slice_data = data[:, :, i].astype(np.uint16)
        ds.PixelData = slice_data.tobytes()

        # Save DICOM file
        filename = os.path.join(output_folder, f'slice_{i:04d}.dcm')
        # fln = os.path.join(parent_folder,  f'{study_instance_uid}_{series_instance_uid}_slice_{i:04d}.dcm')
        pydicom.dcmwrite(filename, ds, write_like_original=False)
        # pydicom.dcmwrite(fln, ds, write_like_original=False)
        # Send file to server
        # post_orthanc( filename )


def converter():
    pth = os.path.join("Outputs","ResampledToOriginalSegmentationPaths.json")
    with open(pth, "r") as f:
        segme_path = json.load(f)
    
    for k,v in segme_path.items():
        pat = os.path.join("Pats", f"{k}.nii.gz")
        study_instance_uid = "1.2.826.0.1.3680043.2.1125.1.22155955583062784993477656808786942"#generate_uid()
        accession_number = "1.2.826.0.1.3680043.8.498.34252221578263319278033248469323077523"#generate_uid()
        series_instance_uid_original = generate_uid()
        series_instance_uid_wg = generate_uid()
        series_instance_uid_pz = generate_uid()
        series_instance_uid_tz = generate_uid()
        wg_segm = v["wg_binary"]
        pz_segm = v["pz_binary"]
        tz_segm = v["tz_binary"]

        os.makedirs(os.path.join("dicom_outputs",k), exist_ok= True)

        save_folder = os.path.join("dicom_outputs",k)

        for key in ["t2_weighted", "prostate", "peripheral","transition"]:
            os.makedirs(os.path.join(save_folder, key), exist_ok= True)

        save_folder_or = os.path.join(save_folder,"t2_weighted")
        save_folder_wg = os.path.join(save_folder,"prostate")
        save_folder_pz = os.path.join(save_folder,"peripheral")
        save_folder_tz = os.path.join(save_folder,"transition")
        convert_nifti_to_dicom(pat, save_folder_or, patient_id=k, parent_folder="dicom_outputs", study_instance_uid=study_instance_uid,series_instance_uid = series_instance_uid_original, accession_number=accession_number, series_descriptor="t2-weighted")
        convert_nifti_to_dicom(wg_segm, save_folder_wg, patient_id=k, parent_folder="dicom_outputs",study_instance_uid=study_instance_uid,series_instance_uid =series_instance_uid_wg, accession_number=accession_number, series_descriptor="wg-segm")
        convert_nifti_to_dicom(pz_segm, save_folder_pz, patient_id=k, parent_folder="dicom_outputs",study_instance_uid=study_instance_uid,series_instance_uid =series_instance_uid_pz, accession_number=accession_number, series_descriptor="pz-segm")
        convert_nifti_to_dicom(tz_segm, save_folder_tz, patient_id=k, parent_folder="dicom_outputs",study_instance_uid=study_instance_uid,series_instance_uid = series_instance_uid_tz, accession_number=accession_number, series_descriptor="tz-segm")
