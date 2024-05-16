''' 
Read dicom directories and nii.gz files from given parent directory.
Converts dcm images to nii.gz files.
Ignores directories with 1 dcm file. Multi-frame not supported.
'''
import os
from shutil import rmtree
from typing import List, Tuple, Dict
from pathlib import Path
from warnings import warn
import yaml
from SimpleITK import ImageSeriesReader
from SimpleITK import WriteImage
from SimpleITK import ImageSeriesReader_GetGDCMSeriesFileNames as GetGDCMSeriesFileNames

def read_dcm_images( image_list: Tuple[Path] ):
    ''' A simple conversion from dcm dataset to sitk object'''

    reader= ImageSeriesReader()
    reader.SetFileNames(image_list)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    itk_image = reader.Execute()

    for i in reader.GetMetaDataKeys(0):
        if "ITK" not in i:

            tag_value = reader.GetMetaData(0,i)
        try:
            itk_image.SetMetaData(i, tag_value)
        except TypeError:
            pass

    return itk_image

def convert_dicoms(dir_path = List[Path]) -> Dict[str,str]:
    ''' 
    Read available dicom directories and stored them in .nii.gz
    '''

    if not dir_path:
        print( "No dicom files found. Exit!")

    dicom_dict = {}
    for dcm_path in dir_path:

        path_split = dcm_path.split(os.sep)
        input_dir = path_split[0]
        output_name = "_".join(path_split[1:])+".nii.gz"

        os.makedirs(os.path.join(input_dir,"_gen_dicom2nifti"), exist_ok= True )

        destination = os.path.join(input_dir, "_gen_dicom2nifti", output_name)

        spatial_ordered_dcm_files = GetGDCMSeriesFileNames(dcm_path)

        if len(spatial_ordered_dcm_files) == 1:
            warn(f"{dcm_path} has one dcm. Multi-frame dicom files are not supported. Skip!")
            continue

        itk_img = read_dcm_images( spatial_ordered_dcm_files )

        WriteImage( itk_img, destination)

        dicom_dict[dcm_path] = {
            "destination_nifti": destination,
            "source_type": "dcm"
        }

    return dicom_dict

def get_images(input_dir:Path)-> list:
    ''' 
    Read .dcm and .nii.gz files inside the given parent directory.
    dcm files are separated by directory and converted to nifti images.
    '''

    dcm_dirs = []
    nii_files = []

    # Delete directory with generated niftis if already exist
    destination = os.path.join(input_dir, "_gen_dicom2nifti")
    if os.path.exists(destination):
        rmtree(destination)

    for dirpath,_,files in os.walk(input_dir):

        for file in files:

            if file.endswith(".nii.gz"):

                relative_path = os.path.join( dirpath, file)

                nii_files.append( relative_path )

            if file.endswith(".dcm"):

                if dirpath not in dcm_dirs:
                    dcm_dirs.append(dirpath)
                continue

    if len(dcm_dirs) + len(nii_files) == 0:
        raise AttributeError("No .nii.gz or .dcm file was found")

    dicom_files = convert_dicoms ( dcm_dirs )

    patient_dict = {}

    for nii in nii_files:
        patient_dict[nii] = {
            "destination_nifti": nii,
            "source_type": "nii.gz"
        }

    patient_dict.update(dicom_files)

    with open( os.path.join(input_dir,'patient_dict.yaml'), "w", encoding= "utf-8") as yfile:
        yaml.safe_dump(patient_dict, yfile, indent=4, sort_keys=False)

    patient_list = [ value["destination_nifti"] for value in patient_dict.values() ]

    return patient_list

if __name__ == "__main__":

    get_images("Pats")
