import os
import SimpleITK as sitk


def load_nii_gz_files(files_to_load):
    """
    Load .nii.gz files from a given path into a dictionary as SimpleITK 
    objects.
    The path can be either a directory containing .nii.gz files or a single 
    .nii.gz file.
    """

    nii_gz_objects = {}

    # Load each file as a SimpleITK object
    for file_path in files_to_load:
        file_key = os.path.basename(file_path).split(".")[0]
        nii_gz_objects[file_key] = sitk.ReadImage(file_path)

    return nii_gz_objects