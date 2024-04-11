import os
import SimpleITK as sitk


def load_nii_gz_files(path):
    """
    Load .nii.gz files from a given path into a dictionary as SimpleITK 
    objects.
    The path can be either a directory containing .nii.gz files or a single 
    .nii.gz file.
    """
    files_to_load = []
    nii_gz_objects = {}

    # Check if path is a directory or a single file
    if os.path.isdir(path):
        # List all .nii.gz files in the directory
        for filename in os.listdir(path):
            if filename.endswith('.nii.gz'):
                files_to_load.append(os.path.join(path, filename))
    elif os.path.isfile(path) and path.endswith('.nii.gz'):
        files_to_load.append(path)
    else:
        raise ValueError("Provided path is neither a .nii.gz file nor a directory containing .nii.gz files.")

    # Load each file as a SimpleITK object
    for file_path in files_to_load:
        file_key = os.path.basename(file_path).split(".")[0]
        nii_gz_objects[file_key] = sitk.ReadImage(file_path)

    return nii_gz_objects