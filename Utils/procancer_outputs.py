import os
import shutil

def cleanup_dicom_output(base_dir):
    """
    Moves prostate_zones.dcm files to the base directory and deletes the original directory structure.
    
    :param base_dir: The base directory containing the DICOM output structure
    """
    for root, dirs, files in os.walk(base_dir):
        if 'prostate_zones.dcm' in files:
            source_path = os.path.join(root, 'prostate_zones.dcm')
            dest_path = os.path.join(base_dir, f"prostate_zones.dcm")
            
            # Move the file
            shutil.move(source_path, dest_path)
            print(f"Moved {source_path} to {dest_path}")
            
            # Delete the original directory structure
            dir_to_delete = os.path.dirname(os.path.dirname(os.path.dirname(root)))
            if os.path.exists(dir_to_delete) and dir_to_delete != base_dir:
                shutil.rmtree(dir_to_delete)
                print(f"Deleted directory: {dir_to_delete}")


def cleanup_output_folder(output_path):
    """
    Cleans up the output folder, keeping only .dcm files in the root directory.
    Deletes all subdirectories and their contents, including .dcm files within them.

    :param output_path: Path to the output folder
    """
    # List items in the root directory
    root_items = os.listdir(output_path)

    # Delete all subdirectories and their contents
    for item in root_items:
        item_path = os.path.join(output_path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted directory and its contents: {item_path}")

    # Delete non-.dcm files in the root directory
    for item in root_items:
        item_path = os.path.join(output_path, item)
        if os.path.isfile(item_path) and not item.endswith('.dcm'):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")

    print("Cleanup completed.")

# if __name__ == "__main__":
#     base_directory = "output"
#     cleanup_dicom_output(base_directory)
#     print("Cleanup completed.")