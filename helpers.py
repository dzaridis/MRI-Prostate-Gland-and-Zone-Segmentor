import os
import numpy as np
import SimpleITK as sitk
import logging
from Utils import ImageProcessor

class ImageProcessorClass:
    def __init__(self, base_output_path, nnUNet_raw):
        self.base_output_path = base_output_path
        self.nnUNet_raw = nnUNet_raw
        self.wg_dict_original = {}
        self.wg_dict_resampled = {}
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def create_directories(self, key):
        try:
            os.makedirs(os.path.join(self.base_output_path, key, "Resampled"), exist_ok=True)
            os.makedirs(os.path.join(self.base_output_path, key, "Original"), exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directories for {key}: {e}")
            raise

    def process_images(self, pats_for_wg_inference, pats_for_wg):
        for key, val in pats_for_wg_inference.items():
            try:
                wg_binary = sitk.ReadImage(val["binary"])
                self.create_directories(key)
                wg_binary = ImageProcessor.process_mask(wg_binary)
                wg_binary = ImageProcessor.remove_small_components(wg_binary)

                filtered_ser = ImageProcessor.filter_ser(pats_for_wg[key], ImageProcessor.mask_dilation(wg_binary))
                probs = np.load(val["probs"])["probabilities"]
                wg_probs = probs[1, :, :, :]
                wg_probs = sitk.GetImageFromArray(wg_probs)
                wg_probs.CopyInformation(wg_binary)

                wg_binary_resampled = sitk.Resample(wg_binary, pats_for_wg[key], sitk.Transform(), sitk.sitkNearestNeighbor)
                wg_probs_resampled = sitk.Resample(wg_probs, pats_for_wg[key], sitk.Transform(), sitk.sitkNearestNeighbor)

                output_paths = {
                    "Original": {
                        "wg_binary": os.path.join(self.base_output_path, key, "Original", "wg_binary.nii.gz"),
                        "wg_probs": os.path.join(self.base_output_path, key, "Original", "wg_probs.nii.gz")
                    },
                    "Resampled": {
                        "wg_binary": os.path.join(self.base_output_path, key, "Resampled", "wg_binary.nii.gz"),
                        "wg_probs": os.path.join(self.base_output_path, key, "Resampled", "wg_probs.nii.gz")
                    }
                }

                self.write_image(wg_binary_resampled, output_paths["Resampled"]["wg_binary"])
                self.write_image(wg_probs_resampled, output_paths["Resampled"]["wg_probs"])
                self.write_image(wg_binary, output_paths["Original"]["wg_binary"])
                self.write_image(wg_probs, output_paths["Original"]["wg_probs"])

                self.wg_dict_original[key] = output_paths["Original"]
                self.wg_dict_resampled[key] = output_paths["Resampled"]

            except Exception as e:
                logging.error(f"Error processing {key}: {e}")

    def write_image(self, image, path):
        try:
            sitk.WriteImage(image, path)
        except Exception as e:
            logging.error(f"Error writing image to {path}: {e}")
    
    def get_paths(self):
        return self.wg_dict_original, self.wg_dict_resampled

class ZoneProcessor:
    def __init__(self, base_output_path):
        self.base_output_path = base_output_path
        self.resampled = {}
        self.original = {}
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def create_directories(self, key):
        try:
            os.makedirs(os.path.join(self.base_output_path, key, "Resampled"), exist_ok=True)
            os.makedirs(os.path.join(self.base_output_path, key, "Original"), exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directories for {key}: {e}")
            raise

    def process_zones(self, pats_for_zones, pats):
        for key, val in pats_for_zones.items():
            try:
                zones = sitk.ReadImage(val["binary"])
                tz_binary, pz_binary = ImageProcessor.create_binary_masks(zones)
                tz_binary = ImageProcessor.process_mask(tz_binary)
                pz_binary = ImageProcessor.process_mask(pz_binary)
                tz_binary = ImageProcessor.remove_small_components(tz_binary)
                pz_binary = ImageProcessor.remove_small_components(pz_binary)
                
                probs = np.load(val["probs"])["probabilities"]
                tz, pz = probs[1,:,:,:], probs[2,:,:,:]
                tz = sitk.GetImageFromArray(tz)
                pz = sitk.GetImageFromArray(pz)
                tz.CopyInformation(tz_binary)
                pz.CopyInformation(pz_binary)

                self.create_directories(key)

                resampled_paths = {
                    "tz_binary": os.path.join("Outputs", key, "Resampled", "tz_binary.nii.gz"),
                    "tz_probs": os.path.join("Outputs", key, "Resampled", "tz_probs.nii.gz"),
                    "pz_binary": os.path.join("Outputs", key, "Resampled", "pz_binary.nii.gz"),
                    "pz_probs": os.path.join("Outputs", key, "Resampled", "pz_probs.nii.gz")
                }

                original_paths = {
                    "tz_binary": os.path.join("Outputs", key, "Original", "tz_binary.nii.gz"),
                    "tz_probs": os.path.join("Outputs", key, "Original", "tz_probs.nii.gz"),
                    "pz_binary": os.path.join("Outputs", key, "Original", "pz_binary.nii.gz"),
                    "pz_probs": os.path.join("Outputs", key, "Original", "pz_probs.nii.gz")
                }

                self.write_image(sitk.Resample(tz_binary, pats[key], sitk.Transform(), sitk.sitkNearestNeighbor),  resampled_paths["tz_binary"])
                self.write_image(sitk.Resample(tz, pats[key], sitk.Transform(), sitk.sitkNearestNeighbor), resampled_paths["tz_probs"])
                self.write_image(sitk.Resample(pz_binary, pats[key], sitk.Transform(), sitk.sitkNearestNeighbor), resampled_paths["pz_binary"])
                self.write_image(sitk.Resample(pz, pats[key], sitk.Transform(), sitk.sitkNearestNeighbor), resampled_paths["pz_probs"])

                self.write_image(tz_binary, original_paths["tz_binary"])
                self.write_image(tz, original_paths["tz_probs"])
                self.write_image(pz_binary, original_paths["pz_binary"])
                self.write_image(pz, original_paths["pz_probs"])

                self.resampled[key] = resampled_paths
                self.original[key] = original_paths

            except Exception as e:
                logging.error(f"Error processing {key}: {e}")

    def write_image(self, image, path):
        try:
            sitk.WriteImage(image, path)
        except Exception as e:
            logging.error(f"Error writing image to {path}: {e}")
    
    def get_paths(self):
        return self.original, self.resampled

class DeleteRedundantfiles:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def clean_workspace_wg(paths_dict):
        for key, val in paths_dict.items():
            for path in val.values():
                try:
                    os.remove(path)
                    print(f"File {path} deleted successfully.")
                except FileNotFoundError:
                    print(f"File {path} not found.")
                except Exception as e:
                    print(f"Error deleting file {path}: {e}")
            nnpaths = os.path.join("nnUnet_paths","nnUNet_raw")
            file = os.path.join(nnpaths,os.path.join("OutcomesWG",f"ProstateWG_{key}.pkl"))
            try:
                os.remove(file)
            except FileNotFoundError:
                    print(f"File {file} not found.")
            except Exception as e:
                    print(f"Error deleting file {file}: {e}")

    @staticmethod
    def clean_workspace_zones(paths_dict):
        for key, val in paths_dict.items():
            for path in val.values():
                try:
                    os.remove(path)
                    print(f"File {path} deleted successfully.")
                except FileNotFoundError:
                    print(f"File {path} not found.")
                except Exception as e:
                    print(f"Error deleting file {path}: {e}")
            nnpaths = os.path.join("nnUnet_paths","nnUNet_raw")
            file = os.path.join(nnpaths,os.path.join("OutcomesZones",f"ProstateZonesFilteredLessDilated_ProstateZones_{key}.pkl"))
            try:
                os.remove(file)
            except FileNotFoundError:
                    print(f"File {file} not found.")
            except Exception as e:
                print(f"Error deleting file {file}: {e}")
                
    @staticmethod
    def clean_patients_directory(wg_paths, zones_paths):
        try:
            for filename in os.listdir(wg_paths):
                file_path = os.path.join(wg_paths, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"File {file_path} deleted successfully.")
        except Exception as e:
            print(f"Error cleaning directory {wg_paths}: {e}")
        
        try:
            for filename in os.listdir(zones_paths):
                file_path = os.path.join(zones_paths, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"File {file_path} deleted successfully.")
        except Exception as e:
            print(f"Error cleaning directory {zones_paths}: {e}")
