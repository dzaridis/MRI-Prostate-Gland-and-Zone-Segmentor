import os
os.environ['nnUNet_raw'] = "/nnUnet_paths/nnUNet_raw" # allagi sto docker
os.environ['nnUNet_preprocessed'] = "/nnUnet_paths/nnUNet_preprocessed"
os.environ['nnUNet_results'] = "/nnUnet_paths/nnUNet_results"

from Utils import InputCheck
from Utils import ImageProcessor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import SimpleITK as sitk
from MedProIO import Coregistrator
import numpy as np
import json
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

def run():
    path = "/Pats" # load the volume here
    pats = InputCheck.load_nii_gz_files(path)

    pats_for_wg = {}
    for key,val in pats.items():
        processed = ImageProcessor.ImageProcessing(val)
        pats_for_wg.update({key:processed})

    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    pats_for_zones = {}

    def create_directories(key):
        try:
            os.makedirs(os.path.join("Outputs", key, "Resampled"), exist_ok=True)
            os.makedirs(os.path.join("Outputs", key, "Original"), exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directories for {key}: {e}")
            raise

    for k,v in pats_for_wg.items():
        sitk.WriteImage(v, os.path.join(nnUNet_raw, os.path.join(os.path.join('Dataset016_WgSegmentationPNetAndPicai', 'ImagesTs'), f"ProstateWG_{k}_0000.nii.gz")))
    # Create input directories for nnU-Net
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=False,
        device = torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, os.path.join('Dataset016_WgSegmentationPNetAndPicai','nnUNetTrainer__nnUNetPlans__3d_fullres')),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    predictor.predict_from_files(join(nnUNet_raw, os.path.join('Dataset016_WgSegmentationPNetAndPicai','ImagesTs')),
                                join(nnUNet_raw,"OutcomesZones"),
                                save_probabilities=True, overwrite=True,
                                num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    
    pats_for_wg_inference = {}
    for key, val in pats_for_wg.items():
        pats_for_wg_inference[key] = {
                "binary": os.path.join(join(nnUNet_raw,"OutcomesWG"), f"ProstateWG_{key}.nii.gz"),
                "probs": os.path.join(join(nnUNet_raw,"OutcomesWG"), f"ProstateWG_{key}.npz")
            }
        
    wg_dict_original = {}
    wg_dict_resampled = {}
    for key, val in pats_for_wg_inference.items():
        try:
            wg_binary = sitk.ReadImage(val["binary"])
            create_directories(key)
            wg_binary = ImageProcessor.process_mask(wg_binary)
            wg_binary = ImageProcessor.remove_small_components(wg_binary)

            filtered_ser = ImageProcessor.filter_ser(pats_for_wg[key], ImageProcessor.mask_dilation(wg_binary)) # dilates & filter WG
            
            probs = np.load(val["probs"])["probabilities"]
            wg_probs = probs[1,:,:,:]
            wg_probs = sitk.GetImageFromArray(wg_probs)
            wg_probs.CopyInformation(wg_binary)

            wg_binary_resampled = sitk.Resample(wg_binary,pats[key],sitk.Transform(),sitk.sitkNearestNeighbor)#ImageProcessor.resample(pats[key], wg_binary)
            wg_probs_resampled = sitk.Resample(wg_probs,pats[key],sitk.Transform(),sitk.sitkNearestNeighbor)#ImageProcessor.resample(pats[key], wg_probs)

            output_paths = {
                "Original": {
                    "wg_binary": os.path.join("Outputs", key, "Original", "wg_binary.nii.gz"),
                    "wg_probs": os.path.join("Outputs", key, "Original", "wg_probs.nii.gz")
                },
                "Resampled": {
                    "wg_binary": os.path.join("Outputs", key, "Resampled", "wg_binary.nii.gz"),
                    "wg_probs": os.path.join("Outputs", key, "Resampled", "wg_probs.nii.gz")
                }
            }

            sitk.WriteImage(wg_binary_resampled, output_paths["Resampled"]["wg_binary"])
            sitk.WriteImage(wg_probs_resampled, output_paths["Resampled"]["wg_probs"])
            sitk.WriteImage(wg_binary, output_paths["Original"]["wg_binary"])
            sitk.WriteImage(wg_probs, output_paths["Original"]["wg_probs"])

            wg_dict_original[key] = output_paths["Original"]
            wg_dict_resampled[key] = output_paths["Resampled"]

            pats_for_zones[key] = {
                "binary": os.path.join(join(nnUNet_raw,"OutcomesZones"), f"ProstateZonesFilteredLessDilated_ProstateZones_{key}.nii.gz"),
                "probs": os.path.join(join(nnUNet_raw,"OutcomesZones"), f"ProstateZonesFilteredLessDilated_ProstateZones_{key}.npz")
            }
            sitk.WriteImage(filtered_ser, os.path.join(nnUNet_raw, os.path.join(os.path.join('Dataset019_ProstateZonesSegmentationWgFilteredLessDilated', 'ImagesTs'), f"ProstateZonesFilteredLessDilated_ProstateZones_{key}_0000.nii.gz")))

        except Exception as e:
            logging.error(f"Error processing {key}: {e}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=False,
        device = torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, os.path.join('Dataset019_ProstateZonesSegmentationWgFilteredLessDilated','nnUNetTrainer__nnUNetPlans__3d_fullres')),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    predictor.predict_from_files(join(nnUNet_raw, os.path.join('Dataset019_ProstateZonesSegmentationWgFilteredLessDilated','ImagesTs')),
                                join(nnUNet_raw,"OutcomesZones"),
                                save_probabilities=True, overwrite=True,
                                num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    resampled, original = {}, {}

    def create_directories(key):
        try:
            os.makedirs(os.path.join("Outputs", key, "Resampled"), exist_ok=True)
            os.makedirs(os.path.join("Outputs", key, "Original"), exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directories for {key}: {e}")
            raise

    def write_image(image, path):
        try:
            sitk.WriteImage(image, path)
        except Exception as e:
            logging.error(f"Error writing image to {path}: {e}")

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

            create_directories(key)

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

            # Write resampled images
            write_image(sitk.Resample(tz_binary,pats[key],sitk.Transform(),sitk.sitkNearestNeighbor),  resampled_paths["tz_binary"]) #write_image(ImageProcessor.resample(pats[key], tz_binary), resampled_paths["tz_binary"])
            write_image(sitk.Resample(tz,pats[key],sitk.Transform(),sitk.sitkNearestNeighbor), resampled_paths["tz_probs"]) #write_image(ImageProcessor.resample(pats[key], tz), resampled_paths["tz_probs"])
            write_image(sitk.Resample(pz_binary,pats[key],sitk.Transform(),sitk.sitkNearestNeighbor), resampled_paths["pz_binary"]) #write_image(ImageProcessor.resample(pats[key], pz_binary), resampled_paths["pz_binary"])
            write_image(sitk.Resample(pz,pats[key],sitk.Transform(),sitk.sitkNearestNeighbor), resampled_paths["pz_probs"]) #write_image(ImageProcessor.resample(pats[key], pz), resampled_paths["pz_probs"])

            # Write original images
            write_image(tz_binary, original_paths["tz_binary"])
            write_image(tz, original_paths["tz_probs"])
            write_image(pz_binary, original_paths["pz_binary"])
            write_image(pz, original_paths["pz_probs"])

            resampled[key] = {**wg_dict_resampled[key], **resampled_paths}
            original[key] = {**wg_dict_original[key], **original_paths}

        except Exception as e:
            logging.error(f"Error processing {key}: {e}")

    for key, value in resampled.items():
        for key1, value in resampled[key].items():
            resampled[key][key1] = value.replace('\\', '/')
    for key, value in original.items():
        for key1, value in original[key].items():
            original[key][key1] = value.replace('\\', '/')

    with open(os.path.join("Outputs","ResampledSegmentationPaths.json"), "w") as file:
        json.dump(resampled, file, indent=4)
    with open(os.path.join("Outputs","OriginalSegmentationPaths.json"), "w") as file:
        json.dump(original, file, indent=4)

if __name__ == '__main__':
    # Initialize and start your multiprocessing objects here
    p = multiprocessing.Process(target=run)
    p.start()
    p.join()