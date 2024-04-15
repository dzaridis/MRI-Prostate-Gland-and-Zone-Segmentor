import os
os.environ['nnUNet_raw'] = os.path.join("nnUnet_paths", "nnUNet_raw") # allagi sto docker
os.environ['nnUNet_preprocessed'] = os.path.join("nnUnet_paths", "nnUNet_preprocessed")
os.environ['nnUNet_results'] = os.path.join("nnUnet_paths", "nnUNet_results")
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import SimpleITK as sitk
from abc import ABC, abstractmethod
import torch

class BaseNNUnetModule(ABC):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
    
    @abstractmethod
    def prediction(self):
        pass
    
    @abstractmethod
    def return_paths(self):
        pass

class WGNNUnet(BaseNNUnetModule):
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
    def __init__(self, input_path:str, output_path:str):
        self.input_path = input_path
        self.output_path = output_path
        self.predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, os.path.join(self.input_path,'nnUNetTrainer__nnUNetPlans__3d_fullres')),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    
    def prediction(self):
        self.predictor.predict_from_files(join(nnUNet_raw, os.path.join(self.input_path,'ImagesTs')),
                            join(nnUNet_raw,self.output_path),
                            save_probabilities=True, overwrite=True,
                            num_processes_preprocessing=2, num_processes_segmentation_export=2,
                            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    
    def return_paths(self, pats_for_wg:dict):
        pats_for_wg_inference = {}
        for key in pats_for_wg.keys():
            pats_for_wg_inference[key] = {
                    "binary": os.path.join(join(nnUNet_raw,self.output_path), f"ProstateWG_{key}.nii.gz"),
                    "probs": os.path.join(join(nnUNet_raw,self.output_path), f"ProstateWG_{key}.npz")
                }
        return pats_for_wg_inference

class ZonesNNUnet(BaseNNUnetModule):
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
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, os.path.join(self.input_path,'nnUNetTrainer__nnUNetPlans__3d_fullres')),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    
    def prediction(self):
        self.predictor.predict_from_files(join(nnUNet_raw, os.path.join(self.input_path,'ImagesTs')),
                            join(nnUNet_raw,"OutcomesZones"),
                            save_probabilities=True, overwrite=True,
                            num_processes_preprocessing=2, num_processes_segmentation_export=2,
                            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    
    def return_paths(self, pats_for_wg_inference:dict):
        pats_for_zones = {}
        for key in pats_for_wg_inference.keys():
            pats_for_zones[key] = {
                "binary": os.path.join(join(nnUNet_raw,"OutcomesZones"), f"ProstateZonesFilteredLessDilated_ProstateZones_{key}.nii.gz"),
                "probs": os.path.join(join(nnUNet_raw,"OutcomesZones"), f"ProstateZonesFilteredLessDilated_ProstateZones_{key}.npz")
            }
        return pats_for_zones