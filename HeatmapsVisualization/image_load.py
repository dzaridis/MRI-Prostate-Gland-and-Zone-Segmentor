import os
import numpy as np
import pydicom
import nibabel as nib
from pathlib import Path

class IdentifyPaths:
    def __init__(self, base_path):
        self.base_path = base_path
        self.anonymized_patient = os.path.join(self.base_path,"anonymized")
        self.patients = {}
        for file in os.listdir(self.anonymized_patient):
            if file.endswith(".yaml"):
                continue
            else:
                self.patients.update({file:os.path.join(self.anonymized_patient,file)})
    
    def get_patient_path(self):
        return self.patients
    
    def get_segm_paths(self):
        segm_paths = {}
        for patient in self.patients.keys():
            segm_path = os.path.join(self.base_path,f"{patient}_1", "Resampled")
            segm_paths.update({patient:segm_path})
        return segm_paths
    
from typing import Dict, List, Optional, Tuple, Union
class ImageLoader:
    """
    Class to handle loading and processing of DICOM series and NIfTI probability masks.
    """
    def __init__(self, dicom_path: str, segm_path: str):
        """
        Initialize the ImageLoader with paths to DICOM and segmentation data.
        
        Args:
            dicom_path (str): Path to the DICOM series directory
            segm_path (str): Path to the segmentation masks directory
        """
        self.dicom_path = Path(dicom_path)
        self.segm_path = Path(segm_path)
        
        # Initialize data containers
        self.dicom_series: List[Dict] = []
        self.probability_masks: Dict[str, np.ndarray] = {}
        
        # Validate paths
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that the provided paths exist."""
        if not self.dicom_path.exists():
            raise FileNotFoundError(f"DICOM path does not exist: {self.dicom_path}")
        if not self.segm_path.exists():
            raise FileNotFoundError(f"Segmentation path does not exist: {self.segm_path}")
    
    def load_dicom_series(self) -> List[Dict]:
        """
        Load all DICOM files from the patient directory.
        
        Returns:
            List[Dict]: List of dictionaries containing DICOM data, sorted by slice position
        """
        try:
            # Navigate through the folder structure
            study_dir = next(self.dicom_path.iterdir())  # First study
            series_dir = next(study_dir.iterdir())  # First series
            
            # Get all DICOM files
            dicom_files = sorted(series_dir.glob('image_*.dcm'))
            
            if not dicom_files:
                raise FileNotFoundError("No DICOM files found in the specified directory")
            
            series_data = []
            
            for dcm_path in dicom_files:
                try:
                    ds = pydicom.dcmread(str(dcm_path))
                    
                    # Extract necessary DICOM attributes
                    image_data = {
                        'image': ds.pixel_array,
                        'position': float(ds.ImagePositionPatient[2]),
                        'slice_thickness': float(ds.SliceThickness),
                        'pixel_spacing': [float(x) for x in ds.PixelSpacing],
                        'instance_number': int(ds.InstanceNumber),
                        'file_path': str(dcm_path),
                        'instance': ds
                    }
                    
                    series_data.append(image_data)
                    
                except Exception as e:
                    print(f"Error reading {dcm_path}: {e}")
                    continue
            
            # Sort by slice position
            self.dicom_series = sorted(series_data, key=lambda x: x['position'])
            
            if not self.dicom_series:
                raise ValueError("No valid DICOM files could be loaded")
            
            return self.dicom_series
        
        except Exception as e:
            raise RuntimeError(f"Error loading DICOM series: {str(e)}")
    
    def load_probability_mask(self, mask_type: str = 'wg') -> Optional[np.ndarray]:
        """
        Load a probability mask NIfTI file.
        
        Args:
            mask_type (str): Type of mask to load ('wg', 'tz', or 'pz')
            
        Returns:
            Optional[np.ndarray]: Probability mask data or None if loading fails
        """
        valid_types = ['wg', 'tz', 'pz']
        if mask_type not in valid_types:
            raise ValueError(f"Invalid mask type. Must be one of {valid_types}")
        
        mask_path = self.segm_path / f'{mask_type}_probs.nii.gz'
        
        try:
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
            nifti = nib.load(str(mask_path))
            mask_data = nifti.get_fdata()
            
            # Store in dictionary
            self.probability_masks[mask_type] = mask_data
            
            return mask_data
            
        except Exception as e:
            print(f"Error loading probability mask {mask_type}: {e}")
            return None
    
    def load_all_probability_masks(self) -> Dict[str, np.ndarray]:
        """
        Load all available probability masks (WG, TZ, PZ).
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of loaded probability masks
        """
        for mask_type in ['wg', 'tz', 'pz']:
            self.load_probability_mask(mask_type)
        
        return self.probability_masks
    
    def get_slice(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get a specific slice of the DICOM image and corresponding probability masks.
        
        Args:
            index (int): Slice index
            
        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: DICOM slice and dictionary of mask slices
        """
        if not self.dicom_series:
            raise ValueError("DICOM series not loaded. Call load_dicom_series() first.")
        
        if not 0 <= index < len(self.dicom_series):
            raise IndexError(f"Slice index {index} out of range [0, {len(self.dicom_series)-1}]")
        
        dicom_slice = self.dicom_series[index]['image']
        mask_slices = {}
        
        for mask_type, mask_data in self.probability_masks.items():
            if mask_data is not None:
                mask_slices[mask_type] = mask_data[:, :, index]
        
        return dicom_slice, mask_slices
    
    def get_metadata(self) -> Dict:
        """
        Get metadata about the loaded images.
        
        Returns:
            Dict: Dictionary containing metadata about the loaded images
        """
        return {
            'num_slices': len(self.dicom_series) if self.dicom_series else 0,
            'slice_thickness': self.dicom_series[0]['slice_thickness'] if self.dicom_series else None,
            'pixel_spacing': self.dicom_series[0]['pixel_spacing'] if self.dicom_series else None,
            'image_shape': self.dicom_series[0]['image'].shape if self.dicom_series else None,
            'loaded_masks': list(self.probability_masks.keys()),
        }