from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Visualizer:
    """
    Class to handle visualization of DICOM images and probability mask overlays.
    """
    def __init__(self, dicom_series: List[Dict], probability_masks: Dict[str, np.ndarray]):
        """
        Initialize the visualizer with DICOM series and probability masks.
        
        Args:
            dicom_series: List of dictionaries containing DICOM data
            probability_masks: Dictionary of probability masks for each zone
        """
        self.dicom_series = dicom_series
        # Rotate masks when storing them
        self.probability_masks = {
            key: self._orient_mask(mask) 
            for key, mask in probability_masks.items()
        }
        self.num_slices = len(dicom_series)
        
        # Default colormap for each mask type
        self.colormaps = {
            'wg': 'Reds',    # Whole gland in red
            'tz': 'Greens',  # Transition zone in green
            'pz': 'Blues'    # Peripheral zone in blue
        }
    
    def _orient_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Orient the mask to match DICOM orientation.
        Rotates mask 90 degrees clockwise and flips if necessary.
        
        Args:
            mask: The input mask array
            
        Returns:
            Properly oriented mask array
        """
        # Rotate 90 degrees clockwise (equivalent to -90 counterclockwise)
        rotated_mask = np.rot90(mask, k=-1, axes=(0, 1))
        
        # Flip if needed (add this if masks are also flipped)
        # rotated_mask = np.flip(rotated_mask, axis=0)  # Uncomment if vertical flip is needed
        rotated_mask = np.flip(rotated_mask, axis=1)  # Uncomment if horizontal flip is needed
        
        return rotated_mask
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range."""
        if image.max() == image.min():
            return image
        return (image - image.min()) / (image.max() - image.min())
    
    def show_slice(self, slice_idx: int, mask_types: Optional[List[str]] = None, 
                alpha: float = 0.3, figsize: tuple = (15, 5),
                prob_min: float = 0.1):  # Minimum probability to show
        """
        Display a single slice with selected mask overlays as continuous probability heatmaps.
        
        Args:
            slice_idx: Index of the slice to display
            mask_types: List of mask types to overlay ('wg', 'tz', 'pz')
            alpha: Base opacity of the overlay
            figsize: Figure size (width, height)
            prob_min: Minimum probability to display (for cleaner visualization)
        """
        if not 0 <= slice_idx < self.num_slices:
            raise ValueError(f"Slice index must be between 0 and {self.num_slices-1}")
        
        if mask_types is None:
            mask_types = list(self.probability_masks.keys())
            
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Plot DICOM
        ax = plt.gca()
        dicom_slice = self.normalize_image(self.dicom_series[slice_idx]['image'])
        ax.imshow(dicom_slice, cmap='gray')
        
        # Add each probability mask as a heatmap overlay
        for mask_type in mask_types:
            if mask_type in self.probability_masks:
                mask_slice = self.probability_masks[mask_type][:, :, slice_idx]
                
                # Mask very low probabilities for cleaner visualization
                # but keep the continuous values for the rest
                masked_slice = np.ma.masked_where(mask_slice < prob_min, mask_slice)
                
                # Create custom alpha that scales with probability
                # Higher probability = more opaque
                variable_alpha = alpha * (mask_slice / mask_slice.max())
                
                # Show the continuous probability values
                im = ax.imshow(masked_slice,
                            cmap=self.colormaps[mask_type],
                            alpha=variable_alpha)
                
                # Add colorbar for each mask type
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax, label=f'{mask_type.upper()} Probability')
        
        ax.set_title('DICOM with Probability Heatmaps')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=plt.get_cmap(self.colormaps[mt])(0.6))
            for mt in mask_types
        ]
        ax.legend(legend_elements, [mt.upper() for mt in mask_types],
                loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    def show_all_slices(self, mask_types: Optional[List[str]] = None, 
                        alpha: float = 0.3, 
                        prob_min: float = 0.1,
                        slices_per_row: int = 5):
        """
        Display all slices in a grid layout, with each zone shown separately.
        
        Args:
            mask_types: List of mask types to display ('wg', 'tz', 'pz')
            alpha: Opacity of the overlay
            prob_min: Minimum probability to display
            slices_per_row: Number of slices to display per row
        """
        if mask_types is None:
            mask_types = list(self.probability_masks.keys())
            
        # Calculate layout
        n_slices = self.num_slices
        n_rows = (n_slices - 1) // slices_per_row + 1
        n_zones = len(mask_types)
        
        # Calculate figure size (increased size)
        fig_width = 6 * slices_per_row  # Increased from 4 to 6
        fig_height = 6 * n_rows * n_zones  # Increased from 4 to 6
        
        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create outer grid with large spacing between zones
        outer_grid = gridspec.GridSpec(
            n_zones, 1,
            height_ratios=[1]*n_zones,
            hspace=0.5  # Increased space between zones
        )
        
        for zone_idx, mask_type in enumerate(mask_types):
            # Calculate position for zone title
            zone_top = 1 - (zone_idx/n_zones)
            zone_height = 1/n_zones
            title_y = zone_top - (zone_height * 0.05)  # Slightly below the top of the zone
            
            # Add zone title
            plt.figtext(
                0.5,  # x position (center)
                title_y,  # y position
                f"{mask_type.upper()} Zone Probabilities",
                ha='center',
                va='bottom',
                fontsize=16,
                fontweight='bold'
            )
            
            # Create subplot grid for this zone with proper spacing
            zone_grid = gridspec.GridSpecFromSubplotSpec(
                n_rows, slices_per_row,
                subplot_spec=outer_grid[zone_idx],
                hspace=0.4,  # Space between rows within a zone
                wspace=0.3   # Space between columns
            )
            
            # Process each slice
            for slice_idx in range(n_slices):
                row = slice_idx // slices_per_row
                col = slice_idx % slices_per_row
                
                # Create subplot
                ax = plt.Subplot(fig, zone_grid[row, col])
                fig.add_subplot(ax)
                
                # Show DICOM image
                dicom_slice = self.normalize_image(self.dicom_series[slice_idx]['image'])
                ax.imshow(dicom_slice, cmap='gray')
                
                # Show probability mask
                if mask_type in self.probability_masks:
                    mask_slice = self.probability_masks[mask_type][:, :, slice_idx]
                    masked_slice = np.ma.masked_where(mask_slice < prob_min, mask_slice)
                    
                    im = ax.imshow(masked_slice,
                                cmap=self.colormaps[mask_type],
                                alpha=alpha)
                    
                    # Add colorbar to last slice in each row
                    if col == slices_per_row-1 or slice_idx == n_slices-1:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        plt.colorbar(im, cax=cax)
                
                # Add slice number with more padding
                ax.set_title(f'Slice {slice_idx+1}', pad=15, fontsize=10)
                ax.axis('off')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.95,      # Leave space at top
            bottom=0.05,   # Leave space at bottom
            hspace=0.5     # Additional space between zones
        )
        plt.show()