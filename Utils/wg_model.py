from abc import ABC, abstractmethod
import SimpleITK as sitk
import numpy as np


class ImagePreprocessor(ABC):
    @abstractmethod
    def normalize(self, image: sitk.Image) -> sitk.Image:
        """
        Abstract method for normalizing a SimpleITK image.
        """
        pass

    def convert_to_float(self, image: sitk.Image) -> sitk.Image:
        """
        Convert image to float32 for processing.
        """
        return sitk.Cast(image, sitk.sitkFloat32)


class ZScoreNormalizer(ImagePreprocessor):
    def normalize(self, image: sitk.Image) -> sitk.Image:
        original_image = image
        try:
            image = self.convert_to_float(image)
        except ValueError:
            image = original_image

        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)

        mean = stats.GetMean()
        std = stats.GetSigma()

        normalized_image = sitk.ShiftScale(image, -mean, 1 / std if std > 0 else 1)
        return normalized_image

class ThresholdMaskFlattener:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def flatten_mask(self, sitk_image):
        # Convert SimpleITK Image to NumPy array
        array = sitk.GetArrayFromImage(sitk_image)

        # Apply threshold to create binary mask
        binary_mask = (array > self.threshold).astype(np.uint8)

        # Convert back to SimpleITK Image (if necessary)
        binary_sitk_image = sitk.GetImageFromArray(binary_mask)
        binary_sitk_image.CopyInformation(sitk_image)

        return binary_sitk_image