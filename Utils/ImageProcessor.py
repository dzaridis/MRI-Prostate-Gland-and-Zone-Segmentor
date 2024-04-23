from MedProIO import Resampler
from MedProIO import CropAndPad
from MedProIO import Coregistrator
import SimpleITK as sitk


def ImageProcessing(sitk_image, spacing = (0.5, 0.5, 3.0), target_size = (256, 256, 24)):
    res = Resampler.ResamplerToSpacing()
    res.set_reference_image(sitk_image)
    res.set_spacing(spacing=spacing)
    res.execute_processing()
    tr = res.get_transformed_image()

    res = CropAndPad.VolumeCropperAndPadder()
    res.set_reference_image(tr)
    res.set_target_size(target_size=target_size)
    res.execute_processing()
    tr = res.get_transformed_image()
    return tr

def resample(fix,mov):
    res = Coregistrator.SequenceResampler()

    # set the reference Sitk object 
    res.set_reference_image(fix)

    # set the moving Sitk object 
    res.set_moving_image(mov)

    # execute
    res.execute_processing()

    # get the transformed moving object
    tr = res.get_transformed_image()

    # Get a list with potential issues arose
    issues = res.get_issues()
    return tr


def mask_dilation(mask):
    radius = 5  # Adjust this to change the amount of dilation

    # Create a binary dilation filter with a ball structuring element
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetKernelRadius(radius)

    # Apply the dilation filter to the mask
    dilated_mask = dilate_filter.Execute(mask)
    return dilated_mask


def filter_ser(image, mask):
    if not (image.GetOrigin() == mask.GetOrigin() and
            image.GetSpacing() == mask.GetSpacing() and
            image.GetDirection() == mask.GetDirection()):
        mask.SetOrigin(image.GetOrigin())
        mask.SetSpacing(image.GetSpacing())
        mask.SetDirection(image.GetDirection())
    return sitk.Mask(image, mask)


def remove_small_components(mask, keep_largest_only=True, size_threshold=None):
    """
    Remove smaller non-connected components from a binary mask.

    :param mask: SimpleITK image object (binary mask).
    :param keep_largest_only: If True, only the largest component is kept.
    :param size_threshold: The size threshold to use for keeping components. 
                            Components smaller than this will be removed.
    :return: SimpleITK image with smaller components removed.
    """
    # Label the connected components
    labeled_mask = sitk.ConnectedComponent(mask)

    # Analyze the labeled components
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(labeled_mask)

    # Determine which labels to keep
    if keep_largest_only:
        # Keep only the largest component
        largest_label = label_stats.GetNumberOfPixels(label_stats.GetLabels()[0])
        labels_to_keep = [label_stats.GetLabels()[0]]
    else:
        # Keep components larger than the size threshold
        labels_to_keep = [label for label in label_stats.GetLabels() if label_stats.GetNumberOfPixels(label) >= size_threshold]

    # Create a binary image to store the result
    output_image = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    output_image.CopyInformation(mask)

    # Iterate over the labels to keep and add them to the output image
    for label in labels_to_keep:
        component = sitk.BinaryThreshold(labeled_mask, lowerThreshold=label, upperThreshold=label, insideValue=1, outsideValue=0)
        output_image = sitk.Or(output_image, component)

    return output_image

def process_mask(mask):
    """
    Perform morphological closing and then dilation on a SimpleITK mask.
    """
    closing_radius = (2, 2, 0)  # Adjust as needed
    closed_mask = sitk.BinaryMorphologicalClosing(mask, closing_radius)

    return closed_mask

def create_binary_masks(image):
    """
    Create binary masks for the given SimpleITK image.
    The image is assumed to have regions with values 0, 1, and 2.
    Returns two binary masks: one for regions 0 and 1, and another for regions 0 and 2.
    """
    mask_tz = sitk.BinaryThreshold(image, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
    mask_pz = sitk.BinaryThreshold(image, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0)

    return mask_tz, mask_pz