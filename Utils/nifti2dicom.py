''' 
Convert nifti image to dicom with SimpleITK, modified example [1].
Nifti images by nature are anonymized contains only metadata related to the image for manipulation.
!Note: DO NOT USE FOR DICOM-SEG!! Not for multi-frame images (e.g. different b-values, planes)
[1]: https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html
'''

import os
import time
from pathlib import Path
import SimpleITK as sitk
from pydicom.uid import generate_uid, MRImageStorage

def nifti2dicom( nifti_path:Path):
    ''' Nifti to Dicom with SimpleITK. Not for multi-frames one dcm file directories!'''

    output_name = ""
    reader = sitk.ReadImage( nifti_path )
    writer = sitk.ImageFileWriter()
    # This guideline is when you load from a dicom directory
    # "Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()
    # There are no UID stored in nifti, so we generate them! for uniqueness,
    # I set a random prefix accompanied from data and time for the uids

    # Copy relevant tags from the original meta-data dictionary (private tags are
    # also accessible).
    # random prefix id

    # prefix normally is related to the organization ID (e.g. OID, OMG)
    patient_id = generate_uid()
    study_id = generate_uid()
    series_id = generate_uid()
    sop_uid = generate_uid()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    _direction = reader.GetDirection()
    rows, columns, frames = map(str,reader.GetSize())
    pixel_spacing = reader.GetSpacing()[0:2]
    pixel_spacing = "\\".join(map(str,pixel_spacing))
    slice_thickness = str(reader.GetSpacing()[2])

    direction = "\\".join(
        map(
            str,
            (
                _direction[0],
                _direction[3],
                _direction[6],
                _direction[1],
                _direction[4],
                _direction[7],
            ),  # Image Orientation (Patient)
        )
    )

    tags_to_insert = {
        "0010|0010":patient_id,  # Patient Name
        "0010|0020":patient_id,  # Patient ID
        "0008|0016": MRImageStorage, # SOP Class UID
        "0008|0018":sop_uid, # SOP Instance UID
        "0008|1030":"Prostate Zone Segmentation", # Study Description
        "0010|0030":"",  # Patient Birth Date
        "0010|0040":"",  # Patient's Sex
        "0010|1010":"",  # Patient's Age
        "0010|1030":"",  # Patient's Weight
        "0020|000d":study_id,  # Study Instance UID, for machine consumption
        "0020|0010":study_id,  # Study ID, for human consumption
        "0020|000e":series_id, # Series Instance UID
        "0020|0052":generate_uid(), # Frame of Reference UID
        "0020|0011":"3", # Series Number
        "0020|0013":"0", # Instance Number
        "0008|0020":modification_date, # Study Date
        "0008|0030":modification_time,  # Study Time
        "0008|0021":modification_date, # Series Date
        "0020|0037":direction, # Image Orientation (Patient)
        "0008|0050":'1',  # Accession Number
        "0008|0060":'MRI',  # Modality
        "0008|0031":modification_time, #Series Time
        "0008|0070":"", # Manufacturer
        "0008|1090":"", # Manufacturer's Model Name
        "0008|103e":"Nifti T2AX MRI", # Series Description
        "0008|0008":"DERIVED\\SECONDARY", # Image Type
        "0018|9004":'RESEARCH', # Content Qualification
        "0028|0030":pixel_spacing,
        "0018|0050":slice_thickness,
        "0028|0010":rows,
        "0028|0011":columns,
        "0028|0008":str(frames), # Number of Frames
        "0028|0004":'MONOCHROME2', # Photometric Interpretation
        "0028|2110":'00', # Lossy Image Compression
        "0028|0002":"1", # Samples per Pixel
        "0028|0100":"16", # Bits Allocated
        "0028|0101":"12", # Bits Stored
        "0028|0102":"11", # High Bit
        "0028|0103":"0" # Pixel Representation

    }

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number,
    # cannot start with zero, and separated by a '.' We create a unique series ID
    # using the date and time.
    # NOTE: Always represent DICOM tags using lower case hexadecimals.
    #       DICOM tags represent hexadecimal numbers, so 0020|000D and 0020|000d
    #       are equivalent. The ITK/SimpleITK dictionary is string based, so these
    #       are two different keys, case sensitive. When read from a DICOM file the
    #       hexadecimal string representations are in lower case. To ensure consistency,
    #       always use lower case for the tags.
    # Tags of interest:

    series_tag_values = [ (k, value) for k,value in tags_to_insert.items() ]

    for i in range(reader.GetDepth()):

        image_slice = reader[:, :, i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)

        # Slice specific tags.
        #   Instance Creation Date
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        #   Instance Creation Time
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        #   Image Position (Patient)
        image_slice.SetMetaData(
            "0020|0032",
            "\\".join(map(str, reader.TransformIndexToPhysicalPoint((0, 0, i)))),
        )
        #   Instance Number
        image_slice.SetMetaData("0020|0013", str(i))

        # Write to the output directory and add the extension dcm, to force writing
        # in DICOM format.
        output_name = os.path.join("output",patient_id,study_id,"t2w")
        os.makedirs( output_name, exist_ok=True)
        output_file = os.path.join( output_name, f"image_{i:04}.dcm")
        writer.SetFileName(output_file)
        writer.Execute(image_slice)

    return output_name
