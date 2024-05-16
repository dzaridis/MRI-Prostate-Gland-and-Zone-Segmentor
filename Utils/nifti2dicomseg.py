''' Created ONLY for binary segmentations '''

import os
import sys
import time # Time used to set time/date point of segmentation created
from pathlib import Path
from typing import List, Union, Dict
from copy import deepcopy
import numpy as np
import SimpleITK as sitk
from pydicom import Dataset, DataElement, dcmread, dcmwrite
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
    SegmentationStorage
)
from pydicom.sequence import Sequence

def _sitk_files_order(series_dir: Path) -> tuple:
    ''' 
    Warning: Spatial ordering (z-axis), 
    Use other GDCM ordering if this does not fit your requirements.
    Also can be set for specific SeriesUID, recursive, etc. 
    '''
    reader=sitk.ImageSeriesReader()
    return reader.GetGDCMSeriesFileNames(series_dir)

def order_dcmfiles(series_dir:Path)-> List[Dataset]:
    '''
    Reads dicom files in Spatial order Image Position Patient(z-axis).
    **Future work: os.walk, DWI multi-frame, multiple series support. 
    '''
    ds = [
        dcmread(x)
        for x in _sitk_files_order(series_dir)
        if x.endswith(".dcm")
    ]

    return ds

def nifti_reader(nifti_path:Path) -> sitk.Image:
    ''' 
    Reads nifti image and convert them in uint8 
    '''
    pixel_type = sitk.sitkUInt8 # this is 1
    return sitk.ReadImage(nifti_path, outputPixelType=pixel_type)

def _safe_read_seg(seg_path:Path)->np.ndarray:
    ''' Checks if path exists, loads image and check if segmentation exists'''
    if not os.path.exists(seg_path):
        return None

    pixel_type = sitk.sitkUInt8
    seg = sitk.ReadImage(seg_path,outputPixelType=pixel_type)
    seg_arr = sitk.GetArrayFromImage(seg)

    if np.max( seg_arr ) == 0:
        #segmentation failed
        return None

    return seg_arr

def auto_seg_reader(seg_directory:Path)->Dict[str,np.ndarray]:
    ''' Read segmentations to nifti'''

    wg_path = os.path.join( seg_directory, "wg_binary.nii.gz")
    pz_path = os.path.join( seg_directory, "pz_binary.nii.gz")
    tz_path = os.path.join( seg_directory, "tz_binary.nii.gz")

    segments_dict = {
        "wg": {"array":_safe_read_seg(wg_path)},
        "pz": {"array":_safe_read_seg(pz_path)},
        "tz": {"array":_safe_read_seg(tz_path)}
    }
    segments_dict = {
        key:value 
        for key, value in segments_dict.items() 
        if value["array"] is not None
    }

    return segments_dict

def clear_overlapping(seg_dict:Dict[str,dict])->dict:
    '''Required for dicom-seg standards'''

    possible_overlap = False
    if 'wg' in seg_dict:
        common_wg = seg_dict['wg']['array']

        if 'pz' in seg_dict:
            possible_overlap = True
            common_wg = common_wg & seg_dict['pz']['array']

        if 'tz' in seg_dict:
            possible_overlap = True
            common_wg = common_wg & seg_dict['pz']['array']

        seg_dict['wg']['no_overlap'] = deepcopy(seg_dict['wg']['array'] )
        if possible_overlap:
            common_wg = common_wg.astype(bool)
            seg_dict['wg']['no_overlap'][common_wg] = 0

    possible_overlap = False
    if 'pz' in seg_dict:
        common_pz = seg_dict['pz']['array']

        if 'tz' in seg_dict:
            possible_overlap = True
            common_pz = common_pz & seg_dict['tz']['array']

        seg_dict['pz']['no_overlap'] = deepcopy(seg_dict['pz']['array'] )
        if possible_overlap:
            common_wg = common_pz.astype(bool)
            seg_dict['pz']['no_overlap'][common_pz] = 0

    if 'tz' in seg_dict:
        seg_dict['tz']['no_overlap'] = deepcopy(seg_dict['tz']['array'] )

    return seg_dict

def clean_zero_slices(seg_dict:Dict[str,dict])->dict:
    '''Removes slices that contains only zeroes'''

    check_key = 'array'
    for value in seg_dict.values():
        if 'no_overlap' in value:
            check_key = 'no_overlap'
        break

    for key, value in seg_dict.items():

        nonzero_slices_indices = np.nonzero(
            np.any(
                value[check_key], axis=(1, 2)
            )
        )[0]

        new_array = []
        for index in nonzero_slices_indices:
            slice_ = value[check_key][index]
            new_array.append(slice_)

        seg_dict[key]["no_zero_slices"] = deepcopy(np.array(new_array))
        seg_dict[key]['seg_index'] = deepcopy(nonzero_slices_indices.tolist())

    return seg_dict

def reference_image_sop(reference_ds:List[Dataset])-> Sequence:
    '''Gathers the reference frames SOP UID from reference image'''
    ref_series_seq = Sequence()
    ref_instance_seq = Sequence()
    frame_ds = reference_ds[0]


    for dataset in reference_ds:

        temp_ds = Dataset()
        temp_ds.ReferencedSOPClassUID = dataset.SOPClassUID
        temp_ds.ReferencedSOPInstanceUID = dataset.SOPInstanceUID

        ref_instance_seq.append( temp_ds )

    temp_ds = Dataset()
    temp_ds.ReferencedInstanceSequence = ref_instance_seq
    temp_ds.SeriesInstanceUID = frame_ds.SeriesInstanceUID
    ref_series_seq.append(temp_ds)

    return ref_series_seq

def dim_organization_sequence(segment_ds:Dataset )->Dataset:
    ''' Guidelines in how to organize the segmentation dataset '''

    org_uid = generate_uid()
    organization_sequence = Dataset()
    organization_sequence.DimensionOrganizationUID = org_uid

    segment_ds.DimensionOrganizationSequence = Sequence()
    segment_ds.DimensionOrganizationSequence.append(organization_sequence)

    dim_idx_seq1 = Dataset()
    dim_idx_seq2 = Dataset()

    dim_idx_seq1.DimensionOrganizationUID = org_uid
    dim_idx_seq1.DimensionIndexPointer = (0x0062, 0x000b)
    dim_idx_seq1.FunctionalGroupPointer = (0x0062, 0x000a)
    dim_idx_seq1.DimensionDescriptionLabel = 'ReferencedSegmentNumber'

    dim_idx_seq2.DimensionOrganizationUID = org_uid
    dim_idx_seq2.DimensionIndexPointer = (0x0020, 0x0032)
    dim_idx_seq2.FunctionalGroupPointer = (0x0020, 0x9113)
    dim_idx_seq2.DimensionDescriptionLabel = 'ImagePositionPatient'

    segment_ds.DimensionIndexSequence = Sequence([
        dim_idx_seq1,
        dim_idx_seq2
    ])

    return segment_ds

def shared_metadata(reference_ds:Dataset)->Sequence:
    '''Shared metadata for all slices '''

    shared_functional = Dataset()
    image_orientation = Dataset()
    pixel_measures = Dataset()

    image_orientation.ImageOrientationPatient = reference_ds.ImageOrientationPatient
    pixel_measures.SliceThickness = reference_ds.SliceThickness
    pixel_measures.SpacingBetweenSlices = reference_ds.SpacingBetweenSlices
    pixel_measures.PixelSpacing = reference_ds.PixelSpacing

    shared_functional.PlaneOrientationSequence = Sequence([image_orientation])
    shared_functional.PixelMeasuresSequence = Sequence([pixel_measures])

    return Sequence([shared_functional])

def structures_dictionary(segmentation_type:str,seg_counter:int)->Sequence:
    '''
    Creates segmentation dictionary in dicom.
    ref: https://dicom.nema.org/medical/Dicom/2016c/output/chtml/part03/sect_C.10.7.html#:~:text=Graphic%20Layer%20Recommended%20Display%20RGB,Value%20(0070%2C0401).
    '''
    segment_ds = Dataset()

    seg_dict = {
        "wg" : {
            "description":"Whole gland of the prostate",
            "code":"T-92000",
            "meaning":'Prostate',
            "color": [42318, 26448, 26367] # Random picks
            },
        "pz" : {
            "description":"Peripheral zone of the prostate",
            "code":"T-D05E4",
            "meaning":'Prostate peripheral zone',
            "color": [34340, 40315, 27406] # Random picks
            },
        "tz" : {
            "description":'Transition zone of the prostate',
            "code":'T-D0823',
            "meaning":"Prostate transition zone",
            "color": [58366, 30737, 53006] # Random picks
            }
    }

    # Segmented Property Category Code Sequence
    segment_property = Dataset()
    segment_property.CodeValue = 'T-D000A'
    segment_property.CodingSchemeDesignator = 'SRT'
    segment_property.CodeMeaning = 'Anatomical Structure'

    segment_ds.SegmentNumber = seg_counter
    segment_ds.SegmentLabel = segmentation_type.upper()
    segment_ds.SegmentAlgorithmType = 'AUTOMATIC'
    segment_ds.SegmentAlgorithmName = 'NNUnet'
    segment_ds.SegmentDescription = seg_dict[segmentation_type]['description']

    segment_property_type = Dataset()
    segment_property_type.CodeValue = seg_dict[segmentation_type]['code']
    segment_property_type.CodingSchemeDesignator = 'SRT'
    segment_property_type.CodeMeaning = seg_dict[segmentation_type]['meaning']

    segment_ds.RecommendedDisplayCIELabValue = seg_dict[segmentation_type]['color']
    segment_ds.SegmentedPropertyCategoryCodeSequence = Sequence([segment_property])
    segment_ds.SegmentedPropertyTypeCodeSequence = Sequence([segment_property_type])

    return segment_ds

def per_frame_group(
        reference_ds_list:List[Dataset],
        seg_dict:Dict[str,dict]) -> Sequence:
    '''Adds meta-data for each slice'''

    # Per-frame Functional Groups Sequence

    # Purpose same for all Dataset
    purpose_ds = Dataset()
    purpose_ds.CodeValue = '121322'
    purpose_ds.CodingSchemeDesignator = 'DCM'
    purpose_ds.CodeMeaning = 'Source image for image processing operation'

    # Derivation same for all Dataset
    derivation_code = Dataset()
    derivation_code.CodeValue = '113076'
    derivation_code.CodingSchemeDesignator = 'DCM'
    derivation_code.CodeMeaning = 'Segmentation'

    def frame_dataset(
            frame_dict:dict,
            slice_index:int,
            dataset:Dataset,
            purpose_ds:Dataset,
            derivation_code:Dataset,
            )->Sequence:

        per_frame = Dataset()

        derivation_image = Dataset()
        source_image = Dataset()

        source_image.ReferencedSOPClassUID = dataset.SOPClassUID
        source_image.ReferencedSOPInstanceUID = dataset.SOPInstanceUID
        source_image.PurposeOfReferenceCodeSequence = Sequence([purpose_ds])

        derivation_image.SourceImageSequence = Sequence([source_image])
        derivation_image.DerivationCodeSequence = Sequence([derivation_code])

        # Only zero frames can be discarded, here we kept them for now

        dim_index = Dataset()
        dim_index.DimensionIndexValues = [frame_dict["label"], slice_index]
        per_frame.FrameContentSequence = Sequence([dim_index])

        img_position = Dataset()
        img_position.ImagePositionPatient = dataset.ImagePositionPatient
        per_frame.PlanePositionSequence = Sequence([img_position])

        ref_segment_num = Dataset()
        ref_segment_num.ReferencedSegmentNumber  = frame_dict["label"]
        per_frame.SegmentIdentificationSequence = Sequence([ref_segment_num])

        per_frame.DerivationImageSequence = Sequence([derivation_image])

        return per_frame

    per_frame_sequence = Sequence()

    for value in seg_dict.values():

        if "seg_index" in value:
            for idx in value["seg_index"]:

                dataset = reference_ds_list[idx]
                per_frame = frame_dataset(value,idx,dataset,purpose_ds,derivation_code)
                per_frame_sequence.append(per_frame)
        else:

            for idx, dataset in enumerate(reference_ds_list):
                per_frame = frame_dataset(value,idx,dataset,purpose_ds,derivation_code)
                per_frame_sequence.append(per_frame)

    return per_frame_sequence

def array2bits(seg_dict:Dict[str,dict])->Union[np.ndarray,str]:
    ''' Converts array to bits to be fed in PixelData'''
    check_value = "array"

    for value in seg_dict.values():

        if "no_zero_slices" in value:
            check_value = "no_zero_slices"
        break

    data = []
    for key in seg_dict:
        data.append(seg_dict[key][check_value])

    data = np.concatenate(data)

    nframes = str( len(data) )

    frames =[]
    for frame in data:
        frames.append(frame.ravel())

    frames = np.concatenate(frames)
    bit_frames = np.packbits(frames, bitorder="little",axis=0).tobytes()

    return bit_frames, nframes

def nifti2dicomseg(seg_dir_path:Path, t2_path:Path, single_seg:str=""):
    ''' 
    Convert nifti segmentations to dcm files.
    Each segmentation file will have their own dcm file. Also, a multi-frame
    dcm file will be available with no overlaps!
    '''

    seg_ds = Dataset()
    seg_ds.is_little_endian = True
    seg_ds.is_implicit_VR = False
    seg_dict = auto_seg_reader(seg_dir_path)

    if not seg_dict:
        print(f"{t2_path} has no segmentations!")
        return

    if single_seg:
        if single_seg not in seg_dict:
            print(f"{t2_path} has no segmentation {single_seg}")
            return

    t2_dataset = order_dcmfiles(t2_path)
    t2_ds = t2_dataset[0]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    temp_prefix = "2.8.846.0."+modification_date+".1."+modification_time+'.'

    meta = Dataset()
    meta.MediaStorageSOPClassUID = SegmentationStorage
    sop_instance_uid = generate_uid(temp_prefix)
    implementation = generate_uid(temp_prefix)
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = implementation
    meta.ImplementationVersionName = 'PYDICOM 2.4.2'

    seg_ds.file_meta = meta
    seg_ds.InstitutionName = "ICS-FORTH"

    seg_ds.ImageType = ['DERIVED', 'PRIMARY']
    seg_ds.SOPClassUID = SegmentationStorage
    seg_ds.SOPInstanceUID = sop_instance_uid

    seg_ds.SeriesDate = modification_date
    seg_ds.ContentDate = modification_date
    seg_ds.SeriesTime = modification_time
    seg_ds.ContentTime = modification_time
    seg_ds.AccessionNumber = ''
    seg_ds.Modality = 'SEG'
    seg_ds.Manufacturer = 'Unspecified'
    seg_ds.SeriesDescription = "Segmentation WG+PZ+TZ"
    seg_ds.ManufacturerModelName = 'Unspecified'
    seg_ds.DeviceSerialNumber = '1'
    seg_ds.SoftwareVersions = '0'
    seg_ds.ContentQualification = 'RESEARCH'
    seg_ds.SeriesInstanceUID = generate_uid(temp_prefix)

    if "FrameOfReferenceUID" in t2_ds:
        seg_ds.FrameOfReferenceUID = t2_ds.FrameOfReferenceUID
    else:
        seg_ds.FrameOfReferenceUID = generate_uid(temp_prefix)

    seg_ds.PositionReferenceIndicator = ''
    seg_ds.SamplesPerPixel  = 1
    seg_ds.PhotometricInterpretation  = 'MONOCHROME2'
    seg_ds.BitsAllocated = 1
    seg_ds.BitsStored = 1
    seg_ds.HighBit = 0
    seg_ds.PixelRepresentation = 0
    seg_ds.LossyImageCompression = '00'
    seg_ds.SegmentationType = 'BINARY'

    if "PatientID" in t2_ds:
        seg_ds.PatientID = t2_ds.PatientID
        patient_id = t2_ds.PatientID

        if "PatientName" in t2_ds:
            seg_ds.PatientName = t2_ds.PatientName
        else:
            seg_ds.PatientName = t2_ds.PatientID

    elif "PatientName" in t2_ds:
        seg_ds.PatientID = t2_ds.PatientName
        seg_ds.PatientName = t2_ds.PatientName
        patient_id = t2_ds.PatientName

    if "StudyInstanceUID" in t2_ds:
        seg_ds.StudyInstanceUID = t2_ds.StudyInstanceUID
        study_uid = t2_ds.StudyInstanceUID

        if "StudyID" in t2_ds:
            seg_ds.StudyID = t2_ds.StudyID
        else:
            seg_ds.StudyID = t2_ds.StudyInstanceUID

    elif "StudyID" in t2_ds:
        seg_ds.StudyInstanceUID = t2_ds.StudyID
        seg_ds.StudyID = t2_ds.StudyID
        study_uid = t2_ds.StudyID

    # Unknown Implementation Error in DataElement, change to if
    if 'StudyTime' in t2_ds:
        seg_ds.StudyTime = t2_ds.StudyTime
    if 'StudyDate' in t2_ds:
        seg_ds.StudyDate = t2_ds.StudyDate
    if 'StudyDescription' in t2_ds:
        seg_ds.StudyDescription = t2_ds.StudyDescription

    tags_to_copy = [
        "PatientBirthDate",
        "PatientWeight",
        "PatientSex",
        "PatientAge",
        'ReferringPhysicianName',
        "PatientIdentityRemoved"
    ]

    for tag in tags_to_copy:

        if tag in t2_ds:

            data_elem = DataElement(
                tag = t2_ds.get_item(tag).tag,
                VR = t2_ds.get_item(tag).VR,
                value = t2_ds.get_item(tag).value
            )

            seg_ds[tag] = data_elem

    seg_ds.Rows = t2_ds.Rows
    seg_ds.Columns = t2_ds.Columns

    seg_ds.SeriesNumber = '100'
    seg_ds.InstanceNumber = '200'

    if single_seg == "wg":
        seg_ds.SeriesNumber = '101'
        seg_ds.InstanceNumber = '201'
        seg_ds.SeriesDescription = "Prostate Whole Gland"

    if single_seg == "pz":
        seg_ds.SeriesNumber = '102'
        seg_ds.InstanceNumber = '202'
        seg_ds.SeriesDescription = "Prostate Peripheral Zone"

    if single_seg == "tz":
        seg_ds.SeriesNumber = '103'
        seg_ds.InstanceNumber = '203'
        seg_ds.SeriesDescription = "Prostate Transition Zone"

    seg_ds.ReferencedSeriesSequence = reference_image_sop(t2_dataset)
    seg_ds = dim_organization_sequence(seg_ds)
    seg_ds.SharedFunctionalGroupsSequence = shared_metadata(t2_ds)

    segment_sequence = Sequence()

    label_count = 1 # counts successful segmentations
    if not single_seg:

        for zone,value in seg_dict.items():

            if value["array"] is None:
                total_failed += 1
                continue

            segment_sequence.append(structures_dictionary(zone,label_count))
            value["label"] = label_count
            label_count += 1

    else:

        seg_dict = { single_seg:seg_dict[single_seg] }
        segment_sequence.append(structures_dictionary(single_seg,label_count))
        seg_dict[single_seg]["label"] = label_count

    seg_ds.SegmentSequence = segment_sequence

    if not single_seg:
        seg_dict = clear_overlapping(seg_dict)
        seg_dict = clean_zero_slices(seg_dict)

    bit_arr, nframes = array2bits(seg_dict)
    seg_ds.NumberOfFrames = nframes
    seg_ds.PixelData = bit_arr
    seg_ds.PerFrameFunctionalGroupsSequence = per_frame_group(t2_dataset, seg_dict)

    seg_ds.ContentLabel = "SEGMENTATION"
    seg_ds.ContentDescription = "NNUnet Prostate Zone Segmentation"
    seg_ds.ContentCreatorName = "Dimitris Player"

    output = os.path.join(
        "dicom_outputs",
        patient_id,
        study_uid,
        "segmentations"
    )

    os.makedirs(output, exist_ok=True)

    if not single_seg:
        dcmwrite(os.path.join(output,'prostate_zones.dcm'), seg_ds, write_like_original=False)
    else:
        dcmwrite(os.path.join(output,f'{single_seg}.dcm'), seg_ds, write_like_original=False)

def keep_same_direction(dcm_image:Dataset, sitk_image:sitk.Image)->sitk.Image:
    ''' 
    ---Not used. Might be needed for future works---
    Set sitk image in same orientation as dicom.
    This may not required.
    '''

    dcm_direction = dcm_image.ImageOrientationPatient # 6 elements
    sitk_direction = sitk_image.GetDirection() # 9 elements

    sitk_direction = list(sitk_direction)

    sitk_direction[0] = float(dcm_direction[0])
    sitk_direction[3] = float(dcm_direction[1])
    sitk_direction[6] = float(dcm_direction[2])

    sitk_direction[1] = float(dcm_direction[3])
    sitk_direction[4] = float(dcm_direction[4])
    sitk_direction[7] = float(dcm_direction[5])

    sitk_direction = tuple(sitk_direction)

    return sitk_image.SetDirection(sitk_direction)
