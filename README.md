
# Prostate Whole Gland and Zone automated segmentor

A Python module to perform Prostate and zonal segmentation from T2 Weighted MR images


## Usage/Examples

- To run the experiments you first need to create a "Pats" folder in the workspace where all the NIfTI T2-weighted MR files are located

- Also Create an "Outputs" folder where the results are going to be stored. The Outputs folder contains subfolders for each patient with the patient name, while in each patient subfolder, 2 more folders are created containing the Original extracted by nnUnet segmentations and resampled to the space of the input NIfTI files segmentations. Further 
```python
nnOutputSegmentationPaths.json 
#file containing the paths for each patient to the segmentations as extracted by nnUnet in 0.5X0.5X3 mm
```
```python
ResampledToOriginalSegmentationPaths.json 
#file containing the paths for each patient to the segmentations with the same spacing as the original images
```

## Execution

To run the script type the following in the Repo's workspace
```Bash
python __main__.py
```



## Authors

- [Dimitrios Zaridis](dimzaridis@gmail.com) * corresponding, M.Eng, PhD Student @National Technical University of Athens
- Charalampos Kalantzopoulos, M.Sc
- Eugenia Mylona, Ph.D
- Nikolaos S. Tachos, Ph.D
- Dimitrios I. Fotiadis, Professor of Biomedical Technology, university of Ioannina

## Badges

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

![Python](https://img.shields.io/badge/Python-3.9.18-green)
