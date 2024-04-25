# Branches Description
## main  
### You can directly create the docker image by cloning the branch and execute docker build      

1. INPUT VOLUME: "/Pats"  
2. OUTPUT VOLUME: "/Outputs"  

## DesktopAPP  

Execute the following command. Then 2 Windows prompt will open  
1. The first one is the input NIfTI folder -> Select a folder where the NIfTI T2-Weighted MR Prostate Volume lie in a NIfTI format (.nii.gz)
2. The second one is the output folder you wish to extract the segmentations.
```Bash
pip install -r requirements.txt
python __main__.py
```

## The structure of the outcome is the following    
### The output folder  
![The structure of the folder](https://github.com/dzaridis/MRI-Prostate-Gland-and-Zone-Segmentor/blob/main/Materials/photo1.jpg)  
### The structure of the 2 json dictionaries that are references of the segmentation for each patient.  

- The first dictionary (ResampledToOriginalSegmentationPaths.json) corresponds to the outputs that are resampled to the initial input
- The second dictionary (nnOutputSegmentationPaths.json) corresponds to the outputs that are extracted directly from the nnUNet in 0.5x0.5x3.0 voxel spacing   
wg_binary: Prostate's whole gland binary mask  
wg_probs: Prostate's whole gland probabilities 
pz_binary: Prostate's Peripheral zone binary mask  
pz_probs: Prostate's Peripheral zone probabilities  
tz_binary: Prostate's Transition zone binary mask  
tz_binary: Prostate's Peripheral zone probabilities  
![Structure of the dictionaries with paths](https://github.com/dzaridis/MRI-Prostate-Gland-and-Zone-Segmentor/blob/main/Materials/photo2.jpg)

## Each patient will contain the following folder and each folder the following NIfTI files. Please use the json files to navigate propertly. THEY ARE MESS!  
![Folders within each patient](https://github.com/dzaridis/MRI-Prostate-Gland-and-Zone-Segmentor/blob/main/Materials/photo3.jpg)  
![Files within each patient's subfolders](https://github.com/dzaridis/MRI-Prostate-Gland-and-Zone-Segmentor/blob/main/Materials/photo4.jpg)  
### The structure of the 2 json dictionaries that are references of the segmentation for each patient.  


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

## Execute as docker

A docker image is available for anyone to use at the following repository
current version:1.4
https://hub.docker.com/repository/docker/dimzaridis/prostai-segmentor/general

## Authors

- [Dimitrios Zaridis](dimzaridis@gmail.com) * corresponding, M.Eng, PhD Student @National Technical University of Athens
- Charalampos Kalantzopoulos, M.Sc
- Eugenia Mylona, Ph.D
- Nikolaos S. Tachos, Ph.D
- Dimitrios I. Fotiadis, Professor of Biomedical Technology, university of Ioannina

## Badges

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

![Python](https://img.shields.io/badge/Python-3.9.18-green)
