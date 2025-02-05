### 
# Cavernoma.py
# Tool for measuring cavernoma volume and signal intensity after normalisation
# Usage : python ComputeStatistics.py
#

import nibabel as nib
import numpy as np
import os
import glob
import pandas as pd
import shutil

### Loading NIFTI file. 
# Each brain file must be named brain_Patient_Lesion.nii.gz
# Each lesion ROI must be named source_Patient_Lesion.nii.gz
brains_world = glob.glob("registered_folder/brain_*.nii.gz")
brains_world = {i.replace("registered_folder/brain_","").replace(".nii.gz","") : i for i in brains_world}
lesions_world = glob.glob("registered_folder/roi_*.nii.gz")
lesions_world = {i.replace("registered_folder/roi_","").replace(".nii.gz","") : i for i in lesions_world}

### Loading of each individual lesion
# Excel file must contain in each row a different lesion
# "Pat" column must contain unique patient identifier
# "Lesion" column must contain lesion identifier per patient
# "Pat_Lesion" column must be a concatenation of "Pat" column, "_" string, and "Lesion column
dataset = pd.read_excel("Cavernoma.xlsx")

### Loading MNI reference volume
mnivol = nib.load("MNI152_T1_1mm_brain.nii.gz")
mnimaskdata = mnivol.get_fdata() > 0

### Normalization of signal intensity
for p, volpath in brains_world.items():
    # Iterate over each patient
    normalized_path = "output_folder/normalized_" + p + ".nii.gz"
    list_lesions_patient = list(dataset[dataset["Pat"] == p]["Lesion"])
    # Load patient MRI data
    patMRIvolume = nib.load(volpath)
    patMRIdata = patMRIvolume.get_fdata()
    # Cleanup mask based on MNI volume and Raw T1 data
    negativemask = patMRIdata == 0
    negativemask =  np.logical_or(negativemask, mnimaskdata == 0)
    # Cleanup mask by removing each lesion
    for l in list_lesions_patient:
        patles = p + "_" + l
        lesMRIVolume = nib.load(lesions_world[patles])
        negativemask = np.logical_or(negativemask, lesMRIVolume.get_fdata())
    # Compute mean and standard deviation in masked volume
    meanval = patMRIdata[~negativemask].mean()
    sdval = patMRIdata[~negativemask].std()
    # Standardize values and multiply by 1000 and add 5000 for better visualisation
    newArray = ((patMRIdata-meanval)/sdval)
    # Cleanup voxels outside of MNI mask
    newArray[mnimaskdata == 0] = 0
    newVolume = nib.Nifti1Image(newArray, affine=patMRIvolume.affine)
    # Save normalized volume
    os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
    nib.save(newVolume, normalized_path)

### Computation of lesions properties
lesions_mean = []
lesions_sd = []
lesion_volumes = []
for patles in dataset["Pat_Lesion"]:
    # Loading patient volume
    p = patles.split("_")[0]
    normalized_path = "output_folder/normalized_" + p + ".nii.gz"
    patMRIvolume = nib.load(normalized_path)
    patMRIdata = patMRIvolume.get_fdata()

    # Loading Lesion ROI
    roipath = lesions_world[patles]
    roivolume = nib.load(roipath)

    # Applying ROI to volume and computing statistics
    masked_data = patMRIdata[np.logical_and(roivolume.get_fdata()>0.5, patMRIdata > 0)]
    lesions_mean.append(masked_data.mean())
    lesions_sd.append(masked_data.std())

    # Computing lesion volume
    my_volume = np.sum(roivolume.get_fdata()>0.5)/1000
    lesion_volumes.append(my_volume)

### Save results in new excel file
dataset["Volume (mL)"] = lesion_volumes
dataset["MeanValue"] = lesions_mean
dataset["StdValue"] = lesions_sd
dataset.to_excel("CavernomaWithIntensity.xlsx")
