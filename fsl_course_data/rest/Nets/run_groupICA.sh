#!/bin/sh

#  Create input list:

ls -1 ~/scratch/FSLcourse/*/*.feat/filtered_func_data_clean_standard.nii.gz > input_files.txt

# Melodic:

melodic -i input_files.txt -o groupICA15 --tr=0.72 --nobet --bgthreshold=1 -a concat --bgimage=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz -m $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask.nii.gz --report --Oall -d 15

melodic -i input_files.txt -o groupICA100 --tr=0.72 --nobet --bgthreshold=1 -a concat --bgimage=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz -m $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask.nii.gz --report --Oall -d 100

# Dual regression:

dual_regression groupICA100/melodic_IC 1 -1 0 groupICA100.dr `cat input_files.txt`

dual_regression groupICA15/melodic_IC 1 design/unpaired_ttest.mat design/unpaired_ttest.con 5000 groupICA15.dr `cat input_files.txt`

# Create slice images for FSLnets:

slices_summary groupICA100/melodic_IC 4 $FSLDIR/data/standard/MNI152_T1_2mm groupICA100.sum
 
