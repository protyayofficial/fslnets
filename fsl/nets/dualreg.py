#!/usr/bin/env python
#
# dualreg.py - Dual regression
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#
"""This module implements the dual regression procedure, for extracting
subject-specific time courses and spatial maps, related to a set of
group-level components/spatial maps.

https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/DualRegression/
"""


from   glob    import glob
import os.path as     op
import numpy   as     np

from fsl.data.image    import Image
from fsl.utils.tempdir import tempdir
from fsl.wrappers      import fslmaths, fslmerge, fslsplit, fsl_glm

from file_tree import FileTree
from fsl_pipe  import Pipeline, In, Out



default_tree = """
melodic_IC.nii.gz (spatialmaps)

sub-{subject} (subject_dir)
  filtered_func_data_clean_standard.nii.gz (fmri_data)

{outdir}
  mask.nii.gz                         (mask)
  maskALL.nii.gz                      (all_mask)
  dr_stage1_subject{subject}.txt      (stage1)
  dr_stage2_subject{subject}.nii.gz   (stage2)
  dr_stage2_subject{subject}_Z.nii.gz (stage2z)
  dr_stage2_ic{ic}.nii.gz             (stage2ic)
"""


def create_common_mask(fmri_data : In,
                       mask      : Out,
                       all_mask  : Out = None):
    """Create a mask by combining the input data from each subject.

    :arg fmri_data:    List of paths to per-subject 4D time series data.

    :arg mask:         Path to save the mask file.

    :arg all_mask:     Path to save a 4D image with per-subject masks.
    """

    # so can work with as piped function or regular python call
    fmri_data = getattr(fmri_data, 'data', fmri_data)
    fmri_data = [op.abspath(f) for f in fmri_data]
    mask      = op.abspath(mask)

    if all_mask is None:
        all_mask = op.join(op.dirname(mask), 'maskALL')

    with tempdir():
        for i, infile in enumerate(fmri_data):
            fslmaths(infile).Tstd().bin().run(f'{i}_mask')
        fslmerge('t', all_mask, *glob('*_mask.*'))
        fslmaths(all_mask).Tmin().run(mask)


def stage1(spatialmaps : In,
           fmri_data   : In,
           mask        : In,
           stage1      : Out):
    """Dual regression stage 1.  Perform a GLM, regressing the spatial-maps
    into each subject's 4D dataset to generate the subject-specific time series
    for each component/node contained in the spatial map file.

    :arg spatialmaps: Path to 4D image containing spatial maps
    :arg fmri_data:   List of paths to per-subject 4D time series data.
    :arg mask:        3D mask to apply to the data
    :arg stage1:      Path to save the stage 1 outputs (as a plain text file)
    """
    fsl_glm(fmri_data, stage1, spatialmaps, demean=True, m=mask)


def stage2(fmri_data : In,
           mask      : In,
           stage1    : In,
           stage2    : Out,
           stage2z   : Out,
           varnorm   : bool = True):
    """Dual regression stage 2. Regresses subject-specific per-component
    timecourses back into subject timeseries data to get a subject-specific
    set of spatial maps.

    :arg fmri_data:   List of paths to per-subject 4D time series data.
    :arg mask:        3D mask to apply to the data
    :arg stage1:      Per-subject time series for each spatial map (stage 1
                      outputs)
    :arg stage2:      Path to save the stage 2 outputs (as a 4D image)
    """
    fsl_glm(fmri_data, stage2, stage1, out_z=stage2z,
            demean=True, des_norm=varnorm, m=mask)


def stage2_transform(stage2 : In, stage2ic : Out):
    """Transform stage 2 outputs from one file per subject (containing all
    spatial maps) to one file per spatial map (containing all subjects).
    """

    stage2 = [Image(s) for s in getattr(stage2, 'data', stage2)]
    hdr    = stage2[0].header

    for i, icfile in enumerate(stage2ic.data):
        data = np.array([s[..., i] for s in stage2])
        Image(data, header=hdr).save(icfile)
