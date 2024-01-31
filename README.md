# FSLNets


> This is a Python-based re-implementation of the MATLAB-based
> [FSLNets](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLNets) network modelling
> toolbox.


FSLNets is a set of simple scripts for carrying out basic network modelling
from (typically FMRI) timeseries data.


The main thing you will feed into FSLNets network modelling is N timecourses
from S subjects' datasets - i.e., timeseries from N network nodes. For display
purposes you will also need the spatial maps associated with the nodes (one
map per node). For example, a good way to get these timeseries and spatial
maps is to use [MELODIC
group-ICA](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC) with a
dimensionality of N, to get the group-level spatial maps, and then use [dual
regression](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/DualRegression) to generate
S subject-specific versions of the N timecourses. Alternatively, you might
have used a set of template images or ROIs from another study, to feed into
the dual regression.


Now you are ready to compute a _network matrix_ for each subject, which in
general will be an NxN matrix of connection strengths. The simplest and most
common approach is just to use "full" correlation, giving an NxN matrix of
correlation coefficients. Or, you might want to estimate the partial
correlation matrix, which should do a better job of only estimating the direct
network connections than the full correlation does. Once you have estimated a
network matrix for each subject, you can then test these matrices across
subjects, for example, testing each matrix element for a two-group subject
difference, or feeding the whole matrices into multivariate discriminant
analysis.


## Example usage


```python
%matplotlib

from glob import glob
import fsl.nets as nets

# Load time series from a dual regression output directory
ts = nets.load('./groupICA100.dr/', 0.72, thumbnaildir='./groupICA100.sum/')

# Or generate time series from a set of
# images using stage 1 of dual regression
subjdata = glob('subject_*/filtered_func_data_standard.nii.gz')
melic    = 'groupICA/melodic_IC.nii.gz'
ts       = nets.load_from_images(melic, subjdata, 0.72)

# View node time series/power spectra
nets.plot_spectra(ts)
nets.plot_timeseries(ts)

# Build a list of the indices of good nodes
# (those which represent signal and not noise)
goodnodes = [0,  1,  2,  4,  5,  6,  7,  8, 10, 11,
             12, 16, 17, 18, 19, 20, 21, 22, 24, 25,
             26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 39, 41, 42, 46, 47, 48, 49, 51,
             52, 54, 55, 56, 57, 58, 60, 61, 63, 64,
             65, 69, 70, 71, 72, 73, 76, 79, 80, 85,
             86, 92, 96]

# Regress out the time courses of bad nodes/components
nets.clean(ts, goodnodes, True)

# Calculate connectivity estimates
Fnetmats = nets.netmats(ts, 'corr')
Pnetmats = nets.netmats(ts, 'ridgep')

# Calculate group mean connectivity
Znet_F, Mnet_F = nets.groupmean(ts, Fnetmats, False)
Znet_P, Mnet_P = nets.groupmean(ts, Pnetmats, True, 'Partial correlation')

# Inspect hierarchical clustering on the full correlation matrix
nets.plot_hierarchy(ts, Znet_F, Znet_P, 'Full correlations', 'Partial correlations')

# Perform a GLM regression using randomise
p_corr,p_uncorr = nets.glm(ts, Pnetmats, 'design/unpaired_ttest.mat', 'design/unpaired_ttest.con');

# View group connectivity distributions for the most significant edges from contrast 2
nets.boxplots(ts, Pnetmats, Znet_P, p_corr[2])

# Train a classifier on edge strengths to differentiate your groups
nets.classify(Pnetmats, (6, 6))

# Display netmats interactively in a web browser
nets.web(ts, (Znet_F, Znet_P), ('Full correlation', 'Partial correlation'))
```


## Installation

FSLNets is installed as part of FSL 6.0.6.6 and newer. To use it, start a FSL
Python session with  `fslipython`, `fslpython`, or `fslnotebook`.

FSLNets can also be installed independently of FSL with `conda`. For example:

```
conda create                                                  \
  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
  -c conda-forge                                              \
  -p ./fslnets.env                                            \
  fsl-nets ipython
```

Finally, FSLNets can be installed into a Python environment with `pip`
(however, this method will **not** install any FSL dependencies):

```
pip install git+https://git.fmrib.ox.ac.uk/fsl/fslnets.git
```

## Notes for maintainers

FSLNets is built and released as a FSL conda package. New versions can be
released by creating a tag on this repository.

When a new version of FSLNets needs to be released, the version number in
`pyproject.toml` must be updated in advance. Use of [Semantic
versioning](https://www.semver.org) is encouraged.

Once the version number has been updated, a git tag should be created on this
repository, set to the new version number.


## Release history

## 0.8.4 (Wednesday 31st January 2024)

- Fixed `netmats(ts, 'amps')` - it was returning the same value for each node
  within each subject (the standard deviation across all nodes).


## 0.8.3 (Monday 27th November 2023)

- Adjusted the `nets.glm` function so that it does not use `$TMPDIR` when calling
  `randomise`, as the `randomise` call may be submitted to a cluster node which has
  a different `$TMPDIR`.


## 0.8.2 (Thursday 16th November 2023)

- Adjust the `nets.glm` function to pause until `randomise` has completed, when
  running on a cluster.


## 0.8.1 (Wednesday 15th November 2023)

- Make the `plot_groupmean` function resilient to NaN values.


## 0.8.0 (Tuesday 15th August 2023)

- The `plot_spectra` function was using a hard-coded window length, rather than
  a window length based on the length of the time series.
- New `plot_timeseries` function, which may be useful for quality-control
  purposes.
- The `load` function will load all `*.txt` files in the input directory, rather
  than only files with the name `dr_stage1_*.txt`.


## 0.7.2 (Wednesday 2nd August 2023)

- Changed the `nets.web` logic so that the web server is kept alive until the
  expected number of successful HTTP requests are made. Naively shutting down
  the server after 5 seconds can be problematic on slow systems.


## 0.7.1 (Tuesday 15th June 2023)

- Adjust the `plot_spectra` function so that the node spectra are normalised
  in the same way as the original `nets_spectra.m` function.
- Fix the `glm` function so that it doesn't crash when given a single contrast.

## 0.7.0 (Thursday 25th May 2023)

- First tagged release of the new Python-based `fslnets` library.
