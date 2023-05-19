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


Example usage:


```python
%matplotlib
import fsl.nets as nets

# Load time series from a dual regression output directory
ts = nets.load('./groupICA100.dr/', 0.72, thumbnaildir='./groupICA100.sum/')

# View node time series power spectra
nets.plot_spectra(ts)

# Regress out the time courses of bad nodes/components
goodnodes = [0,  1,  2,  4,  5,  6,  7,  8, 10, 11,
             12, 16, 17, 18, 19, 20, 21, 22, 24, 25,
             26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 39, 41, 42, 46, 47, 48, 49, 51,
             52, 54, 55, 56, 57, 58, 60, 61, 63, 64,
             65, 69, 70, 71, 72, 73, 76, 79, 80, 85,
             86, 92, 96]

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
