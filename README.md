# fslnets


This is a Python-based re-implementation of the MATLAB
[FSLNets](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLNets) network modelling
toolbox. It aims to provide like-for-like functionality with `FSLNets`.

Example usage:


```python
import fsl.nets as nets

# Load time series from a dual regression output directory
ts = nets.load('data/', 0.72)

# Regress out the time courses of bad nodes/components
ts.goodnodes = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 17,
                18, 19, 20, 21, 22, 23, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                40, 42, 43, 47, 48, 49, 50, 52, 53, 55,
                56, 57, 58, 59, 61, 62, 64, 65, 66, 70,
                71, 72, 73, 74, 77, 80, 81, 86, 87, 93, 97]
nets.clean(ts, True)

# Calculate connectivity estimates
Fnetmats = nets.netmats(ts, 'corr')
Pnetmats = nets.netmats(ts, 'ridgep')

# Calculate group mean connectivity
Znet_F, Mnet_F = nets.groupmean(ts, Fnetmats)
Znet_P, Mnet_P = nets.groupmean(ts, Pnetmats)

# Perform a GLM regression using randomise
p_uncorr,p_corr = nets.glm(ts, Pnetmats, 'unpaired_ttest.mat', 'unpaired_ttest.con');
```
