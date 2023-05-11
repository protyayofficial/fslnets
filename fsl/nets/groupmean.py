#!/usr/bin/env python
#
# groupmean.py - estimate group mean/one-group-t-test and consistency of
#                netmats across runs/subjects
#
# Author: Steve Smith, 2012-2014
# Author: Paul McCarthy, 2023
#


import numpy       as np
import scipy.stats as stats


def groupmean(ts, netmats):
    """
    """
    nsubjs = netmats.shape[0]
    nedges = netmats.shape[1]
    dof    = nsubjs - 1
    mean   = netmats.mean(axis=0)
    std    = netmats.std( axis=0)
    tvals  = np.sqrt(nsubjs) * mean / std
    tpos   = tvals > 0
    tneg   = tvals < 0

    zvals       = np.zeros(tvals.shape)
    zvals[tpos] = -stats.norm.ppf(stats.t.cdf(-tvals[tpos], dof))
    zvals[tneg] =  stats.norm.ppf(stats.t.cdf( tvals[tneg], dof))
    zinf        = ~np.isfinite(zvals)
    zvals[zinf] = tvals[zinf]

    if np.sqrt(nedges) == ts.nnodes:
        zvals = zvals.reshape((ts.nnodes, ts.nnodes))
        mean  = mean .reshape((ts.nnodes, ts.nnodes))

    return zvals, mean
