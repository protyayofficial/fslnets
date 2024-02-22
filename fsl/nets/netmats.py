#!/usr/bin/env python
#
# netmats.py - create network matrices ("netmats") for each separate run/subject in ts
#
# Author: Steve Smith, Ludo Griffanti, Roser Sala and Eugene Duff, 2013-2014
#         Paul McCarthy, 2023
#

import functools as ft
import itertools as it

import numpy as     np
from   numpy import random


def netmats(ts, method, do_rtoz=True, *args, **kwargs):
    """Create network matrices ("netmats") from the time series for each
    subject/run in ts.

    ts:      TimeSeries object

    method:  Connectivity estimation method - one of 'cov', 'amp',
             'correlation', 'rcorr', 'partial', or 'ridgep'.

    do_rtoz: Convert edge strengths to Z values. Ignored if method is
             'cov' or 'amp'. If True, a scaling factor is estimated
             from the time series data. Alternately, you can set this
             to a scaling factor if known for your data.

    All other arguments are passed through to the estimation function.

    Returns a (runs, edges) array, containing per-subject/run netmats. Or if
    method is 'amp', the returned array will have shape (runs, nodes).

    Methods:
     - 'cov':    covariance (non-normalised "correlation")
     - 'amp':    only use nodes' amplitudes
     - 'corr':   full correlation (diagonal is set to zero)
     - 'rcorr':  full correlation after regressing out global mean timecourse
     - 'ridgep': partial correlation using L2-norm Ridge Regression (aka
                 Tikhonov), e.g.:

                   # default regularisation rho=0.1
                   nm = netmats(ts, 'ridgep')
                   # rho=1
                   nm = netmats(ts, 'ridgep', rho=1)
    """

    # R to Z conversion not possible for these estimation methods
    if method in ('cov', 'covariance', 'amp', 'amplitude'):
        do_rtoz = False

    METHODS = {
        'cov'         : np.cov,
        'covariance'  : np.cov,
        'amp'         : ft.partial(np.std, axis=0, ddof=1),
        'amplitude'   : ft.partial(np.std, axis=0, ddof=1),
        'corr'        : correlation,
        'correlation' : correlation,
        'rcorr'       : rcorr,
        'icov'        : partial,
        'partial'     : partial,
        'ridgep'      : ridgep,
    }

    func = METHODS[method]

    # we are keeping just the amplitudes
    if method in ['amp', 'amplitude']:
        nmats = np.zeros((ts.ndatasets, ts.nnodes))
    else:
        nmats = np.zeros((ts.ndatasets, ts.nnodes ** 2))

    for i, subj, run, data in ts.allts:
        nmats[i] = func(data, *args, **kwargs).flatten()

    if do_rtoz:
        nmats = rtoz(ts, nmats, method, do_rtoz)

    return nmats


def correlation(data):
    """Full correlation (diagonal is set to zero). """
    data = data.T
    n    = data.shape[0]
    corr = np.corrcoef(data)
    corr[np.diag_indices(n)] = 0
    return corr


def rcorr(data):
    """Correlation after regressing out mean timecourse. """
    mean = np.atleast_2d(data.mean(axis=1)).T
    data = data - (mean @ (np.linalg.pinv(mean) @ data))
    return correlation(data)


def partial(data, lmbda=None):
    """Partial correlation, optionally "ICOV" L1-regularised if a lambda
    parameter is given.

    L1-regularised partial correlation is not implemented at the moment.
    """

    # simple partial correlation
    if lmbda is None:
        corr = -np.linalg.inv(np.cov(data.T))

    # ICOV L1-norm regularised partial correlation
    else:
        raise NotImplementedError('L1precision')

    n     = data.shape[1]
    diags = np.sqrt(np.abs(corr.diagonal(0)))
    diags = np.tile(diags, (1, n)).reshape((n, n))
    corr  = (corr / diags.T) / diags

    corr[np.diag_indices(n)] = 0

    return corr


def ridgep(data, rho=0.1, partial=True):
    """Partial correlation using L2-norm Ridge Regression (aka Tikhonov). """

    n    = data.shape[1]
    corr = np.cov(data.T)
    corr = corr / np.sqrt(np.mean(corr.diagonal(0)**2))
    corr = np.linalg.inv(corr + (rho * np.eye(n)))

    if partial:
        corr = -corr;
        diags = np.sqrt(np.abs(corr.diagonal(0)))
        diags = np.tile(diags, (1, n)).reshape((n, n))
        corr  = (corr / diags.T) / diags
        corr[np.diag_indices(n)] = 0

    return corr


def rtoz(ts, nmats, method, scale):
    """R to Z transformation. """

    np.random.seed(12345)

    if scale is not True:
        return 0.5 * np.log((1 + nmats) / (1 - nmats)) * scale

    # quick crappy estimate of median AR(1) coefficient
    arone = np.zeros((ts.ndatasets, ts.nnodes))
    for i, subj, run, subjts in ts.allts:
        for node in range(ts.nnodes):
            nodets         = subjts[:, node]
            arone[i, node] = np.sum(nodets[:-1] * nodets[1:]) / np.sum(nodets ** 2)

    arone = np.median(arone)

    # Estimate null correlations
    allcorrs = np.zeros((ts.ndatasets, ts.nnodes * (ts.nnodes - 1)))
    for i, subj, run, subjts in ts.allts:

        # create null data using the estimated AR(1) coefficient
        null = np.zeros((ts.ntimepoints(subj), ts.nnodes))

        for node in range(ts.nnodes):
            null[0, node] = random.randn()
            for tp in range(1, ts.ntimepoints(subj)):
                null[tp, node] = null[tp - 1, node] * arone + random.randn()

        # correlation matrix on null data
        if method in ('corr', 'correlation', 'rcorr'):
            nullr = np.corrcoef(null.T)
        else:
            nullr = -np.linalg.inv(np.cov(null.T))
            diags = np.sqrt(np.abs(nullr.diagonal(0)))
            diags = np.tile(diags, (1, ts.nnodes)).reshape((ts.nnodes, ts.nnodes))
            nullr = (nullr / diags.T) / diags

        allcorrs[i, :] = nullr[~np.eye(ts.nnodes, dtype=bool)].flatten()

    # numpy.log will return nan for real negative values,
    # so make sure the input is complex. numpy.std will
    # take the absolute value (magnitude) of its input,
    # so the resulting scale will be real and non-negative
    allcorrs = allcorrs + 0j
    allz     = 0.5 * np.log((1 + allcorrs) / (1 - allcorrs))
    scale    = 1 / allz.std(ddof=1)
    nmats    = 0.5 * np.log((1 + nmats) / (1 -  nmats)) * scale

    return nmats
