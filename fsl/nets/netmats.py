#!/usr/bin/env python
#
# netmats.py - create network matrices ("netmats") for each separate run/subject in ts
#
# Author: Steve Smith, Ludo Griffanti, Roser Sala and Eugene Duff, 2013-2014
#         Paul McCarthy, 2023
#

import itertools as it

import numpy as     np
from   numpy import random


def netmats(ts, method, do_rtoz=True, *args, **kwargs):

    # R to Z conversion not possible for these estimation methods
    if method in ('cov', 'covariance', 'amp', 'amplitude'):
        do_rtoz = False

    METHODS = {
        'cov'         : np.cov,
        'covariance'  : np.cov,
        'amp'         : np.std,
        'amplitude'   : np.std,
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
        nmats = np.zeros((ts.nsubjects, ts.nnodes))
    else:
        nmats = np.zeros((ts.nsubjects, ts.nnodes ** 2))

    for subj in range(ts.nsubjects):
        subjts         = ts.subjts(subj)
        nmats[subj, :] = func(subjts, *args, **kwargs).flatten()

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

    if scale is not True:
        nmats = 0.5 * np.log((1 + nmats) / (1 -  nmats)) * scale
        return nmats

    # quick crappy estimate of median AR(1) coefficient
    arone = np.zeros(ts.nsubjects * ts.nnodes)
    for i, (subj, node) in enumerate(it.product(range(ts.nsubjects), range(ts.nnodes))):
        subjts   = ts.subjts(subj)
        nodets   = subjts[:, node]
        sarone   = np.sum(nodets[:-1] * nodets[1:]) / np.sum(nodets ** 2)
        arone[i] = sarone
    arone = np.median(arone)

    # Estimate null correlations
    allcorrs = np.zeros((ts.nsubjects, ts.nnodes * (ts.nnodes - 1)))
    for subj in range(ts.nsubjects):

        # create null data using the estimated AR(1) coefficient
        slc  = ts.subjslice(subj)
        null = np.zeros((ts.ntimepoints(subj), ts.nnodes))

        for node in range(ts.nnodes):
            null[0, node] = random.randn()
            for tp in range(1, ts.ntimepoints(subj)):
                null[tp, node] = null[tp - 1, node] * arone + random.randn()

        # correlation matrix on mull data
        if method in ('corr', 'correlation', 'rcorr'):
            nullr = np.corrcoef(null.T)
        else:
            nullr = -np.linalg.inv(np.cov(null.T))
            diags = np.sqrt(np.abs(nullr.diagonal(0)))
            diags = np.tile(diags, (1, ts.nnodes)).reshape((ts.nnodes, ts.nnodes))
            nullr = (nullr / diags.T) / diags
        allcorrs[subj, :] = nullr[~np.eye(ts.nnodes, dtype=bool)].flatten()

    allz       = 0.5 * np.log((1 + allcorrs) / (1 - allcorrs))
    correction = 1 / allz.std()
    nmats      = 0.5 * np.log((1 + nmats) / (1 -  nmats)) * correction

    return nmats
