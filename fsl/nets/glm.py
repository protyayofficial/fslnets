#!/usr/bin/env python
#
# glm.py - do cross-subject GLM on a set of netmats
#
# Author: Steve Smith and Ludo Griffanti, 2013-2014
#         Paul McCarthy, 2023
#

import os.path           as     op
import matplotlib.pyplot as     plt
import numpy             as     np
import fsl.data.vest     as     vest
from   fsl.data.image    import Image
from   fsl.utils.tempdir import tempdir
from   fsl.utils.run     import runfsl
from   fsl.wrappers      import randomise


def glm(ts, netmats, design, contrasts, nperms=5000, plot=True, title=None):
    """
    """

    nsubjs     = netmats.shape[0]
    nedges     = netmats.shape[1]
    ncontrasts = vest.loadVestFile(contrasts).shape[0]
    design     = op.abspath(design)
    contrasts  = op.abspath(contrasts)

    with tempdir():

        netmats = netmats.T.reshape((nedges, 1, 1, nsubjs))

        Image(netmats).save('netmats')

        randomise('netmats', 'output', d=design, t=contrasts, n=nperms,
                  x=True, uncorrp=True, log={'tee' : False})

        puncorr = np.zeros((ncontrasts, nedges))
        pcorr   = np.zeros((ncontrasts, nedges))

        for con in range(ncontrasts):
            puncorr[con] = Image(f'output_vox_p_tstat{con+1}')    .data.flatten()
            pcorr[  con] = Image(f'output_vox_corrp_tstat{con+1}').data.flatten()

    if plot:
        plot_pvalues(ts, pcorr)

    return pcorr, puncorr


def plot_pvalues(ts, pvals, title=None):
    """
    """
    ncontrasts = pvals.shape[0]
    fig        = plt.figure()
    shape      = (ts.nnodes, ts.nnodes)

    tril = np.zeros(shape, dtype=bool)
    triu = np.zeros(shape, dtype=bool)
    tril[np.tril_indices(shape[0], -1)] = True
    triu[np.triu_indices(shape[0],  1)] = True

    axes   = fig.subplots(1, ncontrasts)
    meshes = []

    for ax, contrast in zip(axes, range(ncontrasts)):

        cpvals                = pvals[contrast].reshape(shape)
        lovals                = cpvals[tril]
        hivals                = cpvals[triu]
        hivals[hivals < 0.95] = 0

        data       = np.full(shape, np.nan)
        data[tril] = lovals
        data[triu] = hivals

        # display zero values slightly faded
        ax.pcolormesh(np.zeros(shape), cmap='jet', vmin=0, vmax=1, alpha=0.8)

        # display non-zero values fully opaque
        data[triu & (data == 0)] = np.nan
        m = ax.pcolormesh(np.flipud(data), cmap='jet', vmin=0, vmax=1)

        meshes.append(m)

        ax.axis('off')
        ax.set_title(f'Contrast {contrast+1}')
        ax.plot([0, 1], [1, 0], transform=ax.transAxes, c='#7777ff')

    fig.colorbar(meshes[-1], ax=axes, label='1-P', location='right')
    fig.set_layout_engine('constrained')
