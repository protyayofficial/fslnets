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
from   fsl.nets.util     import printTitle, printColumns


def glm(ts, netmats, design, contrasts, nperms=5000, plot=True, title=None):
    """
    """

    nnodes     = ts.nnodes
    nsubjs     = netmats.shape[0]
    nedges     = netmats.shape[1]
    confile    = op.abspath(contrasts)
    design     = op.abspath(design)
    contrasts  = vest.loadVestFile(confile)
    ncontrasts = contrasts.shape[0]

    with tempdir():

        # TODO NIFTI2 required if nedges >= 32768
        netmats = netmats.T.reshape((nedges, 1, 1, nsubjs))

        Image(netmats).save('netmats')

        randomise('netmats', 'output', d=design, t=confile, n=nperms,
                  x=True, uncorrp=True, log={'tee' : False})

        puncorr = np.zeros((ncontrasts, nedges))
        pcorr   = np.zeros((ncontrasts, nedges))

        for con in range(ncontrasts):
            constr       = ' '.join([f'{c:g}' for c in contrasts[con]])
            tstat        = Image(f'output_tstat{con+1}')          .data.flatten()
            cpuncorr     = Image(f'output_vox_p_tstat{con+1}')    .data.flatten()
            cpcorr       = Image(f'output_vox_corrp_tstat{con+1}').data.flatten()
            puncorr[con] = cpuncorr
            pcorr[  con] = cpcorr

            tstat  = tstat .reshape((nnodes, nnodes))
            cpcorr = cpcorr.reshape((nnodes, nnodes))
            sig    = list(zip(*np.where(np.triu(cpcorr, 1) >= 0.95)))

            if len(sig) == 0:
                printTitle(f'Contrast {con+1} [{constr}] - no results')
                continue

            pvals = [cpcorr[i, j]   for i, j in sig]
            tvals = [tstat[ i, j]   for i, j in sig]
            nis   = [ts.nodes[s[0]] for s    in sig]
            njs   = [ts.nodes[s[1]] for s    in sig]
            rows  = reversed(sorted(zip(pvals, nis, njs, tvals)))

            pvals, nis, njs, tvals = zip(*rows)

            titles  = ['Node i', 'Node j', 'T statistic', 'P value']
            columns = [nis, njs, tvals, pvals]
            printTitle(f'Contrast {con+1} [{constr}]')
            printColumns(titles, columns)

    if plot:
        plot_pvalues(ts, pcorr, title)

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

        textbox = {'facecolor' : 'white',
                   'edgecolor' : 'red',
                   'alpha'     : 0.75,
                   'pad'       : 2}
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(1, 1, '(1-P) >= 0.95', transform=ax.transAxes, bbox=textbox, ha='right', va='top')
        ax.text(0, 0, '1-P',           transform=ax.transAxes, bbox=textbox, ha='left',  va='bottom')
        ax.set_title(f'Contrast {contrast+1}')
        ax.plot([0, 1], [1, 0], transform=ax.transAxes,
                c='#7777ff', linestyle='--')

    if title is not None:
        fig.suptitle(title)
    fig.colorbar(meshes[-1], ax=axes, label='1-P', location='right')
    fig.set_layout_engine('constrained')

    return fig
