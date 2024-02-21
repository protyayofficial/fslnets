#!/usr/bin/env python
#
# glm.py - do cross-subject GLM on a set of netmats
#
# Author: Steve Smith and Ludo Griffanti, 2013-2014
#         Paul McCarthy, 2023
#

import                          os
import os.path           as     op
import matplotlib.pyplot as     plt
import numpy             as     np
import fsl.data.vest     as     vest
from   fsl.data.image    import Image
from   fsl.utils.tempdir import tempdir
from   fsl.utils.run     import runfsl, hold
from   fsl.wrappers      import randomise
from   fsl.nets.util     import printTitle, printColumns


def glm(ts, netmats, design, contrasts, nperms=5000, plot=True, title=None):
    """Perform a cross-subject GLM on a set of netmats, giving uncorrected and
    corrected (1-p) values.

    Randomise (permutation testing) is used to get corrected 1-p-values (i.e.,
    correcting for multiple comparisons across the NxN netmat elements).

    ts:        TimeSeries object

    netmats:   (subjects, edges) array containing per-subject netmats.

    design:    Path to a FSL design matrix file denoting subject groups. The
               rows must be in the same order as the subject order in netmats.

    contrasts: Path to a FSL contrast file specifying the contrasts to test.

    nperms:    Number of non-parametric permutations to apply

    plot:      Display a (nodes, nodes) matrix of P-values (one per contrast)
               highlighting edges that were found to have significantly
               different strength.

               1-corrected-p values are shown below the diagonal. The same
               is shown above the diagonal, but thresholded at 0.95,i.e.
               corrected-p < 0.05

    title:     Plot title

    Returns two (ncontrasts, nedges) arrays, containing the corrected and
    uncorrected p-values.
    """

    nnodes     = ts.nnodes
    nsubjs     = ts.nsubjects
    nedges     = netmats.shape[1]
    confile    = op.abspath(contrasts)
    design     = op.abspath(design)
    contrasts  = np.atleast_2d(vest.loadVestFile(confile))
    ncontrasts = contrasts.shape[0]

    # Average netmats within subject across runs
    avgmats = np.zeros((nsubjs, nedges))
    for subj in range(ts.nsubjects):
        idxs          = ts.datasets(subj)
        avgmats[subj] = netmats[idxs].mean(axis=0)
    netmats = avgmats

    # Store files cwd, in case we are
    # running on a cluster where $TMPDIR
    # may not be shared between nodes
    with tempdir(root=os.getcwd(), prefix='.fslnets', changeto=False) as tdir:

        netmats = netmats.T.reshape((nedges, 1, 1, nsubjs))
        nmfile  = op.join(tdir, 'netmats')
        outpref = op.join(tdir, 'output')

        # Save as NIfTI2 to support large numbers
        # of edges/subjects/timepoints
        Image(netmats, version=2).save(nmfile)

        hold(randomise(nmfile, outpref, d=design, t=confile, n=nperms,
                       x=True, uncorrp=True, submit=True))

        puncorr = np.zeros((ncontrasts, nedges))
        pcorr   = np.zeros((ncontrasts, nedges))

        for con in range(ncontrasts):
            constr       = ' '.join([f'{c:g}' for c in contrasts[con]])
            tstat        = Image(f'{outpref}_tstat{con+1}')          .data.flatten()
            cpuncorr     = Image(f'{outpref}_vox_p_tstat{con+1}')    .data.flatten()
            cpcorr       = Image(f'{outpref}_vox_corrp_tstat{con+1}').data.flatten()
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
    """Plot a (nnodes, nnodes) matrix assumed to contain P values.

    ts:    TimeSeries object
    pvals: (nodes, nodes) array containing P values.
    title: Plot title
    """
    ncontrasts = pvals.shape[0]
    fig        = plt.figure()
    shape      = (ts.nnodes, ts.nnodes)

    tril = np.zeros(shape, dtype=bool)
    triu = np.zeros(shape, dtype=bool)
    tril[np.tril_indices(shape[0], -1)] = True
    triu[np.triu_indices(shape[0],  1)] = True

    meshes = []
    axes   = fig.subplots(1, ncontrasts)
    if ncontrasts == 1:
        axes = [axes]

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
