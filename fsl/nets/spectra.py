#!/usr/bin/env python
#
# spectra.py - calculate and display spectrum for each node, averaged across subjects
#
# Author: Steve Smith and Ludo Griffanti, 2013-2014
#         Paul McCarthy, 2023
#

import numpy               as     np
import numpy.fft           as     fft
import matplotlib.pyplot   as     plt
import matplotlib.image    as     mplimg
from   matplotlib.gridspec import GridSpec


def node_spectrum(ts, node):
    """Calculate the power spectrum for the time series of the specified node.
    """

    # TODO This won't work for varying #timepoints?

    spectra = []
    nodeidx = ts.node_index(node)

    for i, subj, run, data in ts.allts:
        data     = ts.ts[subj][run, :, nodeidx]
        data     = data - np.nanmean(data)
        spectrum = np.abs(fft.fft(data))
        npts     = round(len(spectrum) / 2)
        spectrum = spectrum[:npts]
        spectra.append(spectrum)

    # average node spectrum across subjects
    return np.mean(spectra, axis=0)


def plot_spectra(ts, ncols=4, nodes=None):
    """Calculate and display the power spectrum for each node, averaged across
    subjects.

    ts:    TimeSeries objectg
    ncols: Number of columns in which to arrange the per-node power spectra
    nodes: Sequence of nodes to include (default: all good nodes)
    """

    if nodes is None:
        nodes = list(ts.nodes)

    nnodes = len(nodes)

    # Node plots are distributed over ncols
    # columns, and as many rows as needed.
    nrows = int(np.ceil(nnodes / ncols))

    # Mean plot is at the bottom, spans all
    # columns, and spans 25% of the height
    meanrows = int(np.ceil(0.50 * nrows))

    # And each node plot actually spans six
    # columns (1=thumbnail, 5=spectrum), so
    # we actually have ncols*6 columns.
    gridsz      = 6
    fig         = plt.figure()
    grid        = GridSpec(meanrows + nrows, ncols * gridsz, figure=fig)
    meanax      = fig.add_subplot(grid[nrows:nrows + meanrows, :])
    spectra     = [node_spectrum(ts, n) for n in nodes]
    meanspectra = np.median(spectra, axis=0)

    for i, node in enumerate(nodes):

        spectrum = spectra[i]
        row      = i %  nrows
        col      = i // nrows
        cstart   = col    * gridsz
        cend     = cstart + gridsz
        thumbax  = fig.add_subplot(grid[row, cstart])
        plotax   = fig.add_subplot(grid[row, cstart + 1:cend])

        plotax.plot(spectrum,    color=f'C{i}', lw=2)
        plotax.plot(meanspectra, color='k',     lw=1, alpha=0.5)
        meanax.plot(spectrum,    color=f'C{i}', lw=0.5, alpha=0.5)

        thumbnail = mplimg.imread(ts.thumbnail(node))
        thumbax.imshow(thumbnail, aspect='equal')
        thumbax.set_anchor('E')

        for ax in [thumbax, plotax]:
            if ax is not None:
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])

        if thumbax is not None: titleax = thumbax
        else:                   titleax = plotax
        titleax.set_title(str(node), x=0, y=0.5, fontsize=8,
                          verticalalignment='top',
                          horizontalalignment='right')

    meanax.plot(np.median(spectra, axis=0), color='k', lw=2)
    meanax.autoscale(enable=True, tight=True)
    meanax.spines['right'].set_visible(False)
    meanax.spines['top']  .set_visible(False)
    meanax.set_xticks([])
    meanax.set_yticks([])
    meanax.set_ylabel('Amplitude')
    meanax.set_xlabel('Frequency')
    meanax.set_title('Median across all nodes', y=0)

    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0, 0)
    fig.suptitle('Node time series power spectra')
    return fig
