#!/usr/bin/env python
#
# spectra.py - calculate and display spectrum for each node, averaged across subjects
#
# Author: Steve Smith and Ludo Griffanti, 2013-2014
#         Paul McCarthy, 2023
#

import numpy               as     np
import numpy.random        as     rnd
import numpy.fft           as     fft
import scipy.signal        as     signal
import matplotlib.pyplot   as     plt
import matplotlib.image    as     mplimg
from   matplotlib.gridspec import GridSpec


def node_spectrum(ts, node, windowlen):
    """Calculate the power spectrum for the time series of the specified node.
    """

    spectra = []
    nodeidx = ts.node_index(node)

    for _, _, _, data in ts.allts:
        data     = data[:, nodeidx]
        data     = data - np.nanmean(data)
        spectrum = np.abs(fft.fft(data, windowlen))
        spectrum = spectrum[:round(windowlen / 2)]
        spectra.append(spectrum)

    # average node spectrum across subjects
    return np.mean(spectra, axis=0)


def plot_spectra(ts, ncols=4, nodes=None):
    """Calculate and display the power spectrum for each node, averaged across
    subjects.

    ts:    TimeSeries object
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

    # Window length for FFT calculation -
    # we use the same window length for
    # all data sets, to accommodate
    # varying numbers of timepoints.
    windowlen = max(ts.ntimepoints(subj) for subj in range(ts.nsubjects))
    freqs     = fft.rfftfreq(windowlen - 1, ts.tr)

    # And each node plot actually spans six
    # columns (1=thumbnail, 5=spectrum), so
    # we actually have ncols*6 columns.
    gridsz      = 6
    fig         = plt.figure()
    grid        = GridSpec(meanrows + nrows, ncols * gridsz, figure=fig)
    meanax      = fig.add_subplot(grid[nrows:nrows + meanrows, :])
    spectra     = [node_spectrum(ts, n, windowlen) for n in nodes]
    spectra     = [s / s.max() for s in spectra]
    meanspectra = np.median(spectra, axis=0)

    for i, node in enumerate(nodes):

        spectrum = spectra[i]
        row      = i %  nrows
        col      = i // nrows
        cstart   = col    * gridsz
        cend     = cstart + gridsz
        thumbax  = fig.add_subplot(grid[row, cstart])
        plotax   = fig.add_subplot(grid[row, cstart + 1:cend])

        plotax.plot(freqs, spectrum,    color=f'C{i}', lw=2)
        plotax.plot(freqs, meanspectra, color='k',     lw=1,   alpha=0.5)
        meanax.plot(freqs, spectrum,    color=f'C{i}', lw=0.5, alpha=0.5)

        thumbnail = mplimg.imread(ts.thumbnail(node))
        thumbax.imshow(thumbnail, aspect='equal')
        thumbax.set_anchor('E')

        for ax in [thumbax, plotax]:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

        thumbax.set_title(str(node), x=0, y=0.5, fontsize=8,
                          verticalalignment='top',
                          horizontalalignment='right')

    meanax.plot(freqs, meanspectra, color='k', lw=2)
    meanax.autoscale(enable=True, tight=True)
    meanax.spines['right'].set_visible(False)
    meanax.spines['top']  .set_visible(False)
    meanax.set_yticks([])
    meanax.set_ylabel('Amplitude')
    meanax.set_xlabel('Frequency (Hz)')
    meanax.set_title('Median across all nodes', y=0)

    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0, 0)
    fig.suptitle('Node time series power spectra')
    return fig


def plot_timeseries(ts, ncols=3, nodes=None, subjects=None):
    """Calculate and display the time series for the specified nodes and subjects.

    ts:       TimeSeries object
    ncols:    Number of columns in which to arrange the per-node power spectra
    nodes:    Sequence of nodes to include (default: 30 randomly selected nodes)
    subjects: Subjects to include (default: 4 randomly selected subjects)
    """

    if nodes is None:
        nnodes = min(ts.nnodes, 30)
        nodes  = sorted(rnd.choice(ts.nodes, nnodes, replace=False))
    if subjects is None:
        nsubjs   = min(ts.nsubjects, 4)
        subjects = sorted(rnd.choice(np.arange(ts.nsubjects), nsubjs, replace=False))

    # See comments in plot_spectra for an
    # overview of the grid layout logic.
    nnodes = len(nodes)
    nrows  = int(np.ceil(nnodes / ncols))
    gridsz = 6
    fig    = plt.figure()
    grid   = GridSpec(nrows, ncols * gridsz, figure=fig)

    for i, node in enumerate(nodes):

        row      = i %  nrows
        col      = i // nrows
        cstart   = col    * gridsz
        cend     = cstart + gridsz
        thumbax  = fig.add_subplot(grid[row, cstart])
        plotax   = fig.add_subplot(grid[row, cstart + 1:cend])

        nodeidx = ts.node_index(node)
        for voff, subj in enumerate(subjects):
            for run in range(ts.nruns(subj)):
                data = ts.ts[subj][run, :, nodeidx]
                data = (data - data.min()) / (data.max() - data.min())
                plotax.plot(voff + data, color='k', alpha=0.5, lw=0.5)

        thumbnail = mplimg.imread(ts.thumbnail(node))
        thumbax.imshow(thumbnail, aspect='equal')
        thumbax.set_anchor('E')

        thumbax.axis('off')
        plotax.tick_params(axis='both', which='both', left=False, bottom=False)
        plotax.tick_params(axis='y', direction='in', pad=-20)
        plotax.spines['left']  .set_visible(False)
        plotax.spines['right'] .set_visible(False)
        plotax.spines['bottom'].set_visible(False)
        plotax.spines['top']   .set_visible(False)
        plotax.set_xticks([])
        plotax.set_yticks(np.arange(0.5, len(subjects) + 0.5))
        plotax.set_yticklabels(map(str, subjects), va='center')

        thumbax.set_title(str(node), x=0, y=0.5, fontsize=8,
                          verticalalignment='top',
                          horizontalalignment='right')

    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0, 0)
    fig.suptitle('Node time series')
    return fig
