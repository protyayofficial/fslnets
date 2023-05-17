#!/usr/bin/env python
#
# boxplots.py - show cross-subject boxplots, for a given "netmat" element
#               (IC1,IC2).
#
# Author: Steve Smith, 2013-2014
#         Paul McCarthy, 2023
#

import matplotlib          as     mpl
import matplotlib.pyplot   as     plt
import matplotlib.image    as     mplimg
import numpy               as     np
import scipy.stats         as     sps
from   matplotlib.gridspec import GridSpec


def boxplots(ts, netmats, nodei, nodej, groups, xlabel='Connectivity strength'):
    """
    """

    groupidxs = []
    for i, nsubjs in enumerate(groups):
        start = np.sum(groups[:i])
        groupidxs.append(np.arange(start, start + nsubjs, dtype=int))
    groups = groupidxs

    i    = ts.node_index(nodei)
    j    = ts.node_index(nodej)
    edge = i * ts.nnodes + j
    fig  = plt.figure()
    grid = GridSpec(2 + len(groups), 2, figure=fig)

    thumbi   = mplimg.imread(ts.thumbnail(nodei))
    thumbj   = mplimg.imread(ts.thumbnail(nodej))
    thumbiax = fig.add_subplot(grid[:2, 0])
    thumbjax = fig.add_subplot(grid[:2, 1])
    plotax   = fig.add_subplot(grid[2:, :])

    thumbiax.imshow(thumbi, aspect='equal')
    thumbjax.imshow(thumbj, aspect='equal')
    thumbiax.set_anchor('W')
    thumbjax.set_anchor('E')
    thumbiax.axis('off')
    thumbjax.axis('off')

    # we plot all distributions on a single axes,
    # normalising the y range of each to
    # [groupidx, groupidx+1] (but also adding a
    # gap of 0.2 between each distribution)
    gap = 0.2

    # colour each KDE from this cmap
    cmap = mpl.colormaps['Dark2']

    for i, group in enumerate(groups):

        colour = cmap.colors[i % len(cmap.colors)]
        data   = netmats[group, edge]

        # kernel density estmation of data distribution
        dmin   = np.nanmin(data)
        dmax   = np.nanmax(data)
        dlen   = dmax - dmin
        x      = np.linspace(dmin - dlen, dmax + dlen, 200)
        y      = sps.gaussian_kde(data)(x)

        # find lower/upper bounds of distribution
        # and truncate it there
        nonzero = np.sort(np.where(y != 0)[0])
        x       = x[nonzero[0]:nonzero[-1]]
        y       = y[nonzero[0]:nonzero[-1]]

        # vertical normalise + offset
        y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
        y = y + (i + i * gap)

        # Piot filled kde distribution
        # set z order so everything is above axis grid
        # (https://stackoverflow.com/questions/31506361/grid-zorder-seems-not-to-take-effect-matplotlib)
        baseline = np.full(len(x), i + i * gap)
        plotax.plot(x, y,                   color=colour, zorder=10)
        plotax.plot(x, baseline,            color=colour, zorder=10)
        plotax.fill_between(x, y, baseline, color=colour, zorder=10, alpha=0.7)

        # scatter every individual data point
        # across the x axis below the kde plot
        scaty = np.nanmin(y) - 0.1 * (np.nanmax(y) - np.nanmin(y))
        plotax.scatter(data, np.full(len(data), scaty), marker='x', color=colour)

    # group labels
    plotax.set_yticks(np.arange(0.5, len(groups) + len(groups) * gap, 1.2))
    plotax.set_yticklabels([f'Group {g+1}' for g in range(len(groups))], rotation=90, va='center')
    plotax.set_xlabel(xlabel)
    plotax.xaxis.grid(True)

    fig.suptitle(f'Per-group connectivity between nodes {nodei} and {nodej}')
    fig.set_layout_engine('constrained')

    return fig
