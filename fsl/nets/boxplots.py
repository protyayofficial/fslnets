#!/usr/bin/env python
#
# boxplots.py - Show most significant edges, and per-group edge strength
#               distributions.
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


def boxplots(ts, netmats, znetmat, pnetmat, groups=None, nedges=6, edges=None):
    """Show per-group distributions of connectivity strength for the <nedges>
    most significant edges, or for the specified <edges>.

    ts:      TimeSeries object
    netmats: (subjects, edges) array containing per-subject netmats.
    znetmat: (nodes, nodes) array containing mean netmat values (assumed to be
             Z-normalised)
    pnetmat  (nodes, nodes) array containing P-values for each edge
    groups:  Number of subjects in each group
    nedges:  show the N strongest edges.
    edges:   Show these specific edges - sequence of (nodei, nodej) pairs.
    """

    if groups is None:
        groups = (int(np.floor(netmats.shape[0] / 2)),
                  int(np.ceil( netmats.shape[0] / 2)))

    shape   = (ts.nnodes, ts.nnodes)
    znetmat = znetmat.reshape(shape)
    pnetmat = pnetmat.reshape(shape)

    # If no specific edges have been given,
    # show the <nedges> strongest edgbes
    # according to meannetmat
    if edges is None:
        edges  = np.abs(np.triu(pnetmat, 1))
        edges  = np.flip(np.argsort(edges.ravel())[-nedges:])
        edges  = np.unravel_index(edges, shape)
        edges  = [(ts.nodes[i], ts.nodes[j]) for i, j in zip(*edges)]
        nedges = len(edges)

    # number of columns taken up by thumbnails,
    # edge strength, and boxplots.
    tlen  = 3
    elen  = 1
    plen  = 25
    nrows = len(edges)

    fig      = plt.figure()
    grid     = GridSpec(nrows, 2 * tlen + elen + plen)
    plotaxes = []

    for row, edge in enumerate(edges):

        nodei, nodej = edge
        i            = ts.node_index(nodei)
        j            = ts.node_index(nodej)
        nodeiax      = fig.add_subplot(grid[row,                :    tlen])
        edgeax       = fig.add_subplot(grid[row,     tlen       :    tlen + elen])
        nodejax      = fig.add_subplot(grid[row,     tlen + elen:2 * tlen + elen])
        plotax       = fig.add_subplot(grid[row, 2 * tlen + elen:])

        edgepic(ts, edge, znetmat, axes=(nodeiax, edgeax, nodejax))
        boxplot(ts, edge, netmats, groups, ax=plotax)

        zval = znetmat[i, j]
        pval = pnetmat[i, j]
        text = [f'Edge between nodes {nodei} and {nodej}',
                f'[Mean edge strength: {zval:0.4f}]',
                f'[P value: {pval:0.4f}]']
        text = '        '.join(text)

        textbox = {'facecolor' : '#cccccc',
                   'edgecolor' : '#4444ee',
                   'alpha'     : 0.5,
                   'pad'       : 2}
        plotax.set_title(text, fontsize=8, x=0.01, y=1,
                         va='top', ha='left', pad=-5,
                         bbox=textbox, zorder=9999)
        plotaxes.append(plotax)

    # Make all boxplot limits equal
    xlim = np.array([a.get_xlim() for a in plotaxes])
    xmin = xlim.min()
    xmax = xlim.max()
    for i, p in enumerate(plotaxes):
        if i < len(plotaxes) - 1:
            p.set_xticklabels([])
            p.xaxis.set_tick_params(which='both', size=0)
        else:
            p.set_xlabel('Edge strength')
        p.set_xlim(xmin, xmax)


    fig.suptitle(f'Most significant edges')
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, wspace=0, hspace=0)

    return fig


def edgepic(ts, edge, netmat, axes=None):
    """Show thumbnails for the given edge, and a line between them depicting
    the edge strength.

    ts:     TimeSeries object
    edge:   (nodei, nodej) pair
    netmat: (nodes, nodes) array containing edge strengths
    axes:   (nodei, edge, nodej) matplotlib axes. If not provided, a new figure
            is created and returned.
    """

    shape   = (ts.nnodes, ts.nnodes)
    ownfig  = axes is None

    # normalise the netmat used to size/colour edges to [-1, 1]
    netmat = netmat.reshape(shape)
    netmat = -1 + 2 * (netmat       - netmat.min()) / \
                      (netmat.max() - netmat.min())

    if ownfig:
        thumbcols = 4
        edgecols  = 1
        fig       = plt.figure()
        grid      = GridSpec(1, 2 * thumbcols + edgecols, figure=fig)
        iax       = fig.add_subplot(grid[0,  :4])
        eax       = fig.add_subplot(grid[0, 4:5])
        jax       = fig.add_subplot(grid[0, 5:])
    else:
        fig           = None
        iax, eax, jax = axes

    nodei, nodej = edge
    i            = ts.node_index(nodei)
    j            = ts.node_index(nodej)
    thumbi       = mplimg.imread(ts.thumbnail(nodei))
    thumbj       = mplimg.imread(ts.thumbnail(nodej))

    # show thumbnails
    iax.imshow(thumbi, aspect='equal')
    jax.imshow(thumbj, aspect='equal')
    iax.set_anchor('E')
    jax.set_anchor('W')

    # make sure that edge axis has same
    # height as thumbnail axes take aspect
    # ratio from subplot layout -
    # (thumbcols + edgecols) / 1)
    eax.set_box_aspect(5)

    # Draw thumbnails above edge axis
    # (see hspan hack below)
    iax.set_zorder(eax.get_zorder() + 1)
    jax.set_zorder(eax.get_zorder() + 1)

    # Extend the edge hspan a couple
    # of units beyond the axis limits,
    # as mpl will sometimes unavoidably
    # add gaps between subplots.
    edgeval  = netmat[i, j]
    halfedge = np.abs(edgeval) / 2
    rgb      = mpl.colormaps['coolwarm'](0.5 + np.sign(edgeval) * halfedge)
    eax.axhspan(-halfedge, halfedge, -2, 2, clip_on=False, color=rgb)
    eax.set_xlim([-1, 1])
    eax.set_ylim([-1, 1])

    eax.axis('off')
    iax.spines['left']  .set_visible(False)
    iax.spines['right'] .set_visible(False)
    iax.spines['top']   .set_visible(False)
    iax.spines['bottom'].set_visible(False)
    jax.spines['left']  .set_visible(False)
    jax.spines['right'] .set_visible(False)
    jax.spines['top']   .set_visible(False)
    jax.spines['bottom'].set_visible(False)
    iax.set_xticks([])
    iax.set_yticks([])
    jax.set_xticks([])
    jax.set_yticks([])

    return fig


def boxplot(ts, edge, netmats, groups, ax=None):
    """Show box plots (actually KDE plots) depicting per-group connectivity
    strength for the specified edge.

    ts:      TimeSeries object
    edge:    (nodei, nodej) pair
    netmats: (nsubjects, nedges) array containing per-subject netmats
    groups:  Number of subjects in each group
    ax:      Axis to plot on. If not provided, a new figure is created and
             returned.
    """

    ownfig = ax is None

    # Convert group sizes into sets
    # of indices for each group
    groupidxs = []
    for i, nsubjs in enumerate(groups):
        start = np.sum(groups[:i])
        groupidxs.append(np.arange(start, start + nsubjs, dtype=int))
    groups = groupidxs

    nodei, nodej = edge
    i            = ts.node_index(nodei)
    j            = ts.node_index(nodej)
    edge         = i * ts.nnodes + j

    if ownfig:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
    else:
        fig = None

    # we plot all distributions on a single axes,
    # normalising the y range of each to
    # [groupidx, groupidx+1] (but also adding a
    # gap of 0.2 between each distribution)
    gap = 0.2

    # colour each KDE from this cmap (must be a
    # ListedColorMap)
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
        ax.plot(x, y,                   color=colour, zorder=10)
        ax.plot(x, baseline,            color=colour, zorder=10)
        ax.fill_between(x, y, baseline, color=colour, zorder=10, alpha=0.7)

        # scatter every individual data point
        # across the x axis below the kde plot
        scaty = np.nanmin(y) - 0.1 * (np.nanmax(y) - np.nanmin(y))
        ax.scatter(data, np.full(len(data), scaty), marker='o', alpha=0.2, color=colour)

    # The boxplots function adds a title inside the
    # top of the axis, so expand the upper y limit a bit
    ylim = ax.get_ylim()
    ax.set_ylim((ylim[0], ylim[1] + 0.5))

    # group labels
    ax.set_yticks(np.arange(0.5, len(groups) + len(groups) * gap, 1.2))
    ax.set_yticklabels([f'Group {g+1}' for g in range(len(groups))], rotation=90, va='center')
    ax.yaxis.tick_right()
    ax.xaxis.grid(True)

    return fig
