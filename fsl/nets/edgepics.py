#!/usr/bin/env python
#
# edgepics.py - Show the strongest netmat elements from "netmat"
#
# Author:  Steve Smith, 2013-2014
#          Paul McCarthy, 2023
#


import numpy               as     np
import matplotlib          as     mpl
import matplotlib.pyplot   as     plt
import matplotlib.image    as     mplimg
import matplotlib.patches  as     patches
from   matplotlib.gridspec import GridSpec


def edgepics(ts, netmat, edgemat=None, nedges=6, title=None):
    """Show the strongest edges from netmat.
    """

    shape    = (ts.nnodes, ts.nnodes)
    haveedge = edgemat is not None

    if not haveedge:
        edgemat = netmat

    netmat  = netmat .reshape(shape)
    edgemat = edgemat.reshape(shape)

    # normalise the netmat used to size/colour edges
    normedgemat = -1 + 2 * (edgemat       - edgemat.min()) / \
                           (edgemat.max() - edgemat.min())

    # only consider upper triangle
    edges  = np.abs(np.triu(netmat, 1))
    edges  = np.flip(np.argsort(edges.ravel())[-nedges:])
    edges  = np.unravel_index(edges, shape)
    nedges = len(edges[0])

    # arrange edges in a grid, three columns,
    # and as many rows as needed. Edges are
    # shown as thumbnail pairs with an axhspan
    # plot between them depicting the edge
    # strength.
    thumbsz = 4
    edgesz  = 1
    gridsz  = edgesz + 2 * thumbsz + 1
    ncols   = 3
    nrows   = int(np.ceil(nedges / ncols))

    fig     = plt.figure()
    grid    = GridSpec(nrows, ncols * gridsz, figure=fig, hspace=0, wspace=0)

    for edgei, (i, j) in enumerate(zip(*edges)):

        nodei  = ts.nodes[i]
        nodej  = ts.nodes[j]
        thumbi = mplimg.imread(ts.thumbnail(nodei))
        thumbj = mplimg.imread(ts.thumbnail(nodej))

        row = edgei // ncols
        col = edgei %  ncols

        istart = col    * gridsz
        iend   = istart + thumbsz
        estart = iend
        eend   = estart + edgesz
        jstart = eend
        jend   = jstart + thumbsz

        iax = fig.add_subplot(grid[row, istart:iend])
        eax = fig.add_subplot(grid[row, estart:eend])
        jax = fig.add_subplot(grid[row, jstart:jend])

        # show thumbnails
        iax.imshow(thumbi, aspect='equal')
        jax.imshow(thumbj, aspect='equal')

        # make sure that edge axis has
        # same height as thumbnail axes
        eax.set_box_aspect((thumbsz + 1) / edgesz)

        # Draw thumbnails above edge axis
        # (see hspan hack below)
        iax.set_zorder(eax.get_zorder() + 1)
        jax.set_zorder(eax.get_zorder() + 1)

        # Extend the edge hspan a couple
        # of units beyond the axis limits,
        # as mpl will sometimes unavoidably
        # add gaps between subplots.
        normedgeval = normedgemat[i, j]
        halfedge    = np.abs(normedgeval) / 2
        rgb         = mpl.colormaps['coolwarm'](0.5 + np.sign(normedgeval) * halfedge)
        eax.axhspan(-halfedge, halfedge, -2, 2, clip_on=False, color=rgb)
        eax.set_xlim([-1, 1])
        eax.set_ylim([-1, 1])

        iax.axis('off')
        eax.axis('off')
        jax.axis('off')

        edgeval = edgemat[i, j]
        showval = netmat[ i, j]

        if showval <= 0: colour = 'blue'
        else:            colour = 'red'

        if haveedge: label = f'{showval:0.6f} [{edgeval:0.6f}]'
        else:        label = f'{showval:0.6f}'

        iax.set_title(f'Node {nodei}', pad=12)
        jax.set_title(f'Node {nodej}', pad=12)
        eax.set_title(label, fontsize=8, color=colour, pad=2)

    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, wspace=0, hspace=0)

    return fig
