#!/usr/bin/env python
#
# hierarchy.py - create hierarchical clustering figure
#
# Author: Steve Smith, 2012-2014
#         Paul McCarthy, 2023
#


import numpy                   as     np
import scipy.cluster.hierarchy as     sch
import matplotlib.pyplot       as     plt
import matplotlib.image        as     mplimg
from   matplotlib.gridspec     import GridSpec


def hierarchy(netmat):
    """Perform hierarchical clustering on the given netmat.
    Return a linkage array containing clustering results.
    """

    # zero negative entries - seems to give nicer hierarchies
    clusternet = np.copy(netmat)
    clusternet[clusternet < 0] = 0

    # normalise w.r.t. outliers again, clamp to -1, 1
    # normalise to -0.5, 0.5 and invert
    triu                        = np.triu_indices(netmat.shape[0], 1)
    norm                        = np.percentile(np.abs(clusternet), 99)
    clusternet                  = clusternet / norm
    clusternet[clusternet < -1] = -1
    clusternet[clusternet >  1] =  1
    clusternet                  = 0.5 - clusternet / 2
    clusternet                  = clusternet[triu].flatten()

    return sch.linkage(clusternet, method='ward', optimal_ordering=True)


def plot_hierarchy(ts, netmatl, netmath=None, lowlabel=None, highlabel=None):
    """Perform hierarchical clustering and display a dendrogram and mean
    connectivity matrix re-ordered according to the clustering results.

    ts:        TimeSeries object

    netmatl:   (nodes, nodes) array used to drive the clustering. This is
               typically the z-stat from the one-group-t-test across all
               subjects' netmats

    netmath:   is the net matrix shown above the diagonal, for example
               partial correlation

    lowlabel:  Label for netmatl (e.g. "Full correlation")

    highlabel: Label for netmath (e.g. "Partial correlation")

    Returns:

      - List of re-ordered node indices - *not* the original node labels if
        nets.clean was performed. You can convert a node index into the
        corresponding node label via the TimeSeries.nodes list.

      - Linkage array describing the clustering (see
        scipy.cluster.hierarchy.linkage)

      - Matplotlib Figure object.
    """

    if netmath is None:
        netmath = netmatl

    # normalise w.r.t. outliers
    triu    = np.triu_indices(netmatl.shape[0], 1)
    norm    = np.percentile(np.abs(netmatl[triu]), 99)
    netmatl = netmatl / norm
    netmath = netmath / norm

    link = hierarchy(netmatl)

    # dendrogram takes up 13% height,
    # nodepics 7%, matrix 80%
    fig  = plt.figure()
    grid = GridSpec(15, ts.nnodes, figure=fig)

    dendax = fig.add_subplot(grid[ :2, :])
    matax  = fig.add_subplot(grid[3:,  :])
    dend   = sch.dendrogram(link, ax=dendax, no_labels=True, show_leaf_counts=False)
    nodes  = list(dend['leaves'])

    dendax.axis('off')
    dendax.set_xticks([])
    dendax.set_yticks([])

    for i, node in enumerate(nodes):
        node      = ts.nodes[node]
        thumbnail = ts.thumbnail(node)
        if thumbnail is None:
            continue
        thumbnail = mplimg.imread(thumbnail)
        thumbax   = fig.add_subplot(grid[2, i])
        thumbax.imshow(thumbnail, aspect='equal')
        thumbax.autoscale(enable=True, tight=True)
        thumbax.set_xticks([])
        thumbax.set_yticks([])
        thumbax.set_xlabel(str(node), fontsize=8)
        thumbax.set_anchor('N')

    tril         = np.tril_indices(netmatl.shape[0], -1)
    triu         = np.triu_indices(netmatl.shape[0],  1)
    matrix       = np.zeros(netmatl.shape)
    matrix[tril] = netmatl[tril]
    matrix[triu] = netmath[triu]
    matrix[matrix < -0.95] = -0.95
    matrix[matrix >  0.95] =  0.95

    matrix[np.diag_indices(matrix.shape[0])] = np.nan

    matrix = matrix[nodes, :][:, nodes]
    matrix = np.flipud(matrix.T)

    matax.pcolormesh(matrix, cmap='jet', vmin=-1, vmax=1)
    matax.plot([0, 1], [1, 0], transform=matax.transAxes, c='#000000')

    matax.set_xticks([])
    matax.set_yticks([])

    if lowlabel is not None:
        matax.set_xlabel(lowlabel)
    if highlabel is not None:
        matax.set_ylabel(highlabel)
        matax.yaxis.set_label_position('right')

    fig.suptitle('Netmat hierarchical clustering summary')
    fig.subplots_adjust(0.02, 0.03, 0.98, 0.95, 0, 0)

    return nodes, link, fig
