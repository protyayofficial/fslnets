#!/usr/bin/env python
#
# groupmean.py - estimate group mean/one-group-t-test and consistency of
#                netmats across runs/subjects
#
# Author: Steve Smith, 2012-2014
#         Paul McCarthy, 2023
#


import numpy             as np
import scipy.stats       as stats
import matplotlib.pyplot as plt
import seaborn           as sns


def groupmean(ts, netmats, plot=True, title=None):
    """Perform a one-sample T-test on all subject netmats, and return
    Z-statistic and mean connectivity matrices.
    """
    nsubjs = netmats.shape[0]
    nedges = netmats.shape[1]
    dof    = nsubjs - 1
    mean   = netmats.mean(axis=0)
    std    = netmats.std( axis=0)
    tvals  = np.sqrt(nsubjs) * mean / std
    tpos   = tvals > 0
    tneg   = tvals < 0

    zvals       = np.zeros(tvals.shape)
    zvals[tpos] = -stats.norm.ppf(stats.t.cdf(-tvals[tpos], dof))
    zvals[tneg] =  stats.norm.ppf(stats.t.cdf( tvals[tneg], dof))
    zinf        = ~np.isfinite(zvals)
    zvals[zinf] = tvals[zinf]

    if np.sqrt(nedges) == ts.nnodes:
        zvals = zvals.reshape((ts.nnodes, ts.nnodes))
        mean  = mean .reshape((ts.nnodes, ts.nnodes))

    if plot:
        plot_groupmean(ts, zvals, mean, netmats, title)

    return zvals, mean


def plot_groupmean(ts, zvals, mean, netmats, title):
    """Plot mean connectivity Z value matrix, and a scatter plot of subject
    connectivity vs mean connectivity.
    """

    fig     = plt.figure()
    fig.set_layout_engine('compressed')
    zstatax = fig.add_subplot(1, 2, 1)
    scatax  = fig.add_subplot(1, 2, 2)

    zmax  = np.nanmax(np.abs(zvals))
    zvals = np.flipud(zvals)

    ticklbls = [0, ts.nodes[-1]]
    ticks    = [ts.node_index(t) for t in ticklbls]

    m = zstatax.pcolormesh(zvals, cmap='jet', vmin=-zmax, vmax=zmax)
    fig.colorbar(m, ax=zstatax, label='Z statistic', location='left')
    zstatax.set_xticks(ticks)
    zstatax.set_yticks(ticks)
    zstatax.set_xticklabels(ticklbls)
    zstatax.set_yticklabels(ticklbls)
    zstatax.set_title('Overall connectivity (Z-stat from one-group t-test)')
    zstatax.set_xlabel('Node')
    zstatax.set_ylabel('Node')

    mean = np.tile(mean.flatten(), (netmats.shape[0], 1))

    sns.kdeplot({'mean' : mean.flatten(), 'subject' : netmats.flatten()},
                x='subject', y='mean', ax=scatax, fill=True, levels=50,
                thresh=0, cmap='gist_heat_r')

    smin, smax = np.percentile(netmats, (1, 99))
    scatax.set_xlim((smin, smax))
    scatax.set_ylim((smin, smax))
    scatax.yaxis.tick_right()
    scatax.yaxis.set_label_position('right')

    scatax.set_xlabel('Subject connectivity')
    scatax.set_ylabel('Mean connectivity')
    scatax.set_title('Subject connectivity vs mean connectivity')

    if title is not None:
        fig.suptitle(title)

    fig.show()
