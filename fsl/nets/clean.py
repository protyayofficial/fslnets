#!/usr/bin/env python
#
# clean.py - remove bad nodes, optionally regressing those out of the good
#            (for further cleanup)
#
# Author: Steve Smith and Ludo Griffanti, 2013-2014
#         Paul McCarthy, 2023
#

import numpy as np


def clean(ts, goodnodes, aggressive=False):
    """Remove bad nodes, optionally regressing their time courses out of the
    good node time courses.

    After calling this function, ts.nodes will contain only the good nodes,
    and ts.ts will be of shape (goodnodes, timepoints). You can convert a node
    label into a node index (to use in ts.ts) via the TimeSeries.node_index
    method.

    goodnodes:  List of IDs of nodes to keep

    aggressive: If True, the time courses of bad nodes are regressed from the
                time courses of good nodes. Otherwise the bad nodes are simply
                removed.
    """

    ts.goodnodes = goodnodes
    nnodes       = ts.goodnodes.sum()
    newts        = []

    for subj in range(ts.nsubjects):

        origts      = ts.origts[subj]
        nruns       = ts.nruns(subj)
        ntimepoints = ts.ntimepoints(subj)
        newsubjts   = np.zeros((nruns, ntimepoints, nnodes), dtype=np.float64)

        for run in range(nruns):

            runts  = np.array(origts[run], copy=True)
            goodts = runts[:, ts.goodnodes]
            badts  = runts[:, ts.badnodes]

            if aggressive:
                runts = goodts - badts @ (np.linalg.pinv(badts) @ goodts)
            else:
                runts = goodts

            newsubjts[run] = runts
            newts.append(newsubjts)

    ts.ts = newts
