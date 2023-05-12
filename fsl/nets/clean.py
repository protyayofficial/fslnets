#!/usr/bin/env python
#
# clean.py - remove bad nodes, optionally regressing those out of the good
#            (for further cleanup)
#
# Author: Steve Smith and Ludo Griffanti, 2013-2014
#         Paul McCarthy, 2023
#

import numpy as np

def clean(ts, aggressive=False):

    newts = np.zeros((ts.ntimepoints(), ts.goodnodes.sum()), dtype=np.float64)

    for subj in range(ts.nsubjects):
        start  = subj  * ts.ntimepoints(subj)
        end    = start + ts.ntimepoints(subj)
        subjts = np.array(ts.origts[start:end, :], copy=True)

        goodts = subjts[:, ts.goodnodes]
        badts  = subjts[:, ts.badnodes]

        if aggressive:
            subjts = goodts - badts @ (np.linalg.pinv(badts) @ goodts)
        else:
            subjts = goodts

        newts[start:end, :] = subjts

    ts.ts = newts
