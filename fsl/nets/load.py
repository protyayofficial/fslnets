#!/usr/bin/env python
#
# load.py - load a folder full of individual runs'/subjects' node-timeseries
#           files
#
# Author: Steve Smith and Ludo Griffanti, 2013-2014
# Author: Paul McCarthy, 2023
#


import                        glob
import os.path         as     op
from   pathlib         import Path
from   collections.abc import Sequence

import numpy   as np
import nibabel as nib


def load(infiles,
         tr,
         meldir=None,
         varnorm=0,
         nruns=1,
         demean=True):

    if isinstance(infiles, (str, Path)):
        infiles = sorted(glob.glob(op.join(infiles, 'dr_stage1_*.txt')))

    nsubjects = len(infiles)

    if not isinstance(nruns, Sequence):
        nruns = [nruns] * nsubjects

    timeseries = [np.loadtxt(f, dtype=np.float64) for f in infiles]

    for i, (ts, nr) in enumerate(zip(timeseries, nruns)):

        ntimepoints = ts.shape[0]
        runlen      = ntimepoints // nr

        if ntimepoints % nr != 0:
            ValueError(f'[{infiles[i]}: ntimepoints ({ntimepoints}) '
                       f'must be a multiple of nruns ({nr})!')

        if demean:
            ts = ts - ts.mean(axis=0)
        # normalise across all runs
        if varnorm == 1:
            ts = ts / ts.std(axis=0)
        # normalise time series from each run separately
        elif varnorm == 2:
            for i in range(nr):
                start         = i     * runlen
                end           = start + runlen
                runts         = ts[start:end]
                ts[start:end] = runts / runts.std(axis=0)

        timeseries[i] = ts

    timeseries = np.vstack(timeseries)

    return TimeSeries(timeseries, tr, meldir, nsubjects,  nruns)


class TimeSeries:

    def __init__(self, ts, tr, meldir, nsubjects, nruns):
        self.__ts            = ts
        self.__origts        = np.copy(ts)
        self.__tr            = tr
        self.__meldir        = meldir
        self.__nsubjects     = nsubjects
        self.__nruns         = nruns
        self.__orignnodes    = ts.shape[1]
        self.__goodmask      = np.ones( ts.shape[1], dtype=bool)
        self.__unknownmask   = np.zeros(ts.shape[1], dtype=bool)

    @property
    def nnodes(self):
        return self.__goodmask.sum()

    @property
    def orignnodes(self):
        return self.__orignnodes

    @property
    def nsubjects(self):
        return self.__nsubjects

    def ntimepoints(self, subjidx=None):
        if subjidx is None:
            return self.__ts.shape[0]
        else:
            return self.__ts.shape[0] // self.nsubjects

    def nruns(self, subjidx):
        return self.__nruns[subjidx]

    @property
    def goodnodes(self):
        return self.__goodmask

    @property
    def unknownnodes(self):
        return self.__unknownmask

    @property
    def badnodes(self):
        return ~(self.goodnodes | self.unknownnodes)

    @property
    def origts(self):
        return np.copy(self.__origts)

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, newts):
        self.__ts = newts

    def subjslice(self, subjidx):
        """
        """
        start  = subjidx * self.ntimepoints(subjidx)
        end    = start   + self.ntimepoints(subjidx)
        return slice(start, end, None)


    def subjts(self, subjidx):
        """
        """
        return self.ts[self.subjslice(subjidx)]


    @goodnodes.setter
    def goodnodes(self, nodes):
        nodes = np.asanyarray(nodes) - 1

        if ((nodes < 0) | (nodes >= self.nnodes)).any():
            raise ValueError(f'Invalid node indices (< 0 or > {self.nnodes})')

        mask            = np.zeros(self.ts.shape[1], dtype=bool)
        mask[nodes]     = True
        self.__goodmask = mask

    @unknownnodes.setter
    def unknownnodes(self, nodes):
        nodes = np.asanyarray(nodes) - 1

        if ((nodes < 0) | (nodes >= self.nnodes)).any():
            raise ValueError(f'Invalid node indices (< 0 or > {self.nnodes})')

        mask               = np.zeros(self.ts.shape[1], dtype=bool)
        mask[nodes]        = True
        self.__unknownmask = mask
