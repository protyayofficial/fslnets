#!/usr/bin/env python
#
# test_spectra.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#


import numpy as np

from fsl import nets
from . import create_random_timeseries


def test_plot_spectra():
    """A simple sanity-check test to make sure that the plot_spectra function
    runs to completion on a range of different data sets.
    """

    # nsubjects, nnodes, ntimepoints, nruns
    tests = [
        (10, 50, 50, 1),
        (10, 50, 51, 1),
        (5,  50, [49, 51, 45, 50, 45, 1], 1),
        (5,  50, [49, 51, 45, 50, 45, 1], [2,1,3,2,2]),
    ]

    for nsubjs, nnodes, ntimepoints, nruns in tests:
        with create_random_timeseries(1.2, nsubjs, nnodes,
                                      ntimepoints, nruns) as ts:
            fig = nets.plot_spectra(ts)

            # <nnodes> spectra plots
            # <nnodes> thumbnail plots
            # 1        mean plot
            assert len(fig.get_axes()) == nnodes * 2 + 1


def test_plot_timeseries():
    """A simple sanity-check test to make sure that the plot_timeseries
    function runs to completion on a range of different data sets.
    """

    # nsubjects, nnodes, ntimepoints, nruns
    tests = [
        (10, 50, 50, 1),
        (10, 50, 51, 1),
        (5,  50, [49, 51, 45, 50, 45, 1], 1),
        (5,  50, [49, 51, 45, 50, 45, 1], [2,1,3,2,2]),
    ]

    for nsubjs, nnodes, ntimepoints, nruns in tests:
        with create_random_timeseries(1.2, nsubjs, nnodes,
                                      ntimepoints, nruns) as ts:
            fig = nets.plot_timeseries(ts, nodes=np.arange(nnodes))

            # <nnodes> spectra plots
            # <nnodes> thumbnail plots
            assert len(fig.get_axes()) == nnodes * 2
