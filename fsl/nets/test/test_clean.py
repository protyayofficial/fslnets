#!/usr/bin/env python
#
# test_clean.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

import numpy as np

from fsl import nets

from fsl.nets.test import create_random_timeseries


def check_timeseries(ts, origts, goodnodes):
    cleants = [np.array(t) for t in ts.ts]

    assert len(cleants) == len(origts)
    assert len(cleants) == len(ts.origts)

    for ots, ots2, cts in zip(origts, ts.origts, cleants):
        assert np.all(np.isclose(cts, ots[ :, :, goodnodes]))
        assert np.all(np.isclose(cts, ots2[:, :, goodnodes]))


def test_clean():
    with create_random_timeseries(1, 10, 10, 100, 1) as ts:
        goodnodes = [0, 1, 2, 3, 4]
        origts    = [np.array(t) for t in ts.ts]
        nets.clean(ts, goodnodes)
        check_timeseries(ts, origts, goodnodes)


def test_clean_multiple_runs():

    with create_random_timeseries(1, 10, 10, 100, 3) as ts:
        goodnodes = [0, 1, 2, 3, 4]
        origts    = [np.array(t) for t in ts.ts]
        nets.clean(ts, goodnodes)
        check_timeseries(ts, origts, goodnodes)
