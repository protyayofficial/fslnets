#!/usr/bin/env python


import tempfile
import numpy as np

from fsl import nets

from . import generate_random_data


def check_timeseries(ts, tr, nsubjects, nnodes, ntimepoints, nruns):
    """Check that the basic TimeSeries properties are correct. """

    if np.isscalar(ntimepoints): ntimepoints = [ntimepoints] * nsubjects
    if np.isscalar(nruns):       nruns       = [nruns]       * nsubjects

    ntimepoints = np.array(ntimepoints)
    nruns       = np.array(nruns)

    for s in range(nsubjects):
        assert ts.ts[s].shape == (nruns[s], ntimepoints[s], nnodes)

    for s in range(nsubjects):
        off = sum(nruns[:s])
        assert list(ts.datasets(s)) == list(range(off, off + nruns[s]))

    assert ts.nnodes        == nnodes
    assert ts.nsubjects     == nsubjects
    assert ts.ndatasets     == nruns.sum()
    assert ts.tr            == tr
    assert ts.ntimepoints() == (nruns * ntimepoints).sum()
    assert np.all([ts.nruns(      s) for s in range(nsubjects)] == nruns)
    assert np.all([ts.ntimepoints(s) for s in range(nsubjects)] == ntimepoints)
    assert np.all([ts.node_index(n) == n for n in range(nnodes)])


def test_load_from_dr_directory():
    """Test loading from a dual regression directory."""

    tr          = 1.2
    nsubjects   = 10
    ntimepoints = 100
    nnodes      = 100

    with tempfile.TemporaryDirectory() as drdir:
        filenames, data = generate_random_data(
            nsubjects, nnodes, ntimepoints, 1, drdir, 'dr_stage1_{:02d}.txt')

        ts = nets.load(drdir, tr)
        check_timeseries(ts, tr, nsubjects, nnodes, ntimepoints, 1)


def test_load_from_files():
    """Test loading from a list of text files."""

    tr          = 1.2
    nsubjects   = 10
    ntimepoints = 100
    nnodes      = 100

    with tempfile.TemporaryDirectory() as td:
        filenames, data = generate_random_data(
            nsubjects, nnodes, ntimepoints, 1, td, '{:02d}.txt')

        ts = nets.load(filenames, tr)
        check_timeseries(ts, tr, nsubjects, nnodes, ntimepoints, 1)


def test_load_from_files_varying_timepoints():
    """Test loading data with a different number of timepoints per subject."""

    tr          = 1.2
    nsubjects   = 10
    ntimepoints = [100 + np.random.randint(-10, 11) for _ in range(nsubjects)]
    nnodes      = 100

    with tempfile.TemporaryDirectory() as td:
        filenames, data = generate_random_data(
            nsubjects, nnodes, ntimepoints, 1, td, '{:02d}.txt')
        ts = nets.load(filenames, tr)
        check_timeseries(ts, tr, nsubjects, nnodes, ntimepoints, 1)


def test_load_from_files_varying_timepoints_and_runs():
    """Test loading data with a different number of timepoints and runs per
    subject.
    """

    tr          = 1.2
    nsubjects   = 10
    nnodes      = 100
    nruns       = [np.random.randint(1, 4)            for _ in range(nsubjects)]
    ntimepoints = [(100 + np.random.randint(-10, 11)) for _ in range(nsubjects)]

    with tempfile.TemporaryDirectory() as td:

        filenames, data = generate_random_data(
            nsubjects, nnodes, ntimepoints, nruns, td, '{:02d}.txt')

        ts = nets.load(filenames, tr, nruns=nruns)
        check_timeseries(ts, tr, nsubjects, nnodes, ntimepoints, nruns)
