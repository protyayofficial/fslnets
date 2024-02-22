#!/usr/bin/env python


import contextlib
import tempfile
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

from fsl import nets


def generate_random_data(nsubjects, nnodes, ntimepoints, nruns, outdir, filepat):

    if np.isscalar(ntimepoints): ntimepoints = [ntimepoints] * nsubjects
    if np.isscalar(nruns):       nruns       = [nruns]       * nsubjects

    filenames = []
    datas     = []

    for subject in range(nsubjects):
        data     = np.random.random((ntimepoints[subject] * nruns[subject], nnodes))
        filename = op.join(outdir, filepat.format(subject))

        np.savetxt(filename, data)

        filenames.append(filename)
        datas    .append(data)

    return filenames, data


@contextlib.contextmanager
def create_random_timeseries(tr, nsubjects, nnodes, ntimepoints, nruns):
    with tempfile.TemporaryDirectory() as td:

        filenames, _ = generate_random_data(
            nsubjects, nnodes, ntimepoints, nruns, td, '{:02d}.txt')

        # generate dummy thumbnails
        for i in range(nnodes):
            filename = op.join(td, f'{i:03d}.png')
            fig      = plt.figure()
            ax       = fig.add_subplot(111)

            ax.text(0.5, 0.5, str(i), size=72, ha='center', va='center')
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(filename)

        yield nets.load(filenames, tr, nruns=nruns, thumbnaildir=td)
