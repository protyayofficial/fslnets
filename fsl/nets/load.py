#!/usr/bin/env python
#
# load.py - load a folder full of individual runs'/subjects' node-timeseries
#           files
#
# Author: Steve Smith and Ludo Griffanti, 2013-2014
#         Paul McCarthy, 2023
#


import                        glob
import                        os
import os.path         as     op
import                        tempfile
from   pathlib         import Path
from   collections.abc import Sequence



import numpy             as     np
from   fsl.utils.tempdir import tempdir
from   fsl.utils.run     import runfsl
from   fsl.data.image    import Image
from   fsl.nets          import dualreg


def load(infiles,
         tr,
         varnorm=0,
         demean=True,
         nruns=1,
         spatialmaps=None,
         thumbnaildir=None,
         bgimage=None):
    """Load a folder full of individual runs'/subjects' node-timeseries files.

    Returns a TimeSeries object through which the time series and other
    metadata can be retrieved.

    infiles:      List of files to load, or path to a dual regression
                  directory.

    tr:           Temporal resolution in seconds (assumed to be the same for
                  all subjects/runs)

    varnorm:      Whether to perform temporal variance normalisation:

                   - 0: No variance normalisation (default)

                   - 1: Normalise overall stddev for each run (normally one
                        run/subject per timeseries file)

                   - 2: Normalise separately each separate timeseries from
                        each run

    demean:       Whether or not to demean the time series for each subject
                  and run

    nruns:        Number of runs per subject, if per-subject time courses from
                  separate runs have been temporally concatenated into a
                  single file. May either be an int, where all subjects have
                  the same number of runs, or a list specifying the number of
                  runs per subject. All of the runs for one subject must have
                  the same number of time points.

    spatialmaps:  4D image which contains spatial maps representing the nodes.
                  Used to generate node thumbnails if thumbnaildir is not
                  provided.

    thumbnaildir: Path to a directory containing PNG thumbnails for each
                  node. The PNG files must be named so that, when sorted,
                  their order matches the node order in the input data files.

    bgimage:      3D image which is used as the background in generated
                  thumbnail images. Ignored if thumbnaildir is provided.
    """

    if isinstance(infiles, (str, Path)):
        infiles = sorted(glob.glob(op.join(infiles, '*.txt')))

    nsubjects = len(infiles)

    if not isinstance(nruns, Sequence):
        nruns = [nruns] * nsubjects

    timeseries = []

    for infile, nr in zip(infiles, nruns):
        timeseries.append(load_file(infile, varnorm, demean, nr))

    ts0    = timeseries[0]
    nnodes = ts0.shape[2]

    for i, ts in enumerate(timeseries):
        if ts.shape[2] != nnodes:
            raise ValueError('All input files must have the same number of '
                             f'nodes ({infiles[i]} has {ts.shape[2]} nodes, '
                             f'but ({infiles[0]} has {ts0.shape[2]} nodes)')

    thumbs = load_thumbnails(thumbnaildir, spatialmaps, bgimage)

    return TimeSeries(timeseries, tr, thumbs)


def load_file(infile, varnorm=0, demean=True, nruns=1):
    """Loads a single text file containing (timepoints) rows and (nodes)
    columns.

    If the text file contains multiple temporally concatenated runs (as
    denoted by nruns), the data for each run is split into separate arrays.
    All runs are assumed to have the same number of time points.

    Returns a (nruns, timepoints, nodes) array.
    """

    ts          = np.loadtxt(infile, dtype=np.float64)
    ntimepoints = ts.shape[0]
    nnodes      = ts.shape[1]
    runlen      = ntimepoints // nruns

    if ntimepoints % nruns != 0:
        raise ValueError(f'[{infile}: ntimepoints ({ntimepoints}) '
                         f'must be a multiple of nruns ({nruns})!')

    ts = ts.reshape(nruns, runlen, nnodes)

    # demean time series from each run separately
    if demean:
        for run in range(nruns):
            ts[run] = ts[run] - ts[run].mean(axis=0)

    # normalise across all runs
    if varnorm == 1:
        ts = ts / ts.std(axis=(0, 1))

    # normalise time series from each run separately
    elif varnorm == 2:
        for run in range(nruns):
            ts[run] = ts[run] / ts[run].std(axis=0)

    return ts


def load_from_images(spatialmaps, subjfiles, *args, **kwargs):
    """Use dual regression stage 1 to generate the subject-specific time series
    for each component/node contained in the spatial map file.
    """

    spatialmaps = op.abspath(spatialmaps)
    subjfiles   = [op.abspath(f) for f in subjfiles]
    timeseries  = []

    with tempdir():
        dualreg.create_common_mask(subjfiles, 'mask')

        for i, subjfile in enumerate(subjfiles):
            outfile = f'{i}.txt'
            dualreg.stage1(spatialmaps, subjfile, 'mask', outfile)
            timeseries.append(outfile)

        return load(timeseries, *args, spatialmaps=spatialmaps, **kwargs)


def load_thumbnails(thumbnaildir, spatialmaps, bgimage=None):
    """Loads (and generates if necessary) per-node thumbnail images.
    Returns a list of file paths to each thumbnail.

    thumbnaildir: Directory containing a thumbnail image for each node.
                  Must be able to be sorted into the node order.

    spatialmaps:  4D image which contains spatial maps representing the nodes.
                  Ignored if a thumbnaildir is provided.  Otherwise,
                  thumbnails are generated from this file.

    bgimage:      Passed through to generate_thumbnails. Ignored if a
                  thumbnaildir is provided.
    """

    if thumbnaildir is not None:
        thumbnaildir = op.abspath(thumbnaildir)
    if spatialmaps is not None:
        spatialmaps = op.abspath(spatialmaps)

    if thumbnaildir is None and spatialmaps is None:
        return None
    if thumbnaildir is None:
        thumbnaildir = generate_thumbnails(spatialmaps, bgimage)

    return sorted(glob.glob(op.join(thumbnaildir, '*.png')))


def generate_thumbnails(spatialmaps, bgimage=None):
    """Generate thumbnails from a 4D image.

    spatialmaps: 4D image which contains spatial maps representing the nodes.
    bgimage:     Passed through to generate_thumbnails.
    """

    # We try to save the thumbnails alongside the
    # file, in a directory called .{filename}.thumbnails
    filename     = op.basename(spatialmaps)
    dirname      = op.dirname(spatialmaps)
    thumbnaildir = op.join(dirname, f'.{filename}.thumbnails')
    fsldir       = os.environ['FSLDIR']

    if op.exists(thumbnaildir):
        return thumbnaildir

    try:
        os.makedirs(thumbnaildir, exist_ok=True)
    except Exception:
        thumbnaildir = tempfile.mkdtemp(prefix=op.basename(thumbnaildir))

    if bgimage is None:
        # assuming MNI152
        shape = Image(spatialmaps).shape[:3]

        if   shape == (91,  109, 91):  std = '2mm'
        elif shape == (182, 218, 182): std = '1mm'
        else: raise RuntimeError('Don\'t know what standard template '
                                 f'to use for melodic_IC ({shape})')

        bgimage = op.join(fsldir, 'data', 'standard', f'MNI152_T1_{std}')

    runfsl(f'slices_summary {spatialmaps} 4 {bgimage} {thumbnaildir} -1')

    return thumbnaildir


class TimeSeries:
    """Class which contains a reference to per-subject and node time series
    data, and associated metadata. Created by the load function.
    """

    def __init__(self, ts, tr, thumbnails):
        """Create a TimeSeries object. Don't create a TimeSeries directly - use
        the load function.

        ts:         List of (nruns, ntimepoints, nnodes) arrays, one per
                    subject.

        tr:         Temporal resolution (assumed to be the same for every
                    subject/run)

        thumbnails: List of thumbnail file paths, one per node.
        """

        nnodes             = ts[0].shape[2]
        self.__ts          = list(ts)
        self.__origts      = list(ts)
        self.__tr          = tr
        self.__orignnodes  = nnodes
        self.__goodmask    = np.ones( nnodes, dtype=bool)
        self.__unknownmask = np.zeros(nnodes, dtype=bool)
        self.__thumbnails  = thumbnails

    @property
    def nodes(self):
        """Return node labels. These are the node indices into the original
        time series. If nets.clean has been run, these will *not be* indices
        into the current time series, as bad nodes will have been removed/
        regressed.

        To get the actual index for a given node, use the node_index method.
        """
        return np.where(self.__goodmask)[0]

    @property
    def nnodes(self):
        """Number of nodes. """
        return self.__goodmask.sum()

    @property
    def orignnodes(self):
        """Original number of nodes, before nets.clean was called. """
        return self.__orignnodes

    def node_index(self, node):
        """Return the index into ts for the given node. """
        return np.where(self.nodes == node)[0][0]

    @property
    def tr(self):
        """Return the temporal resolution (TR) of all data sets. """
        return self.__tr

    @property
    def nsubjects(self):
        """Number of subjects represented in ts. """
        return len(self.__ts)

    @property
    def ndatasets(self):
        """Total number of data sets (subjects * runs) contained in ts. """
        return sum([t.shape[0] for t in self.ts])

    def datasets(self, subj):
        """Returns indices for all datasets/runs for the given subject.
        These indices are useful when working with netmats, which are
        arranged as (nsubjects*nruns, edges) arrays.
        """
        offset = sum(self.nruns(s) for s in range(subj))
        return range(offset, offset + self.nruns(subj))

    def nruns(self, subjidx):
        """Number of runs in ts for the given subject. """
        return self.__ts[subjidx].shape[0]

    def ntimepoints(self, subj=None):
        """Number of time points in ts.

        Without any arguments, returns the total number of timepoints across
        all subjects/runs.

        With a subject index, returns the number of time points for the given
        subject (for a single run).
        """
        if subj is None:
            return sum([t.shape[0] * t.shape[1] for t in self.__ts])
        else:
            return self.__ts[subj].shape[1]

    @property
    def ts(self):
        """Returns the time series data - a list of (nruns, ntimepoints, nnodes)
        arrays, one for each subject. If nets.clean has been called, bad nodes
        will have been removed/regressed from the time series.
        """
        return self.__ts

    @property
    def origts(self):
        """Returns the original time series data, prior to manipulation by
        nets.clean.
        """
        return list(self.__origts)

    @property
    def allts(self):
        """Iterate over all data sets (subjects * runs) contained in ts.
        On each iteration, returns:
          - dataset index
          - subject index
          - run index
          - the (timepoints, nodes) array
        """
        i = 0
        for subj in range(self.nsubjects):
            for run in range(self.nruns(subj)):
                yield i, subj, run, self.__ts[subj][run]
                i = i + 1

    def thumbnail(self, nodeidx):
        """Return the thumbnail file path for the given node. """
        if self.__thumbnails is None:
            return None
        return self.__thumbnails[nodeidx]

    @ts.setter
    def ts(self, newts):
        """Replace the time series. Used by nets.clean. """
        self.__ts = newts

    @property
    def goodnodes(self):
        """Return a boolean mask denoting all of the "good" nodes. """
        return self.__goodmask

    @property
    def unknownnodes(self):
        """Return a boolean mask denoting all of the "unknown" nodes (neither
        bad nor good.

        These nodes will be removed from the data set by the nets.clean
        function, but their time courses will not be regressed from
        the good nodes if aggressive cleaning is used.
        """
        return self.__unknownmask

    @property
    def badnodes(self):
        """Return a boolean mask denoting all of the "bad" nodes. """
        return ~(self.goodnodes | self.unknownnodes)

    @goodnodes.setter
    def goodnodes(self, nodes):
        """Specify nodes to keep. Used by nets.clean. """
        nodes = np.asanyarray(nodes)

        if ((nodes < 0) | (nodes >= self.orignnodes)).any():
            raise ValueError(f'Invalid node indices (< 0 or > {self.nnodes})')

        mask            = np.zeros(self.orignnodes, dtype=bool)
        mask[nodes]     = True
        self.__goodmask = mask

    @unknownnodes.setter
    def unknownnodes(self, nodes):
        """Specify nodes that should not be kept, but should not be regressed.
        Used by nets.clean.
        """
        nodes = np.asanyarray(nodes)

        if ((nodes < 0) | (nodes >= self.orignnodes)).any():
            raise ValueError(f'Invalid node indices (< 0 or > {self.nnodes})')

        mask               = np.zeros(self.orignnodes, dtype=bool)
        mask[nodes]        = True
        self.__unknownmask = mask
