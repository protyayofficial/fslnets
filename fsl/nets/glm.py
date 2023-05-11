#!/usr/bin/env python
#
# glm.py - do cross-subject GLM on a set of netmats
#
# Author: Steve Smith and Ludo Griffanti - 2013-2014
# Author: Paul McCarthy, 2023
#

import os.path           as     op
import numpy             as     np
import fsl.data.vest     as     vest
from   fsl.data.image    import Image
from   fsl.utils.tempdir import tempdir
from   fsl.wrappers      import randomise


def glm(ts, netmats, design, contrasts, nperms=5000):

    nsubjs     = netmats.shape[0]
    nedges     = netmats.shape[1]
    ncontrasts = vest.loadVestFile(contrasts).shape[0]
    design     = op.abspath(design)
    contrasts  = op.abspath(contrasts)

    with tempdir():

        netmats = netmats.T.reshape((nedges, 1, 1, nsubjs))

        Image(netmats).save('netmats')

        randomise('netmats', 'output', d=design, t=contrasts, n=nperms,
                  x=True, uncorrp=True)

        puncorr = np.zeros((ncontrasts, nedges))
        pcorr   = np.zeros((ncontrasts, nedges))

        for con in range(ncontrasts):
            puncorr[con] = Image(f'output_vox_p_tstat{con+1}')    .data.flatten()
            pcorr[  con] = Image(f'output_vox_corrp_tstat{con+1}').data.flatten()

    return puncorr, pcorr
