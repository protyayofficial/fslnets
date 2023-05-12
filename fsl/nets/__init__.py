#!/usr/bin/env python
#
# __init__.py -
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

"""
*nets_load
*nets_tsclean
nets_nodepics
nets_spectra
*nets_netmats
*nets_groupmean
nets_hierarchy
nets_netweb
*nets_glm
nets_edgepics
nets_lda
nets_boxplots
"""

from fsl.nets.load      import load
from fsl.nets.clean     import clean
from fsl.nets.glm       import glm
from fsl.nets.netmats   import netmats
from fsl.nets.groupmean import groupmean
from fsl.nets.spectra   import plot_spectra
