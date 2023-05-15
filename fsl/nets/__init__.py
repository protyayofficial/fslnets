#!/usr/bin/env python
#
# __init__.py - The FSLNets package.
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#

"""
*nets_load
*nets_tsclean
nets_nodepics
*nets_spectra
*nets_netmats
*nets_groupmean
nets_hierarchy
nets_netweb
*nets_glm
nets_edgepics
nets_lda
nets_boxplots
"""


from fsl.nets.clean     import  clean
from fsl.nets.glm       import (glm,
                                plot_pvalues)
from fsl.nets.groupmean import (groupmean,
                                plot_groupmean)
from fsl.nets.hierarchy import (hierarchy,
                                plot_hierarchy)
from fsl.nets.glm       import  glm
from fsl.nets.load      import  load
from fsl.nets.netmats   import  netmats
from fsl.nets.spectra   import  plot_spectra
