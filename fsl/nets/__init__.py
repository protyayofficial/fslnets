#!/usr/bin/env python
#
# __init__.py - The FSLNets package.
#
"""FSLNets is a set of simple scripts for carrying out basic network modelling
from (typically FMRI) timeseries data.

https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLNets
"""


__version__ = '0.7.0'


from fsl.nets.boxplots  import  boxplots
from fsl.nets.clean     import  clean
from fsl.nets.edgepics  import  edgepics
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
