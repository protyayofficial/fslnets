#!/usr/bin/env python
#
# __init__.py - The FSLNets package.
#
"""FSLNets is a set of simple scripts for carrying out basic network modelling
from (typically FMRI) timeseries data.

https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLNets
"""


from fsl.nets.boxplots  import (boxplots,
                                boxplot,
                                edgepic)
from fsl.nets.classify  import  classify
from fsl.nets.clean     import  clean
from fsl.nets.glm       import (glm,
                                plot_pvalues)
from fsl.nets.groupmean import (groupmean,
                                plot_groupmean)
from fsl.nets.hierarchy import (hierarchy,
                                plot_hierarchy)
from fsl.nets.load      import (load,
                                load_from_images,
                                load_file)
from fsl.nets.netmats   import  netmats
from fsl.nets.spectra   import  plot_spectra
from fsl.nets.web       import  web
