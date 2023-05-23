#!/usr/bin/env python
#

import                    contextlib
import http.server     as http
import os.path         as op
import                    os
import multiprocessing as mp
import                    shutil
import                    sys
import                    time
import                    webbrowser


import numpy                   as     np
import matplotlib.image        as     mplimg
import scipy.cluster.hierarchy as     sch
from   fsl.utils.tempdir       import tempdir

from fsl.nets.hierarchy import hierarchy

index_template = """
<html>
<head>
<meta http-equiv="Content-Type" content="text/html;charset=utf-8">
<title>FSLNets interactive netmat visualisation</title>
</head>
<body>
<style type="text/css">body { background: white ; font-family: 'Arial'}</style>
<p><b>FSLNets interactive netmat visualisation</b></p>
<p>
  <div id="networkCtrl" style="display: inline-block; vertical-align: top"></div>
  <div id="fullNetwork" style="display: inline"></div>
</p>
<script data-main="js/main" src="js/lib/require.js"></script>
</body>
</html>
"""

js_template = """
require(["netjs", "lib/d3"], function(netjs, d3) {{

  // Implement your own network edge
  // thresholding algorithm here.
  function thresholdMatrix(matrix, args) {{

    var threshold   = args[0];
    var thresMatrix = [];

    for (var i = 0; i < matrix.length; i++) {{

      thresMatrix.push([]);

      for (var j = 0; j < matrix[i].length; j++) {{

        var val = Math.abs(matrix[i][j]);

        if (val < threshold) thresMatrix[i].push(Number.NaN);
        else                 thresMatrix[i].push(matrix[i][j]);
      }}
    }}

    return thresMatrix;
  }}


  // You need to populate two objects:
  //
  //    - The first one ('args' here) is passed to
  //      the loadNetwork function, and specifies
  //      data file locations, labels, and some
  //      initial values. See the loadNetwork
  //      function in netdata.js for detail on all
  //      arguments.

  //
  //    - The second one ('display' here) is passed
  //      to the displayNetwork function, and specifies
  //      display settings. See the displayNetwork
  //      function in netjs.js for details on all
  //      required and optional arguments.
  //
  var args             = {{}};
  var display          = {{}};

  args.matrices        = [{netmats}];
  args.matrixLabels    = [{labels}];
  args.nodeData        = [{clusters}];
  args.nodeDataLabels  = ["Cluster number"];
  args.nodeNames       = [{names}];
  args.nodeNameLabels  = ["Original node indices"];
  args.linkage         = {linkage};
  args.linkageOrder    = {linkageOrder};
  args.nodeOrders      = [{order}];
  args.nodeOrderLabels = ["Default order"];
  args.thumbnails      = [{thumbnails}];
  args.thresFunc       = thresholdMatrix;
  args.thresVals       = [{threshold}];
  args.thresLabels     = ["Thres perc"];
  args.thresholdIdx    = 0;
  args.nodeNameIdx     = 0;
  args.nodeOrderIdx    = -1; // use linkage by default
  args.numClusters     = {nclusts};

  // Figure out a sensible canvas size.
  var w  = window.innerWidth  - 200;
  var h  = window.innerHeight - 50;
  var sz = Math.min(w, h);

  display.networkDiv    = "#fullNetwork";
  display.controlDiv    = "#networkCtrl";
  display.networkWidth  = sz;
  display.networkHeight = sz;

  display.highlightOn   = true;

  // Load the network, and
  // display it when loaded.
  netjs.loadNetwork(args, function(net) {{
    netjs.displayNetwork(net, display);
  }});
}});
"""


def web(ts, netmats, labels, savedir=None, openpage=True, thumbthres=0.25, nclusts=6):
    """Open an interactive netmat viewer in a web browser.

    ts:         TimeSeries object

    netmats:    List of (nodes, nodes) netmat arrays

    labels:     List of labels for each netmat

    savedir:    Optional path to save HTML/Javascript files for later viewing.
                If not provided, files are saved to a temporary location.

    openpage:   Set to False if you don't want the web page to be opened
                immediately.

    thumbthres: Threshold in the range [0, 1] - pixels in thumbnail images with
                an intensity lower than this will be made transparent.
    """

    names     = [f'{n}'   for n in ts.nodes]
    labels    = [f'"{l}"' for l in labels]
    fnetmats  = [f'fnetmat{i}.txt' for i in range(len(netmats))]
    order     = np.arange(ts.nnodes)

    threshold = np.percentile(np.abs(netmats[0]), 75)
    linkage   = hierarchy(netmats[0])
    linkOrder = sch.dendrogram(linkage, no_plot=True)['leaves']
    clusters  = sch.fcluster(linkage, nclusts, 'maxclust')


    netjs = op.join(op.dirname(__file__), 'netjs', 'js')

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

    with tempdir(override=savedir):

        thumbnails = []
        os.makedirs('thumbnails', exist_ok=True)
        for i, n in enumerate(ts.nodes):
            fname = op.join('thumbnails', f'{i:04d}.png');
            adjust_thumbnail(ts.thumbnail(n), fname, thumbthres)
            thumbnails.append(fname)

        for i, netmat in enumerate(netmats):
            np.savetxt(fnetmats[i], netmat, fmt='%0.8f')

        # netjs expects linkage file to be
        # MATLAB style, i.e. one-indexed.
        np.savetxt('linkage.txt',      linkage[:, :3] + 1, fmt='%0.6f')
        np.savetxt('linkageOrder.txt', linkOrder,          fmt='%0.6f')
        np.savetxt('clusters.txt',     clusters,           fmt='%i')
        np.savetxt('order.txt',        order,              fmt='%i')

        with open('names.txt', 'wt') as f:
            f.write('\n'.join(names))

        context = {
            'netmats'      : ','.join([f'"{f}"' for f in fnetmats]),
            'labels'       : ','.join(labels),
            'clusters'     : '"clusters.txt"',
            'names'        : '"names.txt"',
            'linkage'      : '"linkage.txt"',
            'linkageOrder' : '"linkageOrder.txt"',
            'order'        : '"order.txt"',
            'threshold'    : threshold,
            'nclusts'      : nclusts,
            'thumbnails'   : ','.join([f'"{f}"' for f in thumbnails]),
        }

        shutil.copytree(netjs, 'js', dirs_exist_ok=True)
        with open('index.html', 'wt') as f:
            f.write(index_template)
        with open(op.join('js', 'main.js'), 'wt') as f:
            f.write(js_template.format(**context))

        if openpage:
            with HTTPServer.server() as srv:
                url = f'{srv.url}/index.html'
                webbrowser.open_new_tab(url)
                time.sleep(5)
                srv.shutdown.set()


def adjust_thumbnail(infile, outfile, threshold):
    """Adjusts an image, making all pixels with an intensity lower than
    the threshold transparent.
    """

    data = mplimg.imread(infile)

    if data.shape[2] != 4:
        newdata           = np.ones((data.shape[0], data.shape[1], 4),
                                    dtype=np.float32)
        newdata[:, :, :3] = data
        data              = newdata

    intensities = np.sum(data[..., :3], axis=2)
    xs, ys = np.where(intensities <= 0.25)

    data[xs, ys, 3] = 0

    mplimg.imsave(outfile, data)


@contextlib.contextmanager
def indir(dirname):
    """Context manager which temporarily changes into dir."""
    prevdir = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(prevdir)


class HTTPServer(mp.Process):
    """Simple HTTP server which serves files from a specified directory.

    Intended to be used via the server static method.
    """

    @contextlib.contextmanager
    @staticmethod
    def server(rootdir=None):
        """Start a HTTPServer on a separate thread to serve files from
        rootdir (defaults to the current working directory), then shut it down
        afterwards.
        """
        if rootdir is None:
            rootdir = os.getcwd()
        srv = HTTPServer(rootdir)
        srv.start()
        # wait until server has started
        srv.startup.wait()
        srv.url = 'http://localhost:{}'.format(srv.port)
        try:
            yield srv
        finally:
            srv.stop()

    def __init__(self, rootdir):
        mp.Process.__init__(self)
        self.daemon   = True
        self.rootdir  = rootdir
        self.portval  = mp.Value('i', -1)
        self.startup  = mp.Event()
        self.shutdown = mp.Event()

    def stop(self):
        self.shutdown.set()

    @property
    def port(self):
        return self.portval.value

    def run(self):
        # Suppress log messages
        handler             = http.SimpleHTTPRequestHandler
        handler.log_message = lambda *a: None
        server              = http.HTTPServer(('', 0), handler)

        # store port number, notify startup
        self.portval.value = server.server_address[1]
        self.startup.set()

        with indir(self.rootdir):
            while not self.shutdown.is_set():
                server.handle_request()
            server.shutdown()
