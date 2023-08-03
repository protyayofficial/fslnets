#!/usr/bin/env python
#
# web.py - Open an interactive visualisation of a netmat connectivity matrix
#          in a web browser. The visualisation is based on
#          https://github.com/pauldmccarthy/netjs
#


import                    contextlib
import functools       as ft
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
"""Template used for the web page index.html."""

js_template = """
require(["netjs", "lib/d3"], function(netjs, d3) {{

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

  netjs.loadNetwork(args, function(net) {{
    netjs.displayNetwork(net, display);
  }});
}});
"""
"""Template used for the netjs main.js file."""


BASE_EXPECTED_REQUESTS = 16
"""Base number of HTTP requests to expect when the netjs page is opened in
a web browser. The total number of requests to expect is this, plus the
number of netmats (correlation matrix files), plus the number of nodes
(thumbnails). The expected number of base files corresponds to the following
files:

 - index.html
 - js/lib/require.js
 - js/main.js
 - js/netjs.js
 - js/lib/d3.js
 - js/netvis.js
 - js/netdata.js
 - js/netctrl.js
 - js/netvis_dynamics.js
 - js/lib/mustache.js
 - linkage.txt
 - linkageOrder.txt
 - clusters.txt
 - names.txt
 - order.txt
 - js/netctrl.html

This number is obviously very tightly coupled to the netjs implementation, and
would need to be updated whenever netjs is updated.
"""


MAX_SERVER_RUNTIME = 15
"""Maximum number of seconds to keep the temporary HTTP server open while
waiting for the web browser to load the netjs page.
"""


def web(ts, netmats, labels, savedir=None, openpage=True, nclusts=6, thumbthres=0.25):
    """Open an interactive netmat viewer in a web browser.

    ts:         TimeSeries object

    netmats:    List of (nodes, nodes) netmat arrays

    labels:     List of labels for each netmat

    savedir:    Optional path to save HTML/Javascript files for later viewing.
                If not provided, files are saved to a temporary location.

    openpage:   Set to False if you don't want the web page to be opened
                immediately.

    nclusts:    Colour and separate nodes into this many clusters.

    thumbthres: Threshold in the range [0, 1] - pixels in thumbnail images with
                an intensity lower than this will be made transparent.
    """

    netjs    = op.join(op.dirname(__file__), 'netjs', 'js')
    names    = [f'{n}'   for n in ts.nodes]
    labels   = [f'"{l}"' for l in labels]
    fnetmats = [f'fnetmat{i}.txt' for i in range(len(netmats))]
    order    = np.arange(ts.nnodes)

    # generate hierarchy/linkage and cluster labels
    threshold = np.percentile(np.abs(netmats[0]), 75)
    linkage   = hierarchy(netmats[0])
    linkOrder = sch.dendrogram(linkage, no_plot=True)['leaves']
    clusters  = sch.fcluster(linkage, nclusts, 'maxclust')

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

            # If we are opening the netjs page in a web
            # browser, we start a HTTP server, open the
            # netjs page, and wait until either the server
            # has processed a suitable number of requests,
            # or MAX_SERVER_RUNTIME seconds have elapsed.
            elapsed = 0
            timeout = MAX_SERVER_RUNTIME
            expreqs = BASE_EXPECTED_REQUESTS + len(netmats) + ts.nnodes

            with HTTPServer.server() as srv:

                url = f'{srv.url}/index.html'
                webbrowser.open_new_tab(url)

                while all((srv.numrequests < expreqs,
                           elapsed < timeout,
                           srv.is_running())):
                    time.sleep(0.5)
                    elapsed += 0.5

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
        self.daemon         = True
        self.rootdir        = rootdir
        self.portval        = mp.Value('i', -1)
        self.startup        = mp.Event()
        self.shutdown       = mp.Event()
        self.requestcounter = mp.Value('i', 0)

    def stop(self):
        self.shutdown.set()

    def is_running(self):
        return not self.shutdown.is_set()

    @property
    def port(self):
        return self.portval.value

    @property
    def numrequests(self):
        return self.requestcounter.value

    def run(self):
        # Use a custom HTTPRequestHandler
        # class to count successful requests.
        handler = HTTPRequestHandler.factory(self.requestcounter)
        server  = http.HTTPServer(('', 0), handler)

        # store port number, notify startup
        self.portval.value = server.server_address[1]
        self.startup.set()

        # Configure the handle_request method
        # to wait for up to half a second for
        # a request before returning.
        server.timeout = 0.5

        # Serve until the stop() method is called.
        with indir(self.rootdir):
            try:
                while not self.shutdown.is_set():
                    server.handle_request()
                server.shutdown()
            # set the shutdown event on error
            finally:
                self.stop()


class HTTPRequestHandler(http.SimpleHTTPRequestHandler):
    """Custom HTTPRequestHandler which updates a shared multiprocessing.Value
    instance whenever a successful HTTP 200 request is processed.
    """

    @staticmethod
    def factory(requestcounter):
        return ft.partial(HTTPRequestHandler, requestcounter)

    def __init__(self, requestcounter, *args, **kwargs):
        self.requestcounter = requestcounter
        super().__init__(*args, **kwargs)

    def log_message(self, *args, **kwargs):
        """Suppress logging output. """

    def log_request(self, code='-', size='-'):
        """Increment the request counter on HTTP 200. """
        super().log_request(code, size)
        if code == 200:
            self.requestcounter.value += 1
