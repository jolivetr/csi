'''
A class that deals with downsampling the insar data.

Authors: R. Jolivet, January 2014.
         R. Grandin, April 2015
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import matplotlib.path as path
import matplotlib as mpl
import copy
import sys
import os
import multiprocessing as mp

# Personals
from .insar import insar
from .opticorr import opticorr
from .imagecovariance import imagecovariance as imcov
from .csiutils import _split_seq

# Initialize a class for multiprocessing gradient computation
class mpgradcurv(mp.Process):

    def __init__(self, downsampler, Bsize, indexes, queue):
        '''
        Initializes a multiprocessing class

        Args:
            * downsampler   : instance of imagedownsampling
            * Bsize         : Size of blocks
            * indexes       : indexes of blocks
            * queue         : instance of mp.Queue
        '''

        # Save
        self.downsampler = downsampler
        self.Bsize = Bsize
        self.indexes = indexes
        self.queue = queue

        # Initialize the process
        super(mpgradcurv, self).__init__()

        # All done
        return

    def run(self):
        '''
        Runs the gradient/curvature computation
        '''

        gradient = []
        curvature = []

        # Over each block, we average the position and the phase to have a new point
        for i in self.indexes:

            # If Bsize, then set to 0.
            if self.Bsize[i]:

                gradient.append(0.0)
                curvature.append(0.0)

            else:

                # Get block
                block = self.downsampler.blocks[i]

                # Split it in 4 blocks
                subBlocks = self.downsampler.cutblockinfour(block)

                # Create list
                xg = []; yg = []; means = []

                for subblock in subBlocks:
                    # Create a path
                    p = path.Path(subblock, closed=False)
                    # Find those who are inside
                    ii = p.contains_points(self.downsampler.PIXXY)
                    # Check if total area is sufficient
                    check = self.downsampler._isItAGoodBlock(block,
                            np.flatnonzero(ii).shape[0])
                    if check:
                        if self.downsampler.datatype is 'insar':
                            vel = np.mean(self.downsampler.image.vel[ii])
                            means.append(vel)
                        elif self.datatype is 'opticorr':
                            east = np.mean(self.downsampler.image.east[ii])
                            north = np.mean(self.downsampler.image.north[ii])
                            means.append(np.sqrt(east**2+north**2))
                        xg.append(np.mean(self.downsampler.image.x[ii]))
                        yg.append(np.mean(self.downsampler.image.y[ii]))
                means = np.array(means)

                # estimate gradient
                if len(xg)>=2:
                    A = np.zeros((len(xg),3))
                    A[:,0] = 1.
                    A[:,1] = xg
                    A[:,2] = yg
                    cffs = np.linalg.lstsq(A,means,rcond=None)
                    gradient.append(np.abs(np.mean(cffs[0][1:])))
                    curvature.append(np.std(means - np.dot(A,cffs[0])))
                else:
                    gradient.append(0.)
                    curvature.append(0.)

        # Save gradient
        self.queue.put([gradient, curvature, self.indexes])

        # All done
        return

# Initialize a class for multiprocessing downsampling
class mpdownsampler(mp.Process):

    def __init__(self, downsampler, blocks, blocksll, queue):
        '''
        Initialize the multiprocessing class.

        Args:
            * downsampler   : instance of imagedownsampling
            * blocks        : list of blocks
            * blocksll      : list of blocks
            * queue         : Instance of mp.Queue

        Kwargs:
            * datatype  : 'insar' or 'opticorr'
        '''

        # Save
        self.downsampler = downsampler
        self.blocks = blocks
        self.blocksll = blocksll
        self.queue = queue

        # Initialize the process
        super(mpdownsampler, self).__init__()

        # All done
        return

    def run(self):
        '''
        Run the phase averaging.
        '''

        # Initialize lists
        X, Y, Lon, Lat, Wgt = [], [], [], [], []
        if self.downsampler.datatype is 'insar':
            Vel, Err, Los = [], [], []
        elif self.downsampler.datatype is 'opticorr':
            East, North, Err_east, Err_north = [], [], [], []
        outBlocks = []
        outBlocksll = []

        # Over each block, we average the position and the phase to have a new point
        for block, blockll in zip(self.blocks, self.blocksll):

            # Create a path
            p = path.Path(block, closed=False)

            # Find those who are inside
            ii = p.contains_points(self.downsampler.PIXXY)

            # Check if total area is sufficient
            check = self.downsampler._isItAGoodBlock(block, np.flatnonzero(ii).shape[0])

            # If yes
            if check:

                # Save block
                outBlocks.append(block)
                outBlocksll.append(blockll)

                # Get Mean, Std, x, y, ...
                wgt = len(np.flatnonzero(ii))
                if self.downsampler.datatype is 'insar':
                    vel = np.mean(self.downsampler.image.vel[ii])
                    err = np.std(self.downsampler.image.vel[ii])
                    los0 = np.mean(self.downsampler.image.los[ii,0])
                    los1 = np.mean(self.downsampler.image.los[ii,1])
                    los2 = np.mean(self.downsampler.image.los[ii,2])
                    norm = np.sqrt(los0*los0+los1*los1+los2*los2)
                    los0 /= norm
                    los1 /= norm
                    los2 /= norm
                elif self.downsampler.datatype is 'opticorr':
                    east = np.mean(self.downsampler.image.east[ii])
                    north = np.mean(self.downsampler.image.north[ii])
                    err_east = np.std(self.downsampler.image.east[ii])
                    err_north = np.std(self.downsampler.image.north[ii])
                x = np.mean(self.downsampler.image.x[ii])
                y = np.mean(self.downsampler.image.y[ii])
                lon, lat = self.downsampler.xy2ll(x, y)

                # Store that
                if self.downsampler.datatype is 'insar':
                    Vel.append(vel)
                    Err.append(err)
                    Los.append([los0, los1, los2])
                elif self.downsampler.datatype is 'opticorr':
                    East.append(east)
                    North.append(north)
                    Err_east.append(err_east)
                    Err_north.append(err_north)
                X.append(x)
                Y.append(y)
                Lon.append(lon)
                Lat.append(lat)
                Wgt.append(wgt)

        # Save
        if self.downsampler.datatype is 'insar':
            self.queue.put([X, Y, Lon, Lat, Wgt, Vel, Err, Los, outBlocks, outBlocksll])
        elif self.downsampler.datatype is 'opticorr':
            self.queue.put([X, Y, Lon, Lat, Wgt, East, North, Err_east, Err_north, 
                        outBlocks, outBlocksll])

        # All done
        return

class imagedownsampling(object):
    '''
    A class to downsample images

    Args:
        * name      : Name of the downsampler.
        * image     : InSAR or opticorr data set to be downsampled.

    Kwargs:
        * faults    : List of faults.
        * verbose   : Talk to me

    Returns:
        * None
    '''

    def __init__(self, name, image, faults=None, verbose=True):

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize InSAR downsampling tools {}".format(name))

        self.verbose = verbose

        # Set the name
        self.name = name
        self.datatype = image.dtype

        # Set the transformation
        self.utmzone = image.utmzone
        self.lon0 = image.lon0
        self.lat0 = image.lat0
        self.putm = image.putm
        self.ll2xy = image.ll2xy
        self.xy2ll = image.xy2ll

        # Check if the faults are in the same utm zone
        self.faults = []
        if faults is not None:
            if type(faults) is not list:
                faults = [faults]
            for fault in faults:
                assert (fault.utmzone==self.utmzone), 'Fault {} not in utm zone #{}'.format(fault.name, self.utmzone)
                assert (fault.lon0==self.lon0), 'Fault {} does not have same origin Lon {}'.format(fault.name, self.lon0)
                assert (fault.lat0==self.lat0), 'Fault {} does not have same origin Lat {}'.format(fault.name, self.lat0)
                self.faults.append(fault)

        # Save the image
        self.image = image

        # Incidence and heading need to be defined if already defined
        if self.datatype is 'insar':
            if hasattr(self.image, 'heading'):
                self.heading = self.image.heading
            if hasattr(self.image, 'incidence'):
                self.incidence = self.image.incidence

        # Create the initial box
        xmin = np.floor(image.x.min())
        xmax = np.floor(image.x.max())+1.
        ymin = np.floor(image.y.min())
        ymax = np.floor(image.y.max())+1.
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.box = [[xmin, ymin],
                    [xmin, ymax],
                    [xmax, ymax],
                    [xmax, ymin]]
        lonmin = image.lon.min()
        lonmax = image.lon.max()
        latmin = image.lat.min()
        latmax = image.lat.max()
        self.lonmin = lonmin; self.latmax = latmax
        self.latmin = latmin; self.lonmax = lonmax
        self.boxll = [[lonmin, latmin],
                      [lonmin, latmax],
                      [lonmax, latmax],
                      [lonmax, latmin]]

        # Get the original pixel spacing
        self.spacing = distance.cdist([[image.x[0], image.y[0]]], [[image.x[i], image.y[i]] for i in range(1, image.x.shape[0])])[0]
        self.spacing = self.spacing.min()

        if self.verbose:
            print('Effective pixel spacing: {}'.format(self.spacing))

        # Deduce the original pixel area
        self.pixelArea = self.spacing**2

        # All done
        return

    def initialstate(self, startingsize, minimumsize, tolerance=0.5, plot=False, decimorig=10):
        '''
        Does the first cut onto the data.

        Args:
            * startingsize  : Size of the first regular downsampling (it'll be the effective maximum size of windows)
            * minimumsize   : Minimum Size of the blocks.

        Kwargs:
            * tolerance     : Between 0 and 1. If 1, all the pixels must have a value so that the box is kept. If 0, no pixels are needed... Default is 0.5
            * decimorig     : Decimation ofr plotting purposes only.
            * plot          : True/False

        Returns:
            * None
        '''

        # Set the tolerance
        self.tolerance = tolerance
        self.minsize = minimumsize

        # Define Edges
        xLeftEdges = np.arange(self.xmin-startingsize, self.xmax+startingsize, startingsize)[:-1].tolist()
        yUpEdges = np.arange(self.ymin-startingsize, self.ymax+startingsize, startingsize)[1:].tolist()

        # Make blocks
        blocks = []
        for x in xLeftEdges:
            for y in yUpEdges:
                block = [ [x, y],
                          [x+startingsize, y],
                          [x+startingsize, y-startingsize],
                          [x, y-startingsize] ]
                blocks.append(block)
        # Set blocks
        self.setBlocks(blocks)

        # Generate the sampling to test
        self.downsample(plot=plot, decimorig=decimorig)

        # All done
        return

    def setBlocks(self, blocks):
        '''
        Takes a list of blocks and set it in self.

        Args:
            * blocks    : List of blocks (xy coordinates)

        Returns:
            * None
        '''

        # Save the blocks
        self.blocks = blocks

        # Build the list of blocks in lon, lat
        blocksll = []
        for block in blocks:
            c1, c2, c3, c4 = block
            blockll = [ self.xy2ll(c1[0], c1[1]),
                        self.xy2ll(c2[0], c2[1]),
                        self.xy2ll(c3[0], c3[1]),
                        self.xy2ll(c4[0], c4[1]) ]
            blocksll.append(blockll)
        self.blocksll = blocksll

        # All done
        return

    def downsample(self, plot=False, decimorig=10,norm=None):
        '''
        From the saved list of blocks, computes the downsampled data set and the informations that come along.

        Kwargs:
            * plot      : True/False
            * decimorig : decimate a bit for plotting
            * norm      : colorlimits for plotting

        Returns:
            * None
        '''

        # Create the new image object
        if self.datatype is 'insar':
            newimage = insar('Downsampled {}'.format(self.image.name), utmzone=self.utmzone, verbose=False,
                             lon0=self.lon0, lat0=self.lat0)
        elif self.datatype is 'opticorr':
            newimage = opticorr('Downsampled {}'.format(self.image.name), utmzone=self.utmzone, verbose=False,
                                lon0=self.lon0, lat0=self.lat0)

        # Get the blocks
        blocks = self.blocks
        blocksll = self.blocksll

        # Create the variables
        if self.datatype is 'insar':
            newimage.vel = []
            newimage.err = []
            newimage.los = []
        elif self.datatype is 'opticorr':
            newimage.east = []
            newimage.north = []
            newimage.err_east = []
            newimage.err_north = []
        newimage.lon = []
        newimage.lat = []
        newimage.x = []
        newimage.y = []
        newimage.wgt = []

        # Store the factor
        newimage.factor = self.image.factor

        # Build the previous geometry
        self.PIXXY = np.vstack((self.image.x, self.image.y)).T
        # Create a queue to hold the results
        output = mp.Queue()

        # Check how many workers
        try:
            nworkers = int(os.environ['OMP_NUM_THREADS'])
        except:
            nworkers = mp.cpu_count()

        # Create the workers
        seqblocks = _split_seq(blocks, nworkers)
        seqblocksll = _split_seq(blocksll, nworkers)
        workers = [mpdownsampler(self, seqblocks[i], seqblocksll[i], output)\
                                 for i in range(nworkers)]

        # Start
        for w in range(nworkers): workers[w].start()

        # Initialize blocks
        blocks, blocksll = [], []

        # Collect
        for w in range(nworkers):
            if self.datatype is 'insar':
                x, y, lon, lat, wgt, vel, err, los, block, blockll  = output.get()
                newimage.vel.extend(vel)
                newimage.err.extend(err)
                newimage.los.extend(los)
            elif self.datatype is 'opticorr':
                x, y, lon, lat, wgt, east, north, err_east, err_north, block, blockll = output.get()
                newimage.east.extend(east)
                newimage.north.extend(north)
                newimage.err_east.extend(err_east)
                newimage.err_north.extend(err_north)
            newimage.x.extend(x)
            newimage.y.extend(y)
            newimage.lat.extend(lat)
            newimage.lon.extend(lon)
            newimage.wgt.extend(wgt)
            blocks.extend(block)
            blocksll.extend(blockll)

        # Save blocks
        self.blocks = blocks
        self.blocksll = blocksll

        # Convert
        if self.datatype is 'insar':
            newimage.vel = np.array(newimage.vel)
            newimage.err = np.array(newimage.err)
            newimage.los = np.array(newimage.los)
        elif self.datatype is 'opticorr':
            newimage.east = np.array(newimage.east)
            newimage.north = np.array(newimage.north)
            newimage.err_east = np.array(newimage.err_east)
            newimage.err_north = np.array(newimage.err_north)
        newimage.x = np.array(newimage.x)
        newimage.y = np.array(newimage.y)
        newimage.lon = np.array(newimage.lon)
        newimage.lat = np.array(newimage.lat)
        newimage.wgt = np.array(newimage.wgt)

        # Store newimage
        self.newimage = newimage

        # plot y/n
        if plot:
            self.plotDownsampled(decimorig=decimorig,norm=norm)

        # All done
        return

    def downsampleFromSampler(self, sampler, plot=False, decimorig=10):
        '''
        From the downsampling scheme in a previous sampler, downsamples the image.

        Args:
            * sampler       : Sampler which has a blocks instance.

        Kwargs:
            * plot          : Plot the downsampled data (True/False)
            * decimorig     : Stupid decimation factor for lighter plotting.

        Returns:
            * None
        '''

        # set the downsampling scheme
        self.setDownsamplingScheme(sampler)

        # Downsample
        self.downsample(plot=plot, decimorig=decimorig)

        # All done
        return

    def downsampleFromRspFile(self, prefix, tolerance=0.5, plot=False, decimorig=10):
        '''
        From the downsampling scheme saved in a .rsp file, downsamples the image.

        Args:
            * prefix        : Prefix of the rsp file.

        Kwargs:
            * tolerance     : Minimum surface covered in a patch to be kept.
            * plot          : Plot the downsampled data (True/False)
            * decimorig     : Simple decimation factor of the data for lighter plotting.

        Returns:    
            * None
        '''

        # Set tolerance
        self.tolerance = tolerance

        # Read the file
        self.readDownsamplingScheme(prefix)

        # Downsample
        self.downsample(plot=plot, decimorig=decimorig)

        # All done
        return

    def getblockcenter(self, block):
        '''
        Returns the center of a block.

        Args:
            * block         : Block as defined in initialstate.

        Returns:
            * None
        '''

        # Get the four corners
        c1, c2, c3, c4 = block
        x1, y1 = c1
        x2, y2 = c2
        x4, y4 = c4

        xc = x1 + (x2 - x1)/2.
        yc = y1 + (y4 - y1)/2.

        # All done
        return xc, yc

    def cutblockinfour(self, block):
        '''
        From a block, returns 4 equal blocks.

        Args:
            * block         : block as defined in initialstate.

        Returns:
            * 4 lists of block corners
        '''

        # Get the four corners
        c1, c2, c3, c4 = block
        x1, y1 = c1
        x2, y2 = c2
        x3, y3 = c3
        x4, y4 = c4

        # Compute the position of the center
        xc, yc = self.getblockcenter(block)

        # Form the 4 blocks
        b1 = [ [x1, y1],
               [xc, y1],
               [xc, yc],
               [x1, yc] ]
        b2 = [ [xc, y2],
               [x2, y2],
               [x2, yc],
               [xc, yc] ]
        b3 = [ [x4, yc],
               [xc, yc],
               [xc, y4],
               [x4, y4] ]
        b4 = [ [xc, yc],
               [x3, yc],
               [x3, y3],
               [xc, y3] ]

        # all done
        return b1, b2, b3, b4

    def cutblockinthree(self, block):
        '''
        Used to create a smoother downsampled grid. From a single block, returns three blocks. Not used for now.
        T.L. Shreve, January 2018

        Args:
            * block         : block as defined in initialstate.

        Returns:
            * 3 lists of block corners
        '''

        # Get the four corners
        cs1, cs2, cs3, cs4 = block
        xs1, ys1 = cs1
        xs2, ys2 = cs2
        xs3, ys3 = cs3
        xs4, ys4 = cs4

        # Compute the position of the center
        xsc, ysc = self.getblockcenter(block)
        #Where is the large block touching the smaller blocks? [top/bottom/left/right]
        touch = top
        # Form the 3 blocks (if the block is touched by smaller blocks beneath it)
        if touch is 'bottom':
            bs1 = [ [xs1, ys1],
                   [xs2, ys2],
                   [xs2, ysc],
                   [xs1, ysc] ]
            bs2 = [ [xs4, ysc],
                   [xsc, ysc],
                   [xsc, ys4],
                   [xs4, ys4] ]
            bs3 = [ [xsc, ysc],
                   [xs3, ysc],
                   [xs3, ys3],
                   [xsc, ys3] ]
        # Form the 3 blocks (if the block is touched by smaller blocks above it)
        elif touch is 'top':
            bs1 = [ [xs1, ys1],
                   [xsc, ys1],
                   [xsc, ysc],
                   [xs1, ysc] ]
            bs2 = [ [xsc, ys2],
                   [xs2, ys2],
                   [xs2, ysc],
                   [xsc, ysc] ]
            bs3 = [ [xs3, ysc],
                   [xs4, ysc],
                   [xs4, ys4],
                   [xs3, ys3] ]
       # Form the 3 blocks (if the block is touched by smaller blocks to the left)
        elif touch is 'left':
            bs1 = [ [xs1, ys1],
                   [xsc, ys1],
                   [xsc, ysc],
                   [xs1, ysc] ]
            bs2 = [ [xsc, ys2],
                   [xs4, ys2],
                   [xsc, ys4],
                   [xsc, ys2] ]
            bs3 = [ [xsc, ysc],
                   [xs3, ysc],
                   [xs3, ys3],
                   [xsc, ys3] ]
       # Form the 3 blocks (if the block is touched by smaller blocks to the right)
        elif touch is 'right':
            bs1 = [ [xs1, ys1],
                   [xsc, ys1],
                   [xsc, ys3],
                   [xs3, ys3] ]
            bs2 = [ [xsc, ys2],
                   [xs2, ys2],
                   [xs2, ysc],
                   [xsc, ysc] ]
            bs3 = [ [xs4, ysc],
                   [xsc, ysc],
                   [xsc, ys4],
                   [xs4, ys4] ]



        # all done
        return bs1, bs2, bs3



    def distanceBased(self, chardist=15, expodist=1, plot=False, decimorig=10,norm=None):
        '''
        Downsamples the dataset depending on the distance from the fault R.Grandin, April 2015

        Kwargs:
            * chardist      : Characteristic distance of downsampling.
            * expodist      : Exponent of the distance-based downsampling criterion.
            * plot          : True/False
            * decimorig     : decimate for plotting
            * Norm          : colorlimits for plotting

        Returns:
            * None
        '''

        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Distance-based downsampling ")

        # by default, try to do at least one pass
        do_downsamp=True

        # Iteration counter #
        it=0

        # Loops until done
        while do_downsamp:

            # If some block has to be downsampled, "do_resamp" will be set back to "True"
            do_downsamp=False

            # Check if block size is minimum
            Bsize = self._is_minimum_size(self.blocks)

            # Iteration #
            it += 1
            if self.verbose:
                print('Iteration {}: Testing {} data samples '.format(it, len(self.blocks)))

            # New list of blocks
            newblocks = []
            # Iterate over blocks
            for j in range(len(self.blocks)):
                block = self.blocks[j]
                # downsample if the block is too large, given its distance to the fault
                # ( except if the block already has minimum size )
                if ((self.distToFault(block)-chardist)<self.blockSize(block) ** expodist) and not Bsize[j]:
                    b1, b2, b3, b4 = self.cutblockinfour(block)
                    newblocks.append(b1)
                    newblocks.append(b2)
                    newblocks.append(b3)
                    newblocks.append(b4)
                    do_downsamp=True
                # otherwise, leave the block unchanged
                else:
                    newblocks.append(block)

            # Set the blocks
            self.setBlocks(newblocks)
            # Do the downsampling
            self.downsample(plot=plot, decimorig=decimorig,norm=norm)

        # All done
        return

    def computeGradientCurvature(self, smooth=None):
        '''
        Computes the gradient for all the blocks.

        Kwargs:
            * smooth        : Smoothes the Gradient and the Curvature using a Gaussian filter. {smooth} is the kernel size (in km) of the filter.

        Returns:
            * None
        '''

        # Get the XY situation
        self.PIXXY = np.vstack((self.image.x, self.image.y)).T

        # Get minimum size
        Bsize = self._is_minimum_size(self.blocks)

        # Gradient (if we cannot compute the gradient, the value is zero, so the algo stops)
        self.Gradient = np.ones(len(self.blocks,))*1e7
        self.Curvature = np.ones(len(self.blocks,))*1e7

        # Create a queue to hold the results
        output = mp.Queue()

        # Check how many workers
        try:
            nworkers = int(os.environ['OMP_NUM_THREADS'])
        except:
            nworkers = mp.cpu_count()

        # Create the workers
        seqindices = _split_seq(range(len(self.blocks)), nworkers)
        workers = [mpgradcurv(self, Bsize, seqindices[w], output) for w in range(nworkers)]

        # start the workers
        for w in range(nworkers): workers[w].start()

        # Collect
        for w in range(nworkers):
            gradient, curvature, igrad = output.get()
            self.Gradient[igrad] = gradient
            self.Curvature[igrad] = curvature

        # Smooth?
        if smooth is not None:
            centers = [self.getblockcenter(block) for block in self.blocks]
            Distances = distance.cdist(centers,centers)**2
            gauss = np.exp(-0.5*Distances/(smooth**2))
            self.Gradient = np.dot(gauss, self.Gradient)/np.sum(gauss, axis=1)
            self.Curvature = np.dot(gauss, self.Curvature)/np.sum(gauss, axis=1)

        # All done
        return

    def dataBased(self, threshold, plot=False, verboseLevel='minimum', decimorig=10, quantity='curvature', smooth=None, itmax=100):
        '''
        Iteratively downsamples the dataset until value compute inside each block is lower than the threshold.
        Threshold is based on the gradient or curvature of the phase field inside the block.
        The algorithm is based on the varres downsampler. Please check at http://earthdef.caltech.edu

        Args:
            * threshold     : Gradient threshold

        Kwargs:
            * plot          : True/False
            * verboseLevel  : Talk to me
            * decimorig     : decimate before plotting
            * quantity      : curvature or gradient
            * smooth        : Smooth the {quantity} spatial with a filter of kernel size of {smooth} km
            * itmax         : Maximum number of iterations

        Returns:
            * None
        '''

        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Downsampling Iterations")

        # Creates the variable that is supposed to stop the loop
        # Check = [False]*len(self.blocks)
        self.Gradient = np.ones(len(self.blocks),)*(threshold+1.)
        self.Curvature = np.ones(len(self.blocks),)*(threshold+1.)
        do_cut = False

        # counter
        it = 0

        # Check
        if quantity is 'curvature':
            testable = self.Curvature
        elif quantity is 'gradient':
            testable = self.Gradient

        # Check if block size is minimum
        Bsize = self._is_minimum_size(self.blocks)

        # Loops until done
        while not (testable<threshold).all() and it<itmax:

            # Check
            assert testable.shape[0]==len(self.blocks), 'Gradient vector has a size different than number of blocks'

            # Cut if asked
            if do_cut:
                # New list of blocks
                newblocks = []
                # Iterate over blocks
                for j in range(len(self.blocks)):
                    block = self.blocks[j]
                    if (testable[j]>threshold) and not Bsize[j]:
                        b1, b2, b3, b4 = self.cutblockinfour(block)
                        newblocks.append(b1)
                        newblocks.append(b2)
                        newblocks.append(b3)
                        newblocks.append(b4)
                    else:
                        newblocks.append(block)
                # Set the blocks
                self.setBlocks(newblocks)
                # Do the downsampling
                self.downsample(plot=False, decimorig=decimorig)
            else:
                do_cut = True

            # Iteration #
            it += 1
            if self.verbose:
                print('Iteration {}: Testing {} data samples '.format(it, len(self.blocks)))

            # Compute resolution
            self.computeGradientCurvature(smooth=smooth)
            if quantity is 'curvature':
                testable = self.Curvature
            elif quantity is 'gradient':
                testable = self.Gradient

            # initialize
            Bsize = self._is_minimum_size(self.blocks)

            if self.verbose and verboseLevel is not 'minimum':
                sys.stdout.write(' ===> Resolution from {} to {}, Mean = {} +- {} \n'.format(testable.min(),
                    testable.max(), testable.mean(), testable.std()))
                sys.stdout.flush()

            # Plot at the end of that iteration
            if plot:
                self.plotDownsampled(decimorig=decimorig)

        # All done
        return

    def resolutionBased(self, threshold, damping, slipdirection='s', plot=False, verboseLevel='minimum', decimorig=10, vertical=False):
        '''
        Iteratively downsamples the dataset until value compute inside each block is lower than the threshold.

        Args:
            * threshold     : Threshold.
            * damping       : Damping coefficient (damping is made through an identity matrix).

        Kwargs:
            * slipdirection : Which direction to accout for to build the slip Green's functions (s, d or t)
            * plot          : False/True
            * verboseLevel  : talk to me
            * decimorig     : decimate a bit before plotting
            * vertical      : Use vertical green's functions.

        Returns:
            * None
        '''

        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Downsampling Iterations")

        # Check if vertical is set properly
        if not vertical and self.datatype is 'insar':
            print("----------------------------------")
            print("----------------------------------")
            print(" Watch Out!!!!")
            print(" We have set vertical to True, because ")
            print(" LOS is always very sensitive to vertical")
            print(" displacements...")
            vertical = True

        # Creates the variable that is supposed to stop the loop
        # Check = [False]*len(self.blocks)
        self.Rd = np.ones(len(self.blocks),)*(threshold+1.)
        do_cut = False

        # counter
        it = 0

        # Check if block size is minimum
        Bsize = self._is_minimum_size(self.blocks)

        # Loops until done
        while not (self.Rd<threshold).all():

            # Check
            assert self.Rd.shape[0]==len(self.blocks), 'Resolution matrix has a size different than number of blocks'

            # Cut if asked
            if do_cut:
                # New list of blocks
                newblocks = []
                # Iterate over blocks
                for j in range(len(self.blocks)):
                    block = self.blocks[j]
                    if (self.Rd[j]>threshold) and not Bsize[j]:
                        b1, b2, b3, b4 = self.cutblockinfour(block)
                        newblocks.append(b1)
                        newblocks.append(b2)
                        newblocks.append(b3)
                        newblocks.append(b4)
                    else:
                        newblocks.append(block)
                # Set the blocks
                self.setBlocks(newblocks)
                # Do the downsampling
                self.downsample(plot=False, decimorig=decimorig)
            else:
                do_cut = True

            # Iteration #
            it += 1
            if self.verbose:
                print('Iteration {}: Testing {} data samples '.format(it, len(self.blocks)))

            # Compute resolution
            self.computeResolution(slipdirection, damping, vertical=vertical)

            # Blocks that have a minimum size, don't check these
            Bsize = self._is_minimum_size(self.blocks)
            self.Rd[np.where(Bsize)] = 0.0

            if self.verbose and verboseLevel is not 'minimum':
                sys.stdout.write(' ===> Resolution from {} to {}, Mean = {} +- {} \n'.format(self.Rd.min(),
                    self.Rd.max(), self.Rd.mean(), self.Rd.std()))
                sys.stdout.flush()

            # Plot at the end of that iteration
            if plot:
                self.plotDownsampled(decimorig=decimorig)

        if self.verbose:
            print(" ")

        # All done
        return

    def computeResolution(self, slipdirection, damping, vertical=False):
        '''
        Computes the resolution matrix in the data space.

        Args:
            * slipdirection : Directions to include when computing the resolution operator.
            * damping       : Damping coefficient (damping is made through an identity matrix).

        Kwargs:
            * vertical      : Use vertical GFs?

        Returns:
            * None
        '''

        # Check if vertical is set properly
        if not vertical and self.datatype is 'insar':
            print("----------------------------------")
            print("----------------------------------")
            print(" Watch Out!!!!")
            print(" We have set vertical to True, because ")
            print(" LOS is always very sensitive to vertical")
            print(" displacements...")
            vertical = True

        # Create the Greens function
        G = None

        # Compute the greens functions for each fault and cat these together
        for fault in self.faults:
            # build GFs
            fault.buildGFs(self.newimage, vertical=vertical, slipdir=slipdirection, verbose=False)
            fault.assembleGFs([self.newimage], polys=None, slipdir=slipdirection, verbose=False)
            # Cat GFs
            if G is None:
                G = fault.Gassembled
            else:
                G = np.hstack((G, fault.Gassembled))
        # Compute the data resolution matrix
        Npar = G.shape[1]
        if self.datatype is 'opticorr':
            Ndat = int(G.shape[0]/2)
        Ginv = np.dot(np.linalg.inv(np.dot(G.T,G)+ damping*np.eye(Npar)),G.T)
        Rd = np.dot(G, Ginv)
        self.Rd = np.diag(Rd).copy()

        # If we are dealing with opticorr data, the diagonal is twice as long as the number of blocks
        if self.datatype is 'opticorr':
            self.Rd = np.sqrt( self.Rd[:Ndat]**2 + self.Rd[-Ndat:]**2 )

        # All done
        return

    def getblockarea(self, block):
        '''
        Returns the total area of a block.

        Args:
            * block : Block as defined in initialstate.

        Returns:
            * float
        '''

        # All done in one line
        return np.abs(block[0][0]-block[1][0]) * np.abs(block[0][1] - block[2][1])

    def trashblock(self, j):
        '''
        Deletes one block.

        Args:
            * j     : index of a block

        Returns:
            * None
        '''

        del self.blocks[j]
        del self.blocksll[j]

        # all done
        return

    def trashblocks(self, jj):
        '''
        Deletes the blocks corresponding to indexes in the list jj.

        Args:
            * jj    : index of a block

        Returns:
            * None
        '''

        while len(jj)>0:

            # Get index
            j = jj.pop()

            # delete it
            self.trashblock(j)

            # upgrade list
            for i in range(len(jj)):
                if jj[i]>j:
                    jj[i] -= 1

        # all done
        return

    def plotDownsampled(self, figure=145, ref='utm', norm=None, data2plot='north', decimorig=1, savefig=None, show=True):
        '''
        Plots the downsampling as it is at this step.

        Kwargs:
            * figure    : Figure ID.
            * ref       : utm or lonlat
            * Norm      : [colormin, colormax]
            * data2plot : used if datatype is opticorr: can be north or east.
            * decimorig : decimate a bit beofre plotting
            * savefig   : True/False
            * show      : display True/False

        Returns:
            * None
        '''

        # Create the figure
        fig = plt.figure(figure, figsize=(10,5))
        full = fig.add_axes([0.05, 0.05, 0.4, 0.8])
        down = fig.add_axes([0.55, 0.05, 0.4, 0.8])
        colr = fig.add_axes([0.4, 0.9, 0.2, 0.03])

        # Set the axes
        if ref is 'utm':
            full.set_xlabel('Easting (km)')
            full.set_ylabel('Northing (km)')
            down.set_xlabel('Easting (km)')
            down.set_ylabel('Northing (km)')
        else:
            full.set_xlabel('Longitude')
            full.set_ylabel('Latitude')
            down.set_xlabel('Longitude')
            down.set_ylabel('Latitude')

        # Get the datasets
        original = self.image
        downsampled = self.newimage

        # Get what should be plotted
        if self.datatype is 'insar':
            data = original.vel
        elif self.datatype is 'opticorr':
            if data2plot is 'north':
                data = original.north
            elif data2plot is 'east':
                data = original.east

        # Vmin, Vmax
        if norm is not None:
            vmin, vmax = Norm
        else:
            vmin = data.min()
            vmax = data.max()

        # Prepare the colormaps
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        cmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # Plot original dataset
        if ref is 'utm':
            # image
            sca = full.scatter(original.x[::decimorig], original.y[::decimorig], s=10, c=data[::decimorig], cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.)
            # Faults
            for fault in self.faults:
                full.plot(fault.xf, fault.yf, '-k')
        else:
            # image
            sca = full.scatter(original.lon[::decimorig], original.lat[::decimorig], s=10, c=data[::decimorig], cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0.)
            # Faults
            for fault in self.faults:
                full.plot(fault.lon, fault.lat, '-k')

        # Patches
        import matplotlib.collections as colls

        # Get downsampled data
        if self.datatype is 'insar':
            downdata = downsampled.vel
        elif self.datatype is 'opticorr':
            if data2plot is 'north':
                downdata = downsampled.north
            elif data2plot is 'east':
                downdata = downsampled.east

        # Image
        for i in range(len(self.blocks)):
            # Get block
            if ref is 'utm':
                block = self.blocks[i]
            else:
                block = self.blocksll[i]
            # Get value
            val = downdata[i]
            # Build patch
            x = [block[j][0] for j in range(4)]
            y = [block[j][1] for j in range(4)]
            verts = [list(zip(x, y))]
            patch = colls.PolyCollection(verts)
            # Set its color
            patch.set_color(scalarMap.to_rgba(val))
            patch.set_edgecolors('k')
            down.add_collection(patch)

        # Faults
        for fault in self.faults:
            if ref is 'utm':
                down.plot(fault.xf, fault.yf, '-k')
            else:
                down.plot(fault.lon, fault.lat, '-k')

        # Color bar
        cb = mpl.colorbar.ColorbarBase(colr, cmap=cmap, norm=cNorm ,orientation='horizontal')

        # Axes
        if ref is 'utm':
            full.set_xlim([self.xmin, self.xmax])
            full.set_ylim([self.ymin, self.ymax])
            down.set_xlim([self.xmin, self.xmax])
            down.set_ylim([self.ymin, self.ymax])
        else:
            full.set_xlim([self.lonmin, self.lonmax])
            full.set_ylim([self.latmin, self.latmax])
            down.set_xlim([self.lonmin, self.lonmax])
            down.set_ylim([self.latmin, self.latmax])

        # Savefig
        if savefig is not None:
            plt.savefig(savefig)

        # Gradient?
        if hasattr(self, 'Gradient'):
            plt.figure()
            centers = [self.getblockcenter(block) for block in self.blocks]
            x = [c[0] for c in centers]
            y = [c[1] for c in centers]
            plt.scatter(x, y, s=20, c=self.Gradient, linewidth=0.1)
            plt.title('Gradient')
            plt.colorbar(orientation='horizontal')
            plt.figure()
            plt.hist(self.Gradient, bins=10)
            plt.title('Gradient')

        # Curvature
        if hasattr(self, 'Curvature'):
            plt.figure()
            centers = [self.getblockcenter(block) for block in self.blocks]
            x = [c[0] for c in centers]
            y = [c[1] for c in centers]
            plt.scatter(x, y, s=20, c=self.Curvature, linewidth=0.1)
            plt.title('Curvature')
            plt.colorbar(orientation='horizontal')
            plt.figure()
            plt.hist(self.Curvature, bins=10)
            plt.title('Curvature')

        # Resolution
        if hasattr(self, 'Resolution'):
            plt.figure()
            centers = [self.getblockcenter(block) for block in self.blocks]
            x = [c[0] for c in centers]
            y = [c[1] for c in centers]
            plt.scatter(x, y, s=20, c=self.Rd, linewidth=0.1)
            plt.title('Resolution')
            plt.colorbar(orientation='horizontal')
            plt.figure()
            plt.hist(self.Resolution, bins=10)
            plt.title('Resolution')

        # All done
        if show:
            plt.show()

        return

    def reject_pixels_fault(self, distance, fault):
        '''
        Removes pixels that are too close to the fault in the downsampled image.

        Args:
            * distance      : Distance between pixels and fault (scalar)
            * fault         : fault object

        Returns:
            * None
        '''

        # Get image
        image = self.newimage

        # Clean it
        u = image.reject_pixels_fault(distance, fault)

        # Clean blocks
        j = 0
        for i in u.tolist():
            self.blocks.pop(i-j)
            self.blocksll.pop(i-j)
            j += 1

        # All done
        return

    def buildDownsampledCd(self, mu, lam, function='exp'):
        '''
        Builds the covariance matrix by weihgting following the downsampling scheme

        Args:
            * mu        : Autocovariance
            * lam       : Characteristic distance

        Kwargs:
            * function  : 'exp' (:math:`C = \mu^2 e^{\\frac{-d}{\lambda}}`) or 'gauss' (:math:`C = \mu^2 e^{\\frac{-d^2}{2\lambda^2}}`)

        Returns:
            * None
        '''

        assert self.image.dtype=='insar', 'Not implemented for opticorr, too lazy.... Sorry.... Later....'

        # How many samples
        nSamples = self.newimage.lon.shape[0]

        # Create Array
        Cd = np.zeros((nSamples, nSamples))

        # Create the whole geometry
        PIXXY = np.vstack((self.image.x, self.image.y)).T

        # Iterate
        for i in range(nSamples):
            for j in range(i, nSamples):

                # Get blocks
                iBlock = self.blocks[i]
                jBlock = self.blocks[j]

                # Get the pixels concerned
                iPath = path.Path(iBlock, closed=False)
                jPath = path.Path(jBlock, closed=False)
                ii = iPath.contains_points(PIXXY)
                jj = jPath.contains_points(PIXXY)

                # How many pixels
                iSamples = len(np.flatnonzero(ii))
                jSamples = len(np.flatnonzero(jj))

                # Create 2 newimages
                Image = insar('Image', utmzone=self.utmzone, verbose=False, lon0=self.lon0, lat0=self.lat0)

                # Fill them
                Image.x = np.hstack((self.image.x[ii], self.image.x[jj]))
                Image.y = np.hstack((self.image.y[ii], self.image.y[jj]))

                # Create a covariance object
                Cov = imcov('Block i and j', Image, verbose=False)

                # Set sigmas
                Cov.datasets['Block i and j']['Sigma'] = mu
                Cov.datasets['Block i and j']['Lambda'] = lam
                Cov.datasets['Block i and j']['function'] = function

                # Compute local covariances and take only the covariance part
                localCd = Cov.buildCovarianceMatrix(Image, 'Block i and j')[:iSamples,jSamples:]

                # Average
                Cd[i,j] = np.sum(localCd)/(iSamples*jSamples)
                Cd[j,i] = np.sum(localCd)/(iSamples*jSamples)

        # All done
        return Cd

    def setDownsamplingScheme(self, sampler):
        '''
        From an imagedownsampling object, sets the downsampling scheme.

        Args:
            * sampler      : imagedownsampling instance

        Returns:
            * None
        '''

        # Check if it has bocks
        assert hasattr(sampler, 'blocks'), 'imagedownsampling instance {} needs to have blocks'.format(sampler.name)

        # set the blocks
        self.tolerance = sampler.tolerance
        self.setBlocks(sampler.blocks)

        # All done
        return

    def readDownsamplingScheme(self, prefix):
        '''
        Reads a downsampling scheme from a rsp file and set it as self.blocks

        Args:
            * prefix          : Prefix of a .rsp file written by writeDownsampled2File.

        Returns:
            * None
        '''

        # Replace spaces
        prefix = prefix.replace(" ", "_")

        # Open file
        frsp = open(prefix+'.rsp', 'r')

        # Create a block list
        blocks = []

        # Read all the file
        Lines = frsp.readlines()

        # Close the file
        frsp.close()

        # Loop
        for line in Lines[2:]:
            ulx, uly, drx, dry = [np.float(line.split()[i]) for i in range(2,6)]
            c1 = [ulx, uly]
            c2 = [drx, uly]
            c3 = [drx, dry]
            c4 = [ulx, dry]
            blocks.append([c1, c2, c3, c4])

        # Set the blocks
        self.setBlocks(blocks)

        # All done
        return

    def writeDownsampled2File(self, prefix, rsp=False):
        '''
        Writes the downsampled image data to a file. The file will be called prefix.txt. If rsp is True, then it writes a file called prefix.rsp containing the boxes of the downsampling. If prefix has white spaces, those are replaced by "_".

        Args:
            * prefix        : Prefix of the output file

        Kwargs:
            * rsp           : Write the rsp file?

        Returns:
            * None
        '''

        # Replace spaces
        prefix = prefix.replace(" ", "_")

        # Open files
        ftxt = open(prefix+'.txt', 'w')
        if rsp:
            frsp = open(prefix+'.rsp', 'w')

        # Write the header
        if self.datatype is 'insar':
            ftxt.write('Number xind yind east north data err wgt Elos Nlos Ulos\n')
        elif self.datatype is 'opticorr':
            ftxt.write('Number Lon Lat East North EastErr NorthErr \n')
        ftxt.write('********************************************************\n')
        if rsp:
            frsp.write('xind yind UpperLeft-x,y DownRight-x,y\n')
            frsp.write('********************************************************\n')

        # Loop over the samples
        for i in range(len(self.newimage.x)):

            # Write in txt
            wgt = self.newimage.wgt[i]
            x = int(self.newimage.x[i])
            y = int(self.newimage.y[i])
            lon = self.newimage.lon[i]
            lat = self.newimage.lat[i]
            if self.datatype is 'insar':
                vel = self.newimage.vel[i]
                err = self.newimage.err[i]
                elos = self.newimage.los[i,0]
                nlos = self.newimage.los[i,1]
                ulos = self.newimage.los[i,2]
                strg = '{:4d} {:4d} {:4d} {:3.6f} {:3.6f} {} {} {} {} {} {}\n'\
                    .format(i, x, y, lon, lat, vel, err, wgt, elos, nlos, ulos)
            elif self.datatype is 'opticorr':
                east = self.newimage.east[i]
                north = self.newimage.north[i]
                err_east = self.newimage.err_east[i]
                err_north = self.newimage.err_north[i]
                strg = '{:4d} {:3.6f} {:3.6f} {} {} {} {} \n'\
                        .format(i, lon, lat, east, north, err_east, err_north)
            ftxt.write(strg)

            # Write in rsp
            if rsp:
                ulx = self.blocks[i][0][0]
                uly = self.blocks[i][0][1]
                drx = self.blocks[i][2][0]
                dry = self.blocks[i][2][1]
                ullon = self.blocksll[i][0][0]
                ullat = self.blocksll[i][0][1]
                drlon = self.blocksll[i][2][0]
                drlat = self.blocksll[i][2][1]
                strg = '{:4d} {:4d} {} {} {} {} {} {} {} {} \n'\
                        .format(x, y, ulx, uly, drx, dry, ullon, ullat, drlon, drlat)
                frsp.write(strg)

        # Close the files
        ftxt.close()
        if rsp:
            frsp.close()

        # All done
        return

    def _is_minimum_size(self, blocks):
        '''
        Returns a Boolean array. True if block is minimum size, False either.
        '''

        # Initialize
        Bsize = []

        # loop
        for block in self.blocks:
            w = block[1][0] - block[0][0]
            if w<=self.minsize:
                Bsize.append(True)
            else:
                Bsize.append(False)

        # All done
        return Bsize

    def distToFault(self,block):
        '''
        Returns distance from block to fault. The distance is here defined as the minimum distance from any of the four block corners to the fault. (R.Grandin, April 2015)

        Args:
            * block     : Block instance of the imagedownsampling class.

        Returns:
            * None
        '''

        # Get the four corners
        c1, c2, c3, c4 = block
        x1, y1 = c1
        x2, y2 = c2
        x3, y3 = c3
        x4, y4 = c4

        # Compute the position of the center
        xc = x1 + (x2 - x1)/2.
        yc = y1 + (y4 - y1)/2.

        # Faults
        distMin=99999999.
        for fault in self.faults:
            distCorner1=np.min(np.hypot(fault.xf-x1,fault.yf-y1))
            distCorner2=np.min(np.hypot(fault.xf-x2,fault.yf-y2))
            distCorner3=np.min(np.hypot(fault.xf-x3,fault.yf-y3))
            distCorner4=np.min(np.hypot(fault.xf-x4,fault.yf-y4))
            distMin=np.min([distMin,distCorner1,distCorner2,distCorner3,distCorner4])

        # all done
        return distMin

    def blockSize(self,block):
        '''
        Returns block size. R.Grandin, April 2015

        Args:
            * block     : Block instance of the imagedownsampling class.

        Returns:
            * None
        '''

        # compute the size
        BlockSizeW = block[1][0] - block[0][0]

        # all done
        return BlockSizeW

    def _isItAGoodBlock(self, block, num):
        '''
        Returns True or False given the criterion

        Args:
            * block     : Shape of the block
            * num       : Number of pixels
        '''

        if self.tolerance<1.:
            coveredarea = num*self.pixelArea
            blockarea = self.getblockarea(block)
            return coveredarea/blockarea>self.tolerance
        else:
            return num>=self.tolerance

        # All done
        return

#EOF
