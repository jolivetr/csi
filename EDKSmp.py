'''
A bunch of routines to handle EDKS

Written by F. Ortega in 2010.
Modified by R. Jolivet in 2014.
Modified by R. Jolivet in 2017 (multiprocessing added for point dropping)
'''

# Externals
import os
import struct 
import sys
import numpy as np
import multiprocessing as mp

# Scipy
from scipy.io import FortranFile
import scipy.interpolate as sciint

# Initialize a class to allow multiprocessing for EDKS interpolation in Python
class interpolator(mp.Process):
    '''
    Multiprocessing class runing the edks interpolation.
    This class requires one to build the interpolator in advance.

    Args:
        * interpolators     : List of interpolators
        * queue             : Instance of mp.Queue
        * depths            : depths (first dimnesion of the interpolators) 
        * distas            : distances (second dimension of the interpolators)
        * istart            : starting point
        * iend              : ending point

    Returns:
        * None
    '''

    # ----------------------------------------------------------------------
    # Initialize
    def __init__(self, interpolators, queue, depths, distas, istart, iend):

        # Save things
        self.interpolators = interpolators
        self.depths = depths
        self.distas = distas
        self.istart = istart
        self.iend = iend

        # Save the queue
        self.queue = queue 

        # Initialize the process
        super(interpolator, self).__init__()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Run method
    def run(self):
        '''
        Run the interpolation
        '''
        
        # Interpolate
        values = []
        for inter in self.interpolators:
            values.append(inter(np.vstack((self.depths[self.istart:self.iend],
                                           self.distas[self.istart:self.iend])).T))
        
        # Save start/end
        values.append((self.istart, self.iend))

        # Store output
        self.queue.put(values)
        
        # All done
        return
    # ----------------------------------------------------------------------

# Initialize a class to allow multiprocessing to drop points
class pointdropper(mp.Process):
    '''
    Initialize the multiprocessing class to run the point dropper.
    This class drops point sources in the triangular or rectangular mesh.

    Args:
        * fault             : Instance of Fault.py
        * queue             : Instance of mp.Queue
        * charArea          : Characteristic area of the subfaults
        * istart            : Index of the first patch to deal with
        * iend              : Index of the last pacth to deal with

    Returns:
        * None
    '''

    # ----------------------------------------------------------------------
    # Initialize
    def __init__(self, fault, queue, charArea, istart, iend):

        # Save the fault
        self.fault = fault
        self.charArea = charArea
        self.istart = istart
        self.iend = iend

        #print(istart, iend)

        # Save the queue
        self.queue = queue

        # Initialize the Process
        super(pointdropper, self).__init__()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Run routine needed by multiprocessing
    def run(self):
        '''
        Run the subpatch construction
        '''

        # Create lists
        Ids, Xs, Ys, Zs, Strike, Dip, Area = [], [], [], [], [], [], []
        allSplitted = []

        # Iterate overthe patches
        for i in range(self.istart, self.iend):

            # Get patch
            patch = self.fault.patch[i]

            # Check if the Area is bigger than the target
            area = self.fault.patchArea(patch)
            if area>self.charArea[i]:
                keepGoing = True
                tobeSplitted = [patch]
                splittedPatches = []
            else: 
                keepGoing = False
                print('Be carefull, patch {} has not been refined into point sources'.format(self.fault.getindex(patch)))
                print('Possible causes: Area = {}, Nodes = {}'.format(area, patch))
                tobeSplitted = []
                splittedPatches = [patch]

            # Iterate
            while keepGoing:
                
                # Take a patch
                p = tobeSplitted.pop()

                # Split into 4 patches
                Splitted = self.fault.splitPatch(p)

                # Check the area
                for splitted in Splitted:
                    # get area
                    area = self.fault.patchArea(splitted)
                    # check 
                    if area<self.charArea[i]:
                        splittedPatches.append(splitted)
                    else:
                        tobeSplitted.append(splitted)

                # Do we continue?
                if len(tobeSplitted)==0:
                    keepGoing = False

                # Do we have a limit
                if hasattr(self.fault, 'maximumSources'):
                    if len(splittedPatches)>=self.fault.maximumSources:
                        keepGoing = False

            # When all done get their centers
            geometry = [self.fault.getpatchgeometry(p, center=True)[:3] for p in splittedPatches]
            x, y, z = zip(*geometry)
            strike, dip = self.fault.getpatchgeometry(patch)[5:7] 
            strike = np.ones((len(x),))*strike
            strike = strike.tolist()
            dip = np.ones((len(x),))*dip
            dip = dip.tolist()
            areas = [self.fault.patchArea(p) for p in splittedPatches]
            ids = np.ones((len(x),))*(i)
            ids = ids.astype(np.int).tolist()

            # Save
            Ids += ids
            Xs += x
            Ys += y
            Zs += z
            Strike += strike
            Dip += dip
            Area += areas
            allSplitted += splittedPatches

        # Put in the Queue
        self.queue.put([Ids, Xs, Ys, Zs, Strike, Dip, Area, allSplitted])

        # all done
        return
    # ----------------------------------------------------------------------
# end of pointdropper

# ----------------------------------------------------------------------
def dropSourcesInPatches(fault, verbose=False, returnSplittedPatches=False):
    '''
    From a fault object, returns sources to be given to sum_layered_sub.
    The number of sources is determined by the spacing provided in fault.

    Args:
        * fault                   : instance of Fault (Rectangular or Triangular).
        * verbose                 : Talk to me
        * returnSplittedPactches  : Returns a triangularPatches object with the splitted 
                                  patches.

    Return:
        * Ids                   : Id of the subpatches
        * Xs                    : UTM x-coordinate of the subpatches (km)
        * Ys                    : UTM y-coordinate of the subpatches (km)
        * Zs                    : UTM z-coordinate of the subpatches (km)
        * Strikes               : Strike angles of the subpatches (rad)
        * Dips                  : Dip angles of the subpatches (rad)
        * Areas                 : Area of the subpatches (km^2)
        
        if returnSplittedPatches:
        * splitFault            : Fault object with the subpatches
    '''

    # Create lists
    Id, X, Y, Z, Strike, Dip, Area = [], [], [], [], [], [], []
    Splitted = []

    # Check
    if (not hasattr(fault, 'sourceSpacing')) and (not hasattr(fault, 'sourceNumber')) and (not hasattr(fault, 'sourceArea')):
        print('EDKS: Need to provide area, spacing or number of sources...')
        sys.exit(1)
    if hasattr(fault, 'sourceSpacing') and hasattr(fault, 'sourceNumber') and hasattr(fault, 'sourceArea'):
        print('EDKS: Please delete sourceSpacing, sourceNumber or sourceArea...')
        print('EDKS: I do not judge... You decide...')
        sys.exit(1)

    # show me
    if verbose:
        print('Dropping point sources') 

    # Spacing
    if hasattr(fault, 'sourceArea'):
        area = fault.sourceArea
        charArea = np.ones((len(fault.patch),))*area
    if hasattr(fault, 'sourceSpacing'):
        spacing = fault.sourceSpacing
        if fault.patchType == 'rectangle':
            charArea = np.ones((len(fault.patch),))*spacing**2
        elif fault.patchType in ('triangle', 'triangletent'):
            charArea = np.ones((len(fault.patch),))*spacing**2/2.
    if hasattr(fault, 'sourceNumber'):
        number = fault.sourceNumber
        fault.computeArea()
        charArea = np.array(fault.area)/np.float(number)

    # Create a queue
    output = mp.Queue()

    # how many workers
    try:
        nworkers = int(os.environ['OMP_NUM_THREADS'])
    except:
        nworkers = mp.cpu_count()

    # how many patches
    npatches = len(fault.patch)

    # Create them
    workers = [pointdropper(fault, output, charArea, 
                            np.int(np.floor(i*npatches/nworkers)), 
                            np.int(np.floor((i+1)*npatches/nworkers))) for i in range(nworkers)]
    workers[-1].iend = npatches

    # Start them
    for w in range(nworkers): workers[w].start()
    # I don't understand why this guy does not work...
    #for w in range(nworkers): workers[w].join()

    # Get things from the queue
    for i in range(nworkers):
        ids, xs, ys, zs, strike, dip, area, splitted = output.get()
        Id.extend(ids)
        X.extend(xs) 
        Y.extend(ys)
        Z.extend(zs) 
        Strike.extend(strike)
        Dip.extend(dip)
        Area.extend(area)
        Splitted.extend(splitted)

    # Make arrays
    isort = np.argsort(Id)
    Ids = np.array([Id[i] for i in isort])
    Xs = np.array([X[i] for i in isort])
    Ys = np.array([Y[i] for i in isort])
    Zs = np.array([Z[i] for i in isort])
    Strikes = np.array([Strike[i] for i in isort])
    Dips = np.array([Dip[i] for i in isort])
    Areas = np.array([Area[i] for i in isort])
    allSplitted = [Splitted[i] for i in isort]

    # All done
    if returnSplittedPatches:
        from .TriangularPatches import TriangularPatches as trianglePatches
        splitFault = trianglePatches('Splitted {}'.format(fault.name), 
                                     utmzone=fault.utmzone, 
                                     lon0=fault.lon0,
                                     lat0=fault.lat0,
                                     ellps=fault.ellps,
                                     verbose=verbose)
        # set up patches
        splitFault.patch = [np.array(p) for p in allSplitted]
        splitFault.patch2ll()
        # Patches 2 vertices
        splitFault.setVerticesFromPatches()
        # Depth
        splitFault.setdepth()
        return Ids, Xs, Ys, Zs, Strikes, Dips, Areas, splitFault
    else:
        return Ids, Xs, Ys, Zs, Strikes, Dips, Areas
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Compute the Green's functions for the patches
def sum_layered(xs, ys, zs, strike, dip, rake, slip, width, length,\
                npw, npy,\
                xr, yr, edks,\
                prefix, \
                BIN_EDKS = 'EDKS_BIN',
                cleanUp=True, verbose=True):
    '''
    Compute the Green's functions for the given patches

    Args:

        <-- Sources --> 1-D numpy arrays

            * xs                : m, east coord to center of fault patch
            * ys                : m, north coord to center of fault patch
            * zs                : m,depth coord to center of fault patch (+ down) 
            * strike            : deg, clockwise from north 
            * dip               : deg, 90 is vertical 
            * rake              : deg, 0 left lateral strike slip, 90 up-dip slip 
            * slip              : m, slip in the rake direction
            * width             : m, width of the patch
            * length            : m, length of the patch
            * npw               : integers, number of sources along strike
            * npy               : integers, number of sources along dip 
    
        <-- Receivers --> 1-D numpy arrays

            * xr                : m, east coordinate of receivers 
            * yr                : m, north coordinate of receivers 

        <-- Elastic structure -->

            * edks              : string, full name of edks file, e.g., halfspace.edks

        <-- File Naming -->

            * prefix            : string, prefix for the files generated by sum_layered
    
    Kwargs:

            * BIN_EDKS          : Environement variable where EDKS executables are.
            * cleanUp           : Remove the intermediate files
            * verbose           : Talk to me

    Return:
        <-- 2D arrays (#receivers, #fault patches) -->
            * ux                : m, east displacement
            * uy                : m, west displacement
            * uz                : m, up displacement (+ up)
    '''

    # Get executables
    BIN_EDKS = os.environ[BIN_EDKS]

    # Some initializations
    Np = len(xs)            # number of sources
    nrec = len(xr)          # number of receivers
    A = length*width        # Area of the patches

    # Some formats
    BIN_FILE_FMT = 'f' # python float = C/C++ float = Fortran 'real*4' 
    NBYTES_FILE_FMT = 4  # a Fortran (real*4) uses 4 bytes.

    # convert sources from center to top edge of fault patch ("sum_layered" needs that)
    sind = np.sin( dip * np.pi / 180.0 )
    cosd = np.cos( dip * np.pi / 180.0 )
    sins = np.sin( strike * np.pi / 180.0 )
    coss = np.cos( strike * np.pi / 180.0 )

    # displacement in local coordinates (phi, delta)
    dZ = (width/2.0) * sind
    dD = (width/2.0) * cosd

    # rotation to global coordinates 
    xs = xs - dD * coss
    ys = ys + dD * sins
    zs = zs - dZ

    # Define filenames:
    file_rec = prefix + '.rec'
    file_pat = prefix + '.pat'
    file_dux = prefix + '_ux.dis'
    file_duy = prefix + '_uy.dis'
    file_duz = prefix + '_uz.dis'

    # Clean the file if they exist
    cmd = 'rm -f {} {} {} {} {}'.format(file_rec, file_pat, file_dux, file_duy, file_duz)
    os.system(cmd) 
    
    # write receiver location file (observation points)
    temp = [xr, yr]
    file = open(file_rec, 'wb') 
     
    for k in range(0, nrec):
       for i in range(0, len(temp)):
          file.write( struct.pack( BIN_FILE_FMT, temp[i][k] ) )       
    file.close() 
  
    # write point sources information
    temp = [xs, ys, zs, strike, dip, rake, width, length, slip]
    file = open(file_pat, 'wb');
    for k in range(0, Np):
       for i in range(0, len(temp)):
          file.write( struct.pack( BIN_FILE_FMT, temp[i][k] ) )
    file.close()
  
    # call sum_layered
    cmd = '{}/sum_layered {} {} {} {} {} {}'.format(BIN_EDKS, edks, prefix, nrec, Np, npw, npy)
    if verbose:
        print(cmd)
    os.system(cmd)
     
    # read sum_layered output Greens function
    # ux
    ux = np.fromfile(file_dux, 'f').reshape((nrec, Np), order='FORTRAN')

    # uy
    uy = np.fromfile(file_duy, 'f').reshape((nrec, Np), order='FORTRAN')
 
    # uz
    uz = np.fromfile(file_duz, 'f').reshape((nrec, Np), order='FORTRAN')
 
    # remove IO files.
    if cleanUp:
        cmd = 'rm -f {} {} {} {} {}'.format(file_rec, file_pat, file_dux, file_duy, file_duz)
        os.system(cmd)  
 
    # return the GF matrices
    return [ux, uy, uz]
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# A class that interpolates edks Kernels (same as fortran's sum_layered, 
# but with more flexibility for the interpolation part)

class interpolateEDKS(object):
    
    '''
    A class that will interpolate the EDKS Kernels and produce Green's
    functions in a stratified medium. This class will only use point 
    sources as the summation is done in the fault object.

    What goes in this class is a translation of the point source case of 
    EDKS. We use the case where slip perpendicular to the rake angle is 
    equal to zero.

    Args:
        * kernel    : EDKS Kernel file (mykernel.edks). One needs to 
                      provide the header file as well (hdr.mykernel.edks)
    '''

    def __init__(self, kernel, verbose=True):

        # Set verbose
        self.verbose = verbose

        # Set kernel
        self.kernel = kernel

        # Make sure we start on the right foot
        self.interpolationDone = False

        # All done
        return

    def readHeader(self):
        '''
        Read the EDKS Kernel header file and stores it in {self}

        Returns:
            * None
        '''

        # Show me
        if self.verbose:
            print('Read Kernel Header file hdr.{}'.format(self.kernel))

        # Open the header file
        fhd = open('hdr.{}'.format(self.kernel), 'r')

        # Read things
        self.prefix = fhd.readline().split()[0]
        self.nlayer = int(fhd.readline().split()[0])

        # Layer characteristics
        rho = [] 
        alpha = []
        beta = []
        thickness = []

        # Iterate over layers
        for i in range(self.nlayer):
            line = fhd.readline().split()
            rho.append(float(line[0]))
            alpha.append(float(line[1]))
            beta.append(float(line[2]))
            thickness.append(float(line[3]))

        # Software date 
        self.softwareDate = fhd.readline().split()
        self.softwareVersion = fhd.readline().split()
        self.softwareComments = fhd.readline().split()

        # Depths, Distances
        depths = fhd.readline().split()
        self.depthmin = float(depths[0])
        self.depthmax = float(depths[1])
        self.ndepth = int(depths[2])
        distances = fhd.readline().split()
        self.distamin = float(distances[0])
        self.distamax = float(distances[1])
        self.ndista = int(distances[2])

        # Close file
        fhd.close()

        # All done
        return

    def readKernel(self):
        '''
        Read the EDKS Kernel and stores it in {self}

        Returns:
            * None
        '''

        # Show me
        if self.verbose:
            print('Read Kernel file {}'.format(self.kernel))

        # Open the file
        fedks = FortranFile(self.kernel, 'r')

        # Read
        kernel = fedks.read_reals(np.float32).reshape((self.ndepth*self.ndista,12))

        # Save
        self.depths = kernel[:,0]
        self.distas = kernel[:,1]
        self.zrtdsx = kernel[:,2:]

        # CLose file
        fedks.close()

    def interpolate(self, xs, ys, zs, strike, dip, rake, area, slip, xr, yr, method='linear'):
        '''
        Interpolate the Green's functions for a given source in (xs, ys, zs) with
        a strike, dip and rake and slip parameters and a given receiver (xr, yr)

        Args:
            * xs, ys, zs    : Source location (floats or np.array)
            * strike        : strike angle (rad)
            * dip           : dip angle (rad)
            * rake          : rake angle (rad, 0 left-lateral strike slip, 2pi pure thrust)
            * slip          : Slip value. The unit of slip will condition the unit of the output displacement
            * area          : Area of the point source
            * xr, yr        : Receiver location (floats or np.array)

        Kwargs:
            * method        : Interpolation scheme. Can be linear, nearest or CloughTocher.

        Returns:
            * G             : np.array
        '''

        # Arrange things
        if type(xs) in (float, np.float64, np.float32):
            xs = np.array([xs])
            ys = np.array([ys])
            zs = np.array([zs])
            strike = np.array([strike])
            dip = np.array([dip])
            rake = np.array([rake])
            slip = np.array([slip])
            area = np.array([area])
        if type(xr) in (float, np.float64, np.float32):
            xr = np.array([xr])
            yr = np.array([yr])

        # convert sources from center to top edge of fault patch 
        sind = np.sin( dip )
        cosd = np.cos( dip )
        sins = np.sin( strike )
        coss = np.cos( strike )

        # displacement in local coordinates (phi, delta)
        dZ = (np.sqrt(area)/2.0) * sind
        dD = (np.sqrt(area)/2.0) * cosd

        # rotation to global coordinates 
        xs = xs - dD * coss
        ys = ys + dD * sins
        zs = zs - dZ

        #Show me
        if self.verbose:
            print('Interpolate GFs for {} sources and {} receivers'.format(len(xs), len(xr)))

        # Get moment (here potency)
        M = self.src2mom(slip, area, strike, dip, rake)

        # Create an interpolator
        self.createInterpolator(method=method)

        # Calculate geometry -- dim(r) is (sources, receivers)
        if self.verbose:
            print('Calculate geometry')
        distance, depth, caz, saz, c2az, s2az = self._getGeometry(xs, ys, zs, xr, yr)

        if not self.interpolationDone:

            # Interpolate
            if self.verbose:
                print('Interpolate')
            
            # Create holder
            self.interpKernels = np.zeros((len(xs)*len(xr), 10))
            
            # Multiprocessing
            try:
                nworkers = int(os.environ['OMP_NUM_THREADS'])
            except:
                nworkers = mp.cpu_count()

            # Create a queue 
            output = mp.Queue()

            # Create the workers
            todo = len(distance.flatten())
            workers = [interpolator(self.interpolators, output, 
                                    depth.flatten(), distance.flatten(), 
                                    np.int(np.floor(i*todo/nworkers)),
                                    np.int(np.floor((i+1)*todo/nworkers))) for i in range(nworkers)]
            workers[-1].iend = todo

            # Start
            for w in range(nworkers): workers[w].start()

            # Get from the queue
            for worker in workers:
                values = output.get()
                istart,iend = values.pop()
                for iv,value in enumerate(values):
                    self.interpKernels[istart:iend,iv] = value
            
            # Reshape
            self.interpKernels = self.interpKernels.reshape((len(xs),len(xr),10))

            # reconstruction of the actual displacement Xiaobi vs  Herrman
            # The coefficients created by tab5 are in Xiaobi's notation
            # The equations just below follow Herrman's notation; hence
            self.interpKernels[:,:,1] *= -1.
            self.interpKernels[:,:,4] *= -1.
            self.interpKernels[:,:,7] *= -1.

            # Set it to True
            self.interpolationDone = True

        else:
            if self.verbose:
                print('Use interpolated kernels')

        # Get what's done 
        kernels = self.interpKernels

        # Vertical component  (positive down)
        ws = M[:,np.newaxis,1]*( kernels[:,:,2]*c2az/2. - kernels[:,:,0]/6. + kernels[:,:,8]/3.) \
           + M[:,np.newaxis,2]*(-kernels[:,:,2]*c2az/2. - kernels[:,:,0]/6. + kernels[:,:,8]/3.) \
           + M[:,np.newaxis,0]*( kernels[:,:,0] + kernels[:,:,8])/3. \
           + M[:,np.newaxis,5]*  kernels[:,:,2]*s2az \
           + M[:,np.newaxis,3]*  kernels[:,:,1]*caz \
           + M[:,np.newaxis,4]*  kernels[:,:,1]*saz
   
        # Radial component    (positive away from the source)
        qr = M[:,np.newaxis,1]*( kernels[:,:,5]*c2az/2. - kernels[:,:,3]/6. + kernels[:,:,9]/3.) \
           + M[:,np.newaxis,2]*(-kernels[:,:,5]*c2az/2. - kernels[:,:,3]/6. + kernels[:,:,9]/3.) \
           + M[:,np.newaxis,0]*( kernels[:,:,3] + kernels[:,:,9])/3. \
           + M[:,np.newaxis,5]*  kernels[:,:,5]*s2az \
           + M[:,np.newaxis,3]*  kernels[:,:,4]*caz \
           + M[:,np.newaxis,4]*  kernels[:,:,4]*saz 
   
        # Tangential component (positive if clockwise from zenithal view)
        vt = M[:,np.newaxis,1]*kernels[:,:,7]*s2az/2. \
           - M[:,np.newaxis,2]*kernels[:,:,7]*s2az/2. \
           - M[:,np.newaxis,5]*kernels[:,:,7]*c2az \
           + M[:,np.newaxis,3]*kernels[:,:,6]*saz \
           - M[:,np.newaxis,4]*kernels[:,:,6]*caz 

        # Cartesian components
        Ux = qr*saz + vt*caz
        Uy = qr*caz - vt*saz
        Uz = -ws

        # All done
        return Ux.T, Uy.T, Uz.T

    def createInterpolator(self, method='linear'):
        '''
        Create the interpolation method. This is based on scipy.interpolate.LinearNDInterpolator.

        Returns:
            * None
        '''

        # Create Arrays
        depths = np.unique(self.depths)
        distas = np.unique(self.distas)
        values = self.zrtdsx.reshape((self.ndepth, self.ndista,10))

        # Create the interpolators (if points fall outside the interpolating box, 
        # the value will be extrapolated)
        self.interpolators = [sciint.RegularGridInterpolator((depths, 
                                                             distas), 
                                                             values[:,:,i],
                                                             method=method, 
                                                             bounds_error=False, 
                                                             fill_value=None) \
                                                             for i in range(10)]

        # All done
        return

    def src2mom(self, slip, area, strike, dip, rake):
        '''
        Convert slip and point source geometry to moment.

        Args:
            * slip          : Slip value (m). 
            * area          : Area of the point source (m^2)
            * strike        : Strike angle (rad)
            * dip           : Dip angle (rad)
            * rake          : Rake angle (rad, 0 left-lateral strike slip, 2pi pure thrust)

        Returns:
            * M             : Moment tensor
        '''

        # Nor
        nor = []
        nor.append(-1.*np.sin(dip)*np.sin(strike))
        nor.append(np.sin(dip)*np.cos(strike))
        nor.append(-1.*np.cos(dip))

        # Sli
        s = []
        s.append((np.cos(rake)*np.cos(strike)+np.cos(dip)*np.sin(rake)*np.sin(strike))*slip)
        s.append((np.cos(rake)*np.sin(strike)-np.cos(dip)*np.sin(rake)*np.cos(strike))*slip)
        s.append((-1.*np.sin(rake)*np.sin(dip))*slip)

        # Iterate
        Maki = np.zeros((3,3,len(dip)))
        for ix in range(3):
            for iy in range(3):
                Maki[iy,ix,:] += nor[ix]*s[iy] + nor[iy]*s[ix]

        # Order
        M = np.zeros((len(dip),6))
        M[:,0] = Maki[2,2,:]
        M[:,1] = Maki[0,0,:]
        M[:,2] = Maki[1,1,:]
        M[:,3] = Maki[2,0,:]
        M[:,4] = Maki[2,1,:]
        M[:,5] = Maki[1,0,:]

        # All done
        return area[:,np.newaxis]*M

# PRIVATE METHODS
    def _getGeometry(self, xs, ys, zs, xr, yr):
        '''
        Returns some geometrical features
        
        Args:
            * xs, ys, zs    : Source location (floats or np.array)
            * xr, yr        : Receiver location (floats or np.array)

        Returns:
            * distance, depth, caz, saz, c2az, s2az 
        '''
        
        # Machine precision
        eps = np.finfo(float).eps

        # Compute geometry
        distance = np.sqrt( (xs[:,np.newaxis] - xr[np.newaxis,:])**2 +\
                            (ys[:,np.newaxis] - yr[np.newaxis,:])**2)
        depth = zs[:,np.newaxis]*np.ones(distance.shape)
        caz = (yr[np.newaxis,:] - ys[:,np.newaxis])/distance
        saz = (xr[np.newaxis,:] - xs[:,np.newaxis])/distance
        caz[distance<=eps] = 1.
        saz[distance<=eps] = 0.
        c2az = 2.*caz*caz - 1.
        s2az = 2.*saz*caz

        return distance, depth, caz, saz, c2az, s2az

#EOF
