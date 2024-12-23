'''
A surface ground motion with interpolating functions

Written by R. Jolivet in 2024 (first lines at GFZ)

'''

# Import Externals stuff
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.linalg
import copy
import sys
import os

# Spline specific library
import ndsplines

# Personals
from .SourceInv import SourceInv

#class SurfaceMotion
class SurfaceMotion(SourceInv):

    '''
        A class implementing a surface interpolation for ground displacement

        You can specify either an official utm zone number or provide
        longitude and latitude for a custom zone.

        Args:
            * name          : Name of the surface
            * utmzone       : UTM zone  (optional, default=None)
            * lon0          : Longitude defining the center of the custom utm zone
            * lat0          : Latitude defining the center of the custom utm zone
            * ellps         : ellipsoid (optional, default='WGS84')
    '''

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Base class init
        super(SurfaceMotion,self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)

        # Initialize the fault
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing surface motion {}".format(self.name))
        self.verbose = verbose
        
        self.type = "Surface"

        # Specify the type of patch
        self.patchType = None

        # Set the reference point in the x,y domain (not implemented)
        self.xref = 0.0
        self.yref = 0.0

        # Allocate attributes
        self.x    = None # Position of the functions
        self.y    = None
        self.lon  = None
        self.lat  = None
        self.motion = None
        self.Nmotion = None

        # Create a dictionnary for the polysol
        self.polysol = {}

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # A method that returns the node position in an array
    def getnodes(self, coords='xy'):
        '''
        Return the node positions (xy or ll coordinates)

        Kwargs:
            * coords:   'xy' or 'll'

        Returns:
            * nodes: a Nx2 array of the node positions at the surface
        '''

        # Check 
        if coords=='xy':
            x = self.x
            y = self.y
        elif coords=='ll':
            x = self.lon
            y = self.lat

        # All done
        return np.array([(i,j) for i,j in zip(x,y)])
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Initialize the slip vector
    def initializemotion(self, n=None):
        '''
        Re-initializes the motion array. This is motion in 2 or 3D

        - 1st Column is longitudinal motion
        - 2nd Column is latitudinal motion
        - 3rd Column is vertical motion

        Kwargs:
            * n         : If x and y are None, then n is the length of the motion 
                          array to be organized

        Returns:
            * None
        '''

        # Check
        if self.nodes is None:
            self.Nmotion = n
        else:
            self.Nmotion = len(self.nodes)

        # Intialize array
        self.motion = np.zeros((self.Nmotion,3))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def lonlat2xy(self):
        '''
        Transpose the lat/lon position of the nodes into the UTM reference.
        UTM coordinates are stored in self.x and self.y in km

        Returns:
            * None
        '''

        # do it
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def xy2lonlat(self):
        '''
        Transpose the x/y position of the nodes into lon lat
        Lon/Lat coordinates are stored in self.lon and self.lat in degrees

        Returns:
            * None
        '''

        # do it
        self.lon, self.lat = self.xy2ll(self.x, self.y)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def dropNodes(self, llbox, nx, ny, k=3):
        '''
        This will drop nodes into a region. In parallel, it will construct a
        ndspline object able to make predictions.

        Args:
            * llbox         : [lonmin, lonmax, latmin, latmax]
            * nx            : Number of nodes along longitude
            * ny            : Number of nodes along latitude

        Kwargs:
            * k             : Order of the splines (default is 3)
        
        Returns:
            * None.
        '''

        # Transfer into utm coordinates
        xmin, ymin = self.ll2xy(llbox[0], llbox[2])
        xmax, ymax = self.ll2xy(llbox[1], llbox[3])

        # Make a X and Y vectors for the position of the knots. 
        # I need a knot from lonmin to lonmax with nx knots and k extra knots on each side
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        meshx, meshy = np.meshgrid(x,y, indexing='ij')
        xknots = np.r_[(x[0],)*(k+1), x, (x[-1],)*(k+1)]
        yknots = np.r_[(y[0],)*(k+1), y, (y[-1],)*(k+1)]

        # Coefficients of this guys are therefore (len(x),len(y),1)
        self.coefficients = np.zeros((len(yknots)-k-1, len(xknots)-k-1, 1))

        # Create a NDSpline object
        self.spline = ndsplines.NDSpline([yknots,xknots], self.coefficients, np.array([k,k]))

        # Save the coefficient positions (coefficients are not the same size as the xt, yt nodes)
        x = xknots[(k+1)//2:-(k+1)//2]
        y = yknots[(k+1)//2:-(k+1)//2]
        meshx,meshy = np.meshgrid(x,y)

        # Save what needs to be saved
        self.x = meshx.flatten()
        self.y = meshy.flatten()
        self.ny,self.nx = meshx.shape
        self.xy2lonlat()

        # All nodes are active
        self.motion = np.zeros((self.coefficients[:,:,0].flatten().shape[0], 3))
        self.activeNodes = np.ones(self.coefficients[:,:,0].shape)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def selectNodes(self, active):
        '''
        This is a method to modify the activeNodes attribute.
        active is a list of indexes of the nodes that will be included in the inversion.

        Args:
            * active        : List of indexes of the nodes we want to activate.
        
        Returns:
            * None
        '''

        # Check two things
        if active[0] is tuple:
            activeNodes = np.zeros_like(self.activeNodes)
            for act in active: activeNodes[act[0], act[1]] = 1.
        elif active[0] is int:
            activeNodes = np.zeros_like(self.activeNodes.flatten())
            activeNodes[active] = 1.
            activeNodes = activeNodes.reshape(self.activeNodes.shape)

        # Save it
        self.activeNodes = activeNodes

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distanceMatrix(self, lim=None):
        '''
        Returns a matrix of the distances between patches.

        Kwargs:
            * lim       : if not None, list of two float, the first one is 
                          the distance above which d=lim[1].

        Returns:
            * distances : Array of floats
        '''

        # Check
        if self.Nmotion==None:
            self.Nmotion = self.motion.shape[0]

        # Get centers
        nodes = self.getnodes()

        # x, y, z matrices
        x = [c[0] for c in nodes]
        y = [c[1] for c in nodes]

        # Differences
        x1,x2 = np.meshgrid(x,x)
        y1,y2 = np.meshgrid(y,y)

        # Distance
        Distances = np.sqrt((x1-x2)**2 + (y1-y2)**2)

        # All done
        return Distances
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def saveGFs(self, dtype='d', outputDir='.'):
        '''
        Saves the Green's functions in different files.

        Kwargs:
            * dtype       : Format of the binary data saved. 'd' for double. 'f' for np.float32
            * outputDir   : Directory to save binary data.

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing Greens functions to file for surface motion {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # Create one file for each slip componenets
            dataname = data.replace(' ','_')
            objectname = self.name.replace(' ','_')
            filename = '{}_{}_{}.gf'.format(objectname, dataname, self.direction)
            G.astype(dtype).tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildGFs(self, data, direction='enu', vertical=True, method='bsplines', verbose=True):
        '''
        Builds the Green's function matrix following the selected interpolation method 

        The Green's function matrix is stored in a dictionary.
        Each entry of the dictionary is named after the corresponding dataset.
        Each of these entry is a dictionary that contains 'east', 'west' and 'up'

        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * direction     : A single string that is any combination of 'e', 'n' and 'u'
            * vertical      : If True, will produce green's functions for the vertical displacements
            * method        : Can only be 'bsplines' for the moment
            * verbose       : Writes stuff to the screen (overwrites self.verbose)

        Returns:
            * None

        '''

        # Check that the data type makes sense
        assert data.dtype in ('gps', 'insar', 'opticorr'), 'Data set type not supported: {}'.format(data.dtype)

        # Print
        if verbose:
            print('Surface motion prediction tool: {}'.format(method))

        # Data type check
        if data.dtype == 'insar':
            if verbose:
                print('Choice of data is InSAR hence 3D motion is required')
            direction = 'enu'
            if not vertical:
                if verbose:
                    print('---------------------------------')
                    print('---------------------------------')
                    print(' WARNING WARNING WARNING WARNING ')
                    print('  You specified vertical=False   ')
                    print(' As this is quite dangerous, we  ')
                    print(' switched it directly to True... ')
                    print(' SAR data are very sensitive to  ')
                    print('     vertical displacements.     ')
                    print(' WARNING WARNING WARNING WARNING ')
                    print('---------------------------------')
                    print('---------------------------------')
                vertical = True

        # Compute the Green's functions
        # The difference with the Fault and Pressure object is that the GFs building tool includes the 
        # handling of different data (maybe that should change for the other objects as well)
        if method in ('bsplines', 'BSplines'):
            Gin = self.bsplinesGFs(data, vertical=vertical, direction=direction, verbose=verbose)
        else:
            print('Error: Selected method is not implemented ({})'.format(method))
            raise NotImplementedError

        # Make the data object
        if data.dtype == 'insar':
            self.d[data.name] = data.vel
        elif data.dtype in ('gps', 'multigps'):
            if vertical:
                self.d[data.name] = data.vel_enu.flatten()
            else:
                self.d[data.name] = data.vel_enu[:,0:2].flatten()

        # Check
        if not hasattr(self, 'G'): self.G = {}
        self.G[data.name] = {}

        # Organize the GFs accordingly
        Gs = []
        if 'e' in direction: Gs.append(Gin['east'])
        if 'n' in direction: Gs.append(Gin['north'])
        if 'u' in direction: Gs.append(Gin['up'])
        if data.dtype in ('gps', 'multigps'):
            self.G[data.name] = scipy.linalg.block_diag(*Gs)
        elif data.dtype == 'insar':
            self.G[data.name] = np.hstack(Gs)

        # It is important to realize that Gfs are already assembled here
        # We therefore save the direction to make sure we are not forgetting
        if not hasattr(self, 'direction'): self.direction = {}
        self.direction[data.name] = direction

        # All done
        return
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    def bsplinesGFs(self, data, vertical=True, direction='enu', verbose=True):
        '''
        Builds the Green's functions for surface motion parameterized with b-splines.
        The splines are 2D and can be of any order. They are defined by the node position, which 
        should be regularily spaced so that the uplift function is an interpolating function.
        This function does not check the spacing of the spline nodes.

        There is an implementation of this in the ndsplines library but I wanted to understand
        all lines of code, so I decided to write my own (slower probably). 

        Args:
            * data          : data object (gps, insar)

        Kwargs:
            * vertical      : Do we include vertical motion?
            * direction     : Any combination of 'e', 'n' and 'u'
            * verbose       : deafult is True

        returns:
            * G             : Dictionary of Green's functions
        '''

        # Check
        if data.dtype not in ('insar', 'gps'):
            return None

        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building Green's functions for the data set ")
            print("{} of type {} for b-splines".format(data.name,
                                                       data.dtype))
        
        # Number of active nodes
        assert hasattr(self, 'activeNodes'), 'No nodes available for computation of GFs'
        nnodes = np.sum(self.activeNodes).astype(int)

        # Create the matrices to hold the whole thing
        Ge = np.zeros((len(data.x), nnodes))
        Gn = np.zeros((len(data.x), nnodes))
        Gu = np.zeros((len(data.x), nnodes))

        # Go get the data position
        x,y = data.x, data.y
        points = np.vstack((y,x)).T

        # Iterate over the active nodes
        for k,ii in enumerate(zip(np.where(self.activeNodes)[0], np.where(self.activeNodes)[1])):
            self.spline.coefficients[:,:,:] = 0.
            self.spline.coefficients[ii[0],ii[1],0] = 1.
            line = self.spline(points).squeeze()
            Ge[:,k] = line 
            Gn[:,k] = line 
            Gu[:,k] = line 
            
        # Save that in a GFs dictionnary
        G = {}
        if data.dtype=='gps':
            G['east'] = Ge
            G['north'] = Gn
            G['up'] = Gu
        elif data.dtype=='insar':
            G['east'] = data.los[:,0][:,np.newaxis]*Ge
            G['north'] = data.los[:,1][:,np.newaxis]*Gn
            G['up'] = data.los[:,2][:,np.newaxis]*Gu

        # That should do
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setGFsFromFile(self, data, strikeslip=None, dipslip=None,
                                   tensile=None, coupling=None,
                                   custom=None, vertical=False, dtype='d', 
                                   inDir='.'):
        '''
        Sets the Green's functions reading binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already). Basically, it will work better if
        you have computed the GFs using csi...

        Args:
            * data          : Data object

        kwargs:
            * strikeslip    : File containing the Green's functions for strikeslip related displacements.
            * dipslip       : File containing the Green's functions for dipslip related displacements.
            * tensile       : File containing the Green's functions for tensile related displacements.
            * coupling      : File containing the Green's functions for coupling related displacements.
            * vertical      : Deal with the UP component (gps: default is false, insar: it will be true anyway).
            * dtype         : Type of binary data. 'd' for double/float64. 'f' for np.float32

        Returns:
            * None
        '''

        print(NotImplementedError, 'Wait... This must be modified!')
        return

        # Check name conventions
        if strikeslip is None:
            if os.path.isfile(os.path.join(inDir,'{}_{}_SS.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                strikeslip = os.path.join(inDir, '{}_{}_SS.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))
        if dipslip is None:
            if os.path.isfile(os.path.join(inDir,'{}_{}_DS.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                dipslip = os.path.join(inDir, '{}_{}_DS.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))
        if tensile is None:
            if os.path.isfile(os.path.join(inDir,'{}_{}_TS.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                tensile = os.path.join(inDir, '{}_{}_TS.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))
        if coupling is None:
            if os.path.isfile(os.path.join(inDir, '{}_{}_Coupling.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                coupling = os.path.join(inDir, '{}_{}_Coupling.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))

        if self.verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Set up Green's functions for fault {}".format(self.name))
            print("and data {} from files: ".format(data.name))
            print("     strike slip: {}".format(strikeslip))
            print("     dip slip:    {}".format(dipslip))
            print("     tensile:     {}".format(tensile))
            print("     coupling:    {}".format(coupling))

        # Get the number of patches
        if self.N_slip == None:
            self.N_slip = self.slip.shape[0]

        # Read the files and reshape the GFs
        Gss = None; Gds = None; Gts = None; Gcp = None
        if strikeslip is not None:
            Gss = np.fromfile(strikeslip, dtype=dtype)
            ndl = int(Gss.shape[0]/self.N_slip)
            Gss = Gss.reshape((ndl, self.N_slip))
        if dipslip is not None:
            Gds = np.fromfile(dipslip, dtype=dtype)
            ndl = int(Gds.shape[0]/self.N_slip)
            Gds = Gds.reshape((ndl, self.N_slip))
        if tensile is not None:
            Gts = np.fromfile(tensile, dtype=dtype)
            ndl = int(Gts.shape[0]/self.N_slip)
            Gts = Gts.reshape((ndl, self.N_slip))
        if coupling is not None:
            Gcp = np.fromfile(coupling, dtype=dtype)
            ndl = int(Gcp.shape[0]/self.N_slip)
            Gcp = Gcp.reshape((ndl, self.N_slip))

        # Create the big dictionary
        G = {'strikeslip': Gss,
             'dipslip': Gds,
             'tensile': Gts,
             'coupling': Gcp}

        # The dataset sets the Green's functions itself
        data.setGFsInFault(self, G, vertical=vertical)

        # If custom
        if custom is not None:
            self.setCustomGFs(data, custom)

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembled(self, datas, verbose=True):
        '''
        Assembles a data vector for inversion using the list datas
        Assembled vector is stored in self.dassembled

        Args:
            * datas         : list of data objects

        Returns:
            * None
        '''

        # Check
        if type(datas) is not list:
            datas = [datas]

        if verbose:
            # print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling d vector")

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Create a data vector
        d = np.zeros((Nd,))

        # Loop over the datasets
        el = 0
        for data in datas:

                # print
                if verbose:
                    print("Dealing with data {}".format(data.name))

                # Get the local d
                dlocal = self.d[data.name]
                Ndlocal = dlocal.shape[0]

                # Store it in d
                d[el:el+Ndlocal] = dlocal

                # update el
                el += Ndlocal

        # Store d in self
        self.dassembled = d

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleGFs(self, datas, verbose=True):
        '''
        Assemble the Green's functions corresponding to the data in datas.
        This method allows to specify which transformation is going
        to be estimated in the data sets, through the polys argument.
        Contrary to other objects, the directions of motion have already been selected at
        the stage of GFs computation.

        Assembled Green's function matrix is stored in self.Gassembled

        Args:
            * datas : list of data sets. If only one data set is used, can be a data instance only.

        Kwargs:
            * verbose   : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check
        if type(datas) is not list:
            datas = [datas]

        # Make sure directions are similar
        for data in datas:
            assert self.direction[data.name] == self.direction[datas[0].name], \
                    'Data sets have different directions of motion'

        # print
        if verbose:
            print("---------------------------------")
            print("---------------------------------")
            print("Assembling G for surface motion {}".format(self.name))

        # Get the number of parameters
        if self.Nmotion == None: self.Nmotion = self.motion.shape[0]
        Np = self.Nmotion*len(self.direction[datas[0].name])

        # Get the number of data
        Nd = np.sum([len(self.d[data.name]) for data in datas])

        # Allocate G and d
        G = np.zeros((Nd, Np))

        # Create the list of data names, to keep track of it
        self.datanames = []

        # loop over the datasets
        el = 0
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            if verbose:
                print("Dealing with {} of type {}".format(data.name, data.dtype))

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Glocal = self.G[data.name]

            # Put Glocal into the big G
            G[el:el+Ndlocal,0:Np] = Glocal

            # Update el to check where we are
            el = el + Ndlocal

        # Store G in self
        self.Gassembled = G

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleCd(self, datas, add_prediction=None, verbose=False):
        '''
        Assembles the data covariance matrices that have been built for each
        data structure.

        Args:
            * datas         : List of data instances or one data instance

        Kwargs:
            * add_prediction: Precentage of displacement to add to the Cd diagonal to simulate a Cp (dirty version of a prediction error covariance, see Duputel et al 2013, GJI).
            * verbose       : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check if the Green's function are ready
        assert self.Gassembled is not None, \
                "You should assemble the Green's function matrix first"

        # Check
        if type(datas) is not list:
            datas = [datas]

        # Get the total number of data
        Nd = self.Gassembled.shape[0]
        Cd = np.zeros((Nd, Nd))

        # Loop over the data sets
        st = 0
        for data in datas:
            # Fill in Cd
            if verbose:
                print("{0:s}: data vector shape {1:s}".format(data.name, self.d[data.name].shape))
            se = st + self.d[data.name].shape[0]
            Cd[st:se, st:se] = data.Cd
            # Add some Cp if asked
            if add_prediction is not None:
                Cd[st:se, st:se] += np.diag((self.d[data.name]*add_prediction/100.)**2)
            st += self.d[data.name].shape[0]

        # Store Cd in self
        self.Cd = Cd

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCmGaussian(self, sigma):
        '''
        Builds a diagonal Cm with sigma values on the diagonal.
        Sigma is a list of numbers, as long as you have components of motion (1, 2 or 3).

        Model covariance is hold in self.Cm

        Args:
            * sigma         : List of numbers the size of the slip components requried for the modeling

        Returns:
            * None
        '''

        # Make sure directions are all the same
        directions = [self.direction[data] for data in self.direction]
        assert all(item == directions[0] for item in directions), 'Data sets have different directions of motion'

        # Get the number of slip directions
        direction = len(directions[0])
        if self.Nmotion==None: self.Nmotion = self.motion.shape[0]

        # Number of parameters
        Np = self.Nmotion * direction

        # Create Cm
        Cm = np.zeros((Np, Np))

        # Loop over slip dir
        for i in range(direction):
            Cmt = np.diag(sigma[i] * np.ones(self.Nmotion,))
            Cm[i*self.Nmotion:(i+1)*self.Nmotion,i*self.Nmotion:(i+1)*self.Nmotion] = Cmt

        # Stores Cm
        self.Cm = Cm

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCm(self, sigma, lam, lam0=None, lim=None, verbose=True):
        '''
        Builds a model covariance matrix using the equation described in
        Radiguet et al 2010. We use

        :math:`C_m(i,j) = \\frac{\sigma \lambda_0}{ \lambda }^2 e^{-\\frac{||i,j||_2}{ \lambda }}`

        Model covariance is stored in self.Cm

        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.

        Kwargs:
            * lam0          : Normalizing distance. if None, lam0=min(distance between patches)
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
            * verbose       : Talk to me (overwrites self.verrbose)

        Returns:
            * None
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the Cm matrix ")
            print ("Sigma = {}".format(sigma))
            print ("Lambda = {}".format(lam))

        # Make sure directions are all the same
        directions = [self.direction[data] for data in self.direction]
        assert all(item == directions[0] for item in directions), 'Data sets have different directions of motion'

        # Get the number of slip directions
        direction = len(directions[0])

        # Get the node positions
        self.nodes = self.getnodes()

        # Sets the lambda0 value
        if lam0 is None:
            xd = ((np.unique(self.nodes[:,0]).max() - np.unique(self.nodes[:,0]).min())
                / (np.unique(self.nodes[:,0]).size))
            yd = ((np.unique(self.nodes[:,1]).max() - np.unique(self.nodes[:,1]).min())
                / (np.unique(self.nodes[:,1]).size))
            lam0 = np.sqrt(xd**2 + yd**2)
        if verbose:
            print("Lambda0 = {}".format(lam0))
        C = (sigma * lam0 / lam)**2

        # Creates the principal Cm matrix
        if self.Nmotion==None:
            self.Nmotion = self.motion.shape[0]
        Np = self.Nmotion * direction
        Cm = np.zeros((Np, Np))

        # Loop over the patches
        distances = self.distanceMatrix(lim=lim)
        Cmt = C * np.exp(-distances / lam)

        # Store that into Cm
        st = 0
        for i in range(direction):
            se = st + self.Nmotion
            Cm[st:se, st:se] = Cmt
            st += self.Nmotion

        # Store Cm into self
        self.Cm = Cm

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # pickle myself
    def pickle(self, filename):
        '''
        Pickle myself.

        :Args:
            * filename      : Name of the pickle fault.
        '''

        # Open the file
        fout = open(filename, 'wb')

        # Pickle
        pickle.dump(self, fout)

        # Close 
        fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

#EOF
