'''
A class that deals with transformations

Written by R. Jolivet, Dec 2017
'''

# Externals
import glob
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import itertools
import copy
import sys
import os

# Personals
from .SourceInv import SourceInv

#class transformation
class transformation(SourceInv):

    # ----------------------------------------------------------------------
    # Initialize class 
    def __init__(self, name, utmzone=None, ellps='WGS84', 
                             lon0=None, lat0=None, verbose=True):
        '''
        Args:
            * name          : Name of the object
            * utmzone       : UTM zone  (optional, default=None)
            * lon0/lat0     : Center of the custom UTM zone
            * ellps         : ellipsoid (optional, default='WGS84')
            * verbose       : talk to me
        '''

        super(transformation,self).__init__(name,
                                            utmzone = utmzone,
                                            ellps = ellps, 
                                            lon0 = lon0, 
                                            lat0 = lat0)
        # Initialize the class
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing transformation {}".format(self.name))
        self.verbose = verbose

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}
        self.m = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # Something important for consistency
        self.patchType = 'transformation'
        self.type = 'transformation'
        self.slipdir = ''

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Build the Green's functions for the transformations
    def buildGFs(self, datas, transformations, verbose=True, computeNormFact=True):
        '''
        Builds the design matrix for the datasets given. 

        The GFs are stored in a dictionary. 
        Each entry of the dictionary is named after the corresponding dataset. 
        Each of these entry is a dictionary that contains the different cases 
        of transformations.

        Args:   
            * datas             : List of datasets (gps, insar, optical, ...)
            * transformations   : List of transformation types

        Kwargs:
            * verbose           : Talk to me
            * computeNormFact   : Compute the Normalization factors or not

        Returns:
            * None

            Transformation types can be:
                    
                For InSAR, Optical:
                    'strain':Estimates a strain tensor
                    string of polynomial components to include in the estimator. Possible values are: '1', 'x', 'y', 'xy', 'x2', 'y2'. Example: '1,x,y' 
                
                For optical correlation:
                    string of polynomial components to include in the estimator. Possible values are: '1', 'x', 'y', 'xy', 'x2', 'y2'. Example: '1,x,y'
             
                For GPS:
                    'strain': Estimates a strain tensor
                    'full': Estimates a full Helmert
                    'translation': Estimates a translation
                    'rotation': Estimates a rotation

        '''

        # Check something
        if type(datas) is not list:
            datas = [datas]

        # Pre compute Normalizing factors
        if computeNormFact:
            self.computeNormFactors(datas)

        # Save
        if not hasattr(self, 'transformations'):
            self.transformations = {}

        # Iterate over the data
        for data, transformation in zip(datas, transformations):
            
            # Check something
            assert data.dtype in ('insar', 'gps', 'tsunami', 'multigps', 'opticorr', 'surfaceslip'), \
                    'Unknown data type {}'.format(data.dtype)
            
            # Print
            if verbose:
                print('---------------------------------')
                print('---------------------------------')
                print("Building transformation Green's functions")
                print("for the data set {} of type {}".format(data.name, data.dtype))

            # Check the GFs
            if data.name not in self.G.keys(): self.G[data.name] = {}

            # Save
            self.transformations[data.name] = transformation

            # Check iterations
            if type(transformation) is not list:
                transformation = [transformation]

            for trans in transformation:
                G = data.getTransformEstimator(trans, computeNormFact=False)
                # One case is tricky so we build strings
                if type(trans) is list:
                    trans = ''.join(itertools.chain.from_iterable(trans))
                self.G[data.name][trans] = G

            # Set data in the GFs
            if data.dtype in ('insar', 'surfaceslip'):
                self.d[data.name] = data.vel
            elif data.dtype == 'tsunami':
                self.d[data.name] = data.d
            elif data.dtype in ('gps', 'multigps'):
                locdat = []
                for i in range(3):
                    if not np.isnan(data.vel_enu[:,i]).any():
                        locdat.append(data.vel_enu[:,i])
                    self.d[data.name] = np.array(locdat).flatten()
            elif data.dtype == 'opticorr':
                self.d[data.name] = np.hstack((data.east.T.flatten(),
                                               data.north.T.flatten()))

        # Consistency
        self.poly = self.transformations

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Compute the Normalization factors
    def computeNormFactors(self, datas):
        '''
        Sets a common reference for the computation of the transformations

        Args:
            * datas         : list of data

        Returns:
            * None
        '''

        # Initialize
        x, y, refx, base, refy = 0., 0., 0., 0., 0. 

        # Iterate
        for data in datas:
            self.computeTransformNormFactor(data)
            x += data.TransformNormalizingFactor['x']
            y += data.TransformNormalizingFactor['y']
            refx += data.TransformNormalizingFactor['ref'][0]
            refy += data.TransformNormalizingFactor['ref'][1]
            base += data.TransformNormalizingFactor['base']

        # Average
        x /= len(datas)
        y /= len(datas)
        refx /= len(datas)
        refy /= len(datas)
        base /= len(datas)

        # Set
        for data in datas:
            data.TransformNormalizingFactor['x'] = x
            data.TransformNormalizingFactor['y'] = y
            data.TransformNormalizingFactor['ref'] = [refx, refy]
            data.TransformNormalizingFactor['base'] = base

        # All done
        return 
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Compute the Normalizing factors
    def computeTransformNormFactor(self, data):
        '''
        Computes quantities needed to build the transformation object for 
        a dataset

        Args:
            * data          : instance of a data class
        '''

        # Calculate
        x0 = np.mean(data.x)
        y0 = np.mean(data.y)
        base_x = data.x - x0
        base_y = data.y - y0
        normX = np.abs(base_x).max()
        normY = np.abs(base_y).max()
        base_max = np.max([np.abs(base_x).max(), np.abs(base_y).max()])

        # Set in place
        data.TransformNormalizingFactor = {}
        data.TransformNormalizingFactor['x'] = normX
        data.TransformNormalizingFactor['y'] = normY
        data.TransformNormalizingFactor['ref'] = [x0, y0]
        data.TransformNormalizingFactor['base'] = base_max

        # Special case of a multigps dataset
        if data.dtype=='multigps':
            for d in data.gpsobjects:
                self.computeTransformNormFactor(d)

        # All done
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
            print ("Assembling d for transformation {}".format(self.name))

        # Create a data vector
        d = []

        # Loop over the datasets
        for data in datas:

            # print
            if verbose:
                print("Dealing with data {} of type {}".format(data.name, data.dtype))

            # Get the local d
            dlocal = self.d[data.name].tolist()

            # Store it in d
            d += dlocal

        # Store d in self
        self.dassembled = np.array(d)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleCd(self, datas, add_prediction=None, verbose=True):
        '''
        Assembles the data covariance matrices that have been built for each 
        data structure.

        Args:
            * datas         : List of data instances or one data instance

        Kwargs:
            * add_prediction: Precentage of displacement to add to the Cd 
                              diagonal to simulate a Cp (dirty version of 
                              a prediction error covariance, see Duputel et
                              al 2013, GJI).
            * verbose       : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check if the Green's function are ready
        assert self.Gassembled is not None, \
                "You should assemble the Green's function matrix first"
        
        if verbose:
            # print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling Cd for transformation {}".format(self.name))

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
                print("Dealing with data {} of type {}".format(data.name, data.dtype))
                
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
    # Assemble the Green's functions
    def assembleGFs(self, datas, verbose=True):
        '''
        Assemble the Green's functions corresponding to the data in datas.
        Assembled Greens' functions are stored in self.Gassembled

        Special case: If 'strain' is in self.transformations, this parameter will 
        be placed as first and will be common to all data sets (i.e. there is
        only one strain tensor for a region, although there can be multiple 
        translation, rotations, etc for individual networks)

        Args:   
            * datas         : list of data objects

        Returns:
            * None
        '''

        # Check 
        if type(datas) is not list:
            datas = [datas]

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print("Assembling G for transformation {}".format(self.name))

        # Checker
        strainCase = False

        # Sizes
        Nd = 0; Np = 0; dindex = {}
        for dname in self.G:

            # Parameters
            Nplocal = 0
            for trans in self.G[dname]:
                Ndlocal = self.d[dname].shape[0]
                if trans is not None:
                    Nplocal += self.G[dname][trans].shape[1]

            # Strain case
            if 'strain' in self.G[dname]:
                Nplocal -= 3
                strainCase = True
            Np += Nplocal

            # Data
            if Nplocal > 0:
                assert all([self.G[dname][trans].shape[0]==Ndlocal \
                            for trans in self.G[dname]]),\
                            'GFs size issue for data set {}: {} vs {}'.format(dname, 
                                    self.G[dname][trans].shape[0], 
                                    Ndlocal)
            dindex[dname] = (Nd, Nd+Ndlocal)
            Nd += Ndlocal

        # initialize counters
        if strainCase: 
            Np += 3
            Npl = 3
        else:
            Npl = 0
        Ndl = 0

        # Create G
        G = np.zeros((Nd, Np))

        # Keep track of the transform orders
        self.transOrder = []
        self.transIndices = []
        if strainCase: 
            self.transOrder.append('strain')
            self.transIndices.append((0,3))

        # Keep track of data names
        self.datanames = []

        # Iterate over the data and transforms
        for data in datas:
            
            # print
            if verbose:
                print("Dealing with {} of type {}".format(data.name, data.dtype))
            
            dname = data.name
            self.datanames.append(data.name)
            # Which transform do we care about
            transformations = self.transformations[dname]
            # Which lines do we care about
            Nds, Nde = dindex[dname]
            for trans in self.G[dname]:
                # Get G
                Glocal = self.G[dname][trans]
                # Strain case
                if trans == 'strain':
                    G[Nds:Nde,:3] = Glocal
                elif trans is None:
                    self.transOrder.append('{} --//-- {}'.format(dname, trans))
                    self.transIndices.append(None)
                else:
                    Npe = Npl + Glocal.shape[1]
                    G[Nds:Nde,Npl:Npe] = Glocal
                    self.transOrder.append('{} --//-- {}'.format(dname, trans))
                    self.transIndices.append((Npl,Npe))
                    Npl = Npe

        # all done
        self.Gassembled = G
        self.TransformationParameters = G.shape[1]

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Build Cm
    def buildCm(self, sigma, verbose=True):
        '''
        Builds a model covariance matrix from std deviation values.
        The matrix is diagonal with sigma**2 values.
        Requires an assembled Green's function matrix.

        Args:
            * sigma         : float, list or array
        
        Kwargs:
            * verbose       : Talk to me
        '''

        # Check
        assert hasattr(self, 'Gassembled'), 'Assemble Greens functions first'
        
        # Talk to me
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print (f"Assembling the Cm matrix for transformation {self.name}")
            print (f"sigma = {sigma}")

        # Get numbers
        Np = self.Gassembled.shape[1]

        # Create 
        if type(sigma) is float:
            self.Cm = np.diag(np.ones((Np,))*sigma)
        else:
            self.Cm = np.diag(np.array(sigma))

        # Check
        assert self.Cm.shape[0]==Np, \
                "Something's wrong with the shape of Cm: {}".format(self.Cm.shape)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Save Cm to file
    def writeCm2File(self, dtype='d', outputDir='.'):
        '''
        Write the model a priori covariance matrix to a binary file.

        Args:
            * filename      : Name of the file.
        
        Kwargs:
            * dtype         : Data type to use. Default is double ('d'). Can be 'f' for float32.
            * outputDir     : Directory to write the file in.

        Returns:
            * None
        '''

        # Check that Cm exists
        assert hasattr(self, 'Cm'), "No Cm matrix to write"

        # Write to file
        filename = f"{self.name.replace(' ', '_')}.cm"
        self.Cm.astype(dtype).tofile(os.path.join(outputDir, filename))
        print('Writing Cm matrix to file {}'.format(filename))

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Read Cm from file
    def setCmFromFile(self, filename=None, dtype='d', inDir='.'):
        '''
        Read the model a priori covariance matrix from a binary file.

        Args:
            * filename      : Name of the file.
            
        Kwargs:
            * dtype         : Data type to use. Default is double ('d'). Can be 'f' for float32.
            * inDir         : Directory to read the file from.
        
        Returns:
            * None
        '''
        
        # Check name conventions
        if filename is None:
            if os.path.isfile(os.path.join(inDir, f'{self.name.replace(" ","_")}.cm')):
                filename = os.path.join(inDir, f'{self.name.replace(" ","_")}.cm')

        # Read the files and reshape the Cm
        Cm = np.fromfile(filename, dtype=dtype)
        n = int(np.sqrt(Cm.size))
        Cm = Cm.reshape((n, n))
        
        # Store Cm
        self.Cm = Cm
        print('Reading Cm matrix from file {}'.format(filename))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Remove synthetics
    def removePredictions(self, datas, verbose=True):
        '''
        Given a list of data, predicts the surface displacements from what
        is stored in the self.m dictionary and corrects the data

        Args:
            * datas         : list of data instances

        Kwargs:
            * verbose       : Talk to me
        '''

        # Check something
        if type(datas) is not list:
            datas = [datas]

        # remove
        for data in datas:
            data.removeTransformation(self)
            
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # distribute mpost to self.m following what is in self.Gassembled
    def distributem(self):
        '''
        Uses self.mpost to distribute the values to self.m following the 
        organization of self.Gassembled.

        Args:
            * None

        Returns:
            * None
        '''

        # Check something
        assert self.mpost.shape[0]==self.Gassembled.shape[1],\
                'Wrong size for mpost: {}. Should be {}'.\
                format(self.mpost.shape[0],self.Gassembled.shape[1])

        # Check
        start = 0

        # Check strain case
        if self.transOrder[0]=='strain':
            start += 1
            index = self.transIndices[0]
            for dname in self.G:
                if dname not in self.m:
                    self.m[dname] = {}
                self.m[dname]['strain'] = self.mpost[index[0]:index[1]]


        # Iterate over transOrder
        for datatrans, index in zip(self.transOrder[start:], self.transIndices[start:]):

            # Get names
            dname, trans = datatrans.split(' --//-- ')

            # Convert to int if possible
            # try:
            #     trans = int(trans)
            # except:
            #     trans = trans

            # Check
            if dname not in self.m:
                self.m[dname] = {}

            if index is not None:
                self.m[dname][trans] = self.mpost[index[0]:index[1]]
            else:
                self.m[dname][trans] = None

        # Consistency
        self.polysol = self.m

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    def saveGFs(self, dtype='d', outputDir='.'):
        '''
        Saves the Green's functions in different files.

        Kwargs:
            * dtype       : Format of the binary data saved. 'd' for double. 'f' for np.float32
            * outputDir   : Directory to save binary data.
            * suffix      : suffix for GFs name (dictionary)

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing Greens functions to file for transformation {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # Create one file for each type of transformation
            for t in G.keys():
                if t is not None:
                    g = G[t].flatten()
                    n = self.name.replace(' ', '_')
                    d = data.replace(' ', '_')
                    filename = '{}_{}_trans_{}.gf'.format(n, d, str(t))
                    g = g.astype(dtype)
                    g.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------
    
     # ----------------------------------------------------------------------
    # Set the Green's functions from file
    def setGFsFromFile(self, data, transformation=None, custom=None,
                       vertical=False, dtype='d', inDir='.'):
        '''
        Sets the Green's functions reading binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already). Basically, it will work better if
        you have computed the GFs using csi...

        Args:
            * data          : Data object

        kwargs:
            * transformation : File or list of files containing the Green's functions for geometrical transformation(s).
            * vertical       : Deal with the UP component (gps: default is false, insar: it will be true anyway).
            * dtype          : Type of binary data. 'd' for double/float64. 'f' for np.float32

        Returns:
            * None
        '''
        
        # Initialize the dictionaries of transformations
        if not hasattr(self, 'transformations'):
            self.transformations = {}
        
        # Initialize the dictionary for the data set
        if not hasattr(self.G, data.name):
            self.G[data.name] = {}
            
        # Set data in the GFs
        if data.dtype in ('insar', 'surfaceslip'):
            self.d[data.name] = data.vel
        elif data.dtype == 'tsunami':
            self.d[data.name] = data.d
        elif data.dtype in ('gps', 'multigps'):
            locdat = []
            for i in range(3):
                if not np.isnan(data.vel_enu[:,i]).any():
                    locdat.append(data.vel_enu[:,i])
                self.d[data.name] = np.array(locdat).flatten()
        elif data.dtype == 'opticorr':
            self.d[data.name] = np.hstack((data.east.T.flatten(),
                                            data.north.T.flatten()))

        # Check filenames
        if transformation is None:
            
            list_fname_transfo = glob.glob(os.path.join(inDir, f"{self.name.replace(' ','_')}_{data.name.replace(' ','_')}_trans_*.gf"))
            
            if len(list_fname_transfo) == 0:
                transformation = None
            else:
                transformation = list_fname_transfo
        
        # Load the transformation Green's functions
        if transformation is not None:
            
            for trans_ in transformation:
                
                # Name of the transformation
                trans_name = trans_.split('trans_')[1].split('.gf')[0]
                try:
                    trans_name = int(trans_name)
                except:
                    pass

                # Talk to me
                if self.verbose:
                    print('---------------------------------')
                    print('---------------------------------')
                    print("Set up Green's functions for transformation {}".format(self.name))
                    print("and data {} from files: ".format(data.name))
                    print("     transformation {}: {}".format(trans_name, trans_))
                
                # Read
                Gtrans = np.fromfile(trans_, dtype=dtype)
                ndl = len(self.d[data.name])
                Gtrans = Gtrans.reshape(ndl, int(len(Gtrans)/ndl))

                # Create the dictionary
                self.G[data.name][str(trans_name)] = Gtrans
                
        else:
            
            # Name of the transformation
            trans_name = None
            
            # Talk to me
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print("Set up Green's functions for transformation {}".format(self.name))
                print("and data {} : ".format(data.name))
                print("     transformation: {}".format(None))
            
                # Create the dictionary
                self.G[data.name][trans_name] = None
            
        # Save
        self.transformations[data.name] = trans_name

        # Consistency
        self.poly = self.transformations
        
        # all done
        return
    # ----------------------------------------------------------------------

#EOF
