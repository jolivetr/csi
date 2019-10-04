'''
A class that deals with transformations

Written by R. Jolivet, Dec 2017
'''

# Externals
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

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}
        self.m = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # Something important
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
                    
                 For InSAR, Optical, GPS:
                       1 -> estimate a constant offset
                       3 -> estimate z = ax + by + c
                       4 -> estimate z = axy + bx + cy + d
                       'strain'              -> Estimates a strain tensor 
             
                 For GPS only:
                       'full'                -> Estimates a rotation, 
                                                translation and scaling
                                                (Helmert transform).
                       'translation'         -> Estimates a translation
                       'rotation'            -> Estimates a rotation

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
            assert data.dtype in ('insar', 'gps', 'tsunami', 'multigps', 'opticorr'), \
                    'Unknown data type {}'.format(data.dtype)

            # Check the GFs
            if data.name not in self.G.keys(): self.G[data.name] = {}

            # Save
            self.transformations[data.name] = transformation

            # A case that will need to change in the future
            if data.dtype in ('gps') and transformation=='strain':
                T = data.getTransformEstimator('strainonly', computeNormFact=False)
            else:
                T = data.getTransformEstimator(transformation, computeNormFact=False)

            # One case is tricky so we build strings
            trans = '{}'.format(transformation)
            self.G[data.name][trans] = T

            # Set data in the GFs
            if data.dtype == 'insar':
                self.d[data.name] = data.vel
            elif data.dtype == 'tsunami':
                self.d[data.name] = data.d
            elif data.dtype in ('gps', 'multigps'):
                if not np.isnan(data.vel_enu[:,2]).any():
                    self.d[data.name] = data.vel_enu.T.flatten()
                else:
                    self.d[data.name] = data.vel_enu[:,0:2].T.flatten()
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
        if data.dtype is 'multigps':
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
            print ("Assembling d vector")

        # Create a data vector
        d = []

        # Loop over the datasets
        for data in datas:

                # print
                if verbose:
                    print("Dealing with data {}".format(data.name))

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
    def assembleCd(self, datas, add_prediction=None, verbose=False):
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
                print("{0:s}: data vector shape {1:s}"\
                        .format(data.name, self.d[data.name].shape))
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
                            'GFs size issue for data set {}'.format(dname)
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
                    self.transIndices.append((Npl, Npe))
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
    def buildCm(self, sigma):
        '''
        Builds a model covariance matrix from std deviation values.
        The matrix is diagonal with sigma**2 values.
        Requires an assembled Green's function matrix.

        Args:
            * sigma         : float, list or array
        '''

        # Check
        assert hasattr(self, 'Gassembled'), 'Assemble Greens functions first'

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
            data.removeTransformation(self, verbose=verbose)
            
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
            try:
                trans = int(trans)
            except:
                trans = trans

            # Check
            if dname not in self.m:
                self.m[dname] = {}
            self.m[dname][trans] = self.mpost[index[0]:index[1]]

        # Consistency
        self.polysol = self.m

        # All done
        return
    # ----------------------------------------------------------------------

#EOF
