'''
A class the allows to compute various things using a fault object.

Written by R. Jolivet, Z. Duputel and B. Riel, April 2013
'''

import numpy as np
import pyproj as pp
import copy
import matplotlib.pyplot as plt
import sys
import os

# Locals
from . import csiutils as utils

class faultpostproctents(object):

    def __init__(self, name, fault, Mu=24e9, samplesh5=None, verbose=True):
        '''
        Args:
            * name          : Name of the InSAR dataset.
            * fault         : Fault object
            * Mu            : Shear modulus. Default is 24e9 GPa, because it is the PREM value for the upper 15km. Can be a scalar or a list/array of len=len(fault.patch)
            * samplesh5     : file name of h5 file containing samples
            * verbose       : Verbose (True/False)
        '''

        # Initialize the data set 
        self.name = name
        self.fault = copy.deepcopy(fault) # we don't want to modify fault slip
        self.utmzone = fault.utmzone
        self.sourceDepths = None
        self.numNodes = len(fault.tent)
 
        # Create the interpolating sources
        if not hasattr(self.fault, 'sourceNumber'):
            self.fault.sourceNumber = npoints
        if not hasattr(self.fault, 'plotSources'):
            from .EDKSmp import dropSourcesInPatches as Patches2Sources
            Ids, xs, ys, zs, strike, dip, Areas = Patches2Sources(self.fault, verbose=False)  
            self.fault.plotSources = [Ids, xs, ys, zs, strike, dip, Areas] 

        # Get the interpolating sources
        Ids, xs, ys, zs, strike, dip, Areas = self.fault.plotSources
        Lon, Lat = self.fault.xy2ll(xs, ys)
        X, Y, Z = xs, ys, zs

        # Store what is needed (one source, not three)
        self.lon = Lon
        self.lat = Lat
        self.x = X
        self.y = Y
        self.depth = Z
        self.areas = Areas
        self.strike = strike
        self.dip = dip
        self.ids = Ids

        # Get number of Nodes
        self.numSources = len(self.ids)

        # Create the slip vector
        self.setSlipToSources()

        # Assign Mu to each node
        if len(np.array(Mu).flatten())==1:
            self.Mu = Mu * np.ones((self.numSources,))
        else:
            assert len(Mu)==self.numSources, 'length of Mu must be 1 or numPatch'
            self.Mu = np.array(Mu)
            
        # Display
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initialize Post Processing object {} on fault {}".format(self.name, fault.name))

        # Check to see if we're reading in an h5 file for posterior samples
        self.samplesh5 = samplesh5

        # All done
        return

    def h5_init(self, decim=1,indss=None,indds=None):
        '''
        If the attribute self.samplesh5 is not None, we open the h5 file specified by 
        self.samplesh5 and copy the slip values to self.fault.slip (hopefully without loading 
        into memory).

        kwargs:
            decim                       decimation factor for skipping samples
            indss :  tuples (size (2,)) containing desired indices of strike slip in h5File
            indds :  tuples (size (2,)) containing desired indices of dip slip in h5File
        '''

        if self.samplesh5 is None:
            return
        else:
            try:
                import h5py
            except ImportError:
                print('Cannot import h5py. Computing scalar moments only')
                return
            self.hfid = h5py.File(self.samplesh5, 'r')
            samples = self.hfid['Sample Set']
            nsamples = np.arange(0, samples.shape[0], decim).size
            self.fault.slip = np.zeros((self.numNodes,3,nsamples))
            self.fault.slip[:,0,:] = samples[::decim,:self.numNodes].T
            self.fault.slip[:,1,:] = samples[::decim,self.numNodes:2*self.numNodes].T

            if indss is None or indds is None:
                nsamples = np.arange(0, samples.shape[0], decim).size
                self.fault.slip = np.zeros((self.numNodes,3,nsamples))
                self.fault.slip[:,0,:] = samples[::decim,:self.numNodes].T
                self.fault.slip[:,1,:] = samples[::decim,self.numNodes:2*self.numNodes].T

            else:
                assert indss[1]-indss[0] == self.numNodes, 'indss[1] - indss[0] different from number of patches'
                assert indss[1]-indss[0] == self.numNodes, 'indds[1] - indds[0] different from number of patches'
                nsamples =  np.arange(0, samples.shape[0], decim).size
                self.fault.slip = np.zeros((self.numNodes,3,nsamples))
                self.fault.slip[:,0,:] = samples[::decim,indss[0]:indss[1]].T
                self.fault.slip[:,1,:] = samples[::decim,indds[0]:indds[1]].T


            self.h5 = True # set flag for the rest of the process

        return

    def setMuDepth(self, depthMu):
        '''
        Assign values of Mu as a function of depth.

        Args:
            * depthMu       : List of increasing (depth,mu) tuple.
                depthMu = [(3, 1e9), (5, 3e9), (10, 5e9), (0, 1e10)]
                The last value, with depth 0 means everything under the last-1 depth.
        '''

        # Lists
        depthup, depthdown = [], []
        mu = []

        # Set depths
        dup = 0
        for dm in depthMu:
            ddown = dm[0]
            m = dm[1]
            depthup.append(dup)
            depthdown.append(ddown)
            mu.append(m)
            dup = ddown

        # Arraying
        depthup = np.array(depthup)
        depthdown = np.array(depthdown)
        mu = np.array(mu)

        # Iterate over subsources
        self.Mu = []
        for z in self.depth:
            u = np.flatnonzero(np.logical_and((z>=depthup), (z<depthdown)))
            assert (len(u)==1), 'More than one value or no value found'
            self.Mu.append(mu[u])
    
        self.Mu = np.array(self.Mu).squeeze()

        # All done
        return

    def setSlipToSources(self):
        '''
        Takes the slip vector from a fault and organize the slip vector of the postprocessing tool.
        '''

        # Get values
        X = self.x
        Y = self.y
        Z = self.depth
        Ids = self.ids

        # Get the corresponding Slip distribution
        slip = []
        for i in range(3):
            slip.append(self.fault._getSlipOnSubSources(Ids, X, Y, Z, self.fault.slip[:,i]))
        self.slip = np.array(slip).T

        # All done
        return

    def h5_finalize(self):
        '''
        Close the (potentially) open h5 file.
        '''
        if hasattr(self, 'hfid'):
            self.hfid.close()

        return
            
    def ll2xy(self, lon, lat):
        '''
        Uses the transformation in self to convert  lon/lat vector to x/y utm.
        Args:
            * lon           : Longitude array.
            * lat           : Latitude array.
        '''
    
        return self.fault.ll2xy(lon, lat)

    def xy2ll(self, x, y):
        '''
        Uses the transformation in self to convert x.y vectors to lon/lat.
        Args:
            * x             : Xarray
            * y             : Yarray
        '''

        return self.fault.xy2ll(x, y)

    def sourceNormal(self):
        '''
        Returns the Normal of the subsources
        '''

        # Get the geometry of the subSources
        strike, dip = self.strike, self.dip

        # Normal
        n1 = -1.0*np.sin(dip)*np.sin(strike)
        n2 = np.sin(dip)*np.cos(strike)
        n3 = -1.0*np.cos(dip)
        N = np.sqrt(n1**2+ n2**2 + n3**2)

        # All done
        return np.array([n1/N, n2/N, n3/N]).T

    def slipVector(self):
        '''
        Returns the slip vector in the cartesian space for all subsources. We do not deal with 
        the opening component. The fault slip may be a 3D array for multiple samples of slip.
        '''

        # Get the geometry of the patch
        strike, dip = self.strike, self.dip

        # Get the slip
        strikeslip, dipslip = self.slip[:,0,...], self.slip[:,1,...]
        slip = np.sqrt(strikeslip**2 + dipslip**2)

        # Get the rake
        rake = np.arctan2(dipslip, strikeslip)

        # Vectors
        ux = slip*(np.cos(rake)*np.cos(strike) + np.cos(dip)*np.sin(rake)*np.sin(strike))
        uy = slip*(np.cos(rake)*np.sin(strike) - np.cos(dip)*np.sin(rake)*np.cos(strike))
        uz = -1.0*slip*np.sin(rake)*np.sin(dip)

        # All done
        if ux.ndim==2:
            outArr = np.zeros((3,self.nSamples,ux.size))
            outArr[0,:,:] = ux
            outArr[1,:,:] = uy
            outArr[2,:,:] = uz
            return outArr
        else:
            return np.array([ux, uy, uz]).T

    def computeSourcesMoments(self) :
        '''
        Computes the Moment tensors for all subsources.
        '''

        # Get the normal
        normals = self.sourceNormal()

        # Get the slip vector
        slip = self.slipVector()

        # Compute the moment density
        if slip.ndim == 2:
            p1 = np.array([np.dot(slip[i,:].reshape((3,1)), normals[i,:].reshape((1,3))) for i in range(slip.shape[0])])
            p2 = np.array([np.dot(normals[i,:].reshape((3,1)), slip[i,:].reshape((1,3))) for i in range(slip.shape[0])])
            mt = self.Mu[:,None,None] * (p1 + p2) 
        elif slip.ndim == 3:
            assert False, 'Not implemented yet'
            # Careful about tiling - result is already transposed
            #nT = np.tile(n, (1,1,slip.shape[2]))
            #n = np.transpose(nT, (1,0,2))
            #uT = np.transpose(u, (1,0,2))
            ## Tricky 3D multiplication
            #mt = self.Mu[p] * ((u[:,:,None]*nT).sum(axis=1) + (n[:,:,None]*uT).sum(axis=1))

        # Multiply by the area 
        mt *= self.areas[:,None,None]*1e6

        # Save it
        self.Moments = mt

        # All done
        return 

    def computeMomentTensor(self):
        '''
        Computes the full seismic (0-order) moment tensor from the slip distribution.
        '''


        # Compute the tensor for each subsource
        self.computeSourcesMoments()
        
        # Sum
        M = self.Moments.sum(axis=0)

        # Check if symmetric
        self.checkSymmetric(M)

        # Store it (Aki convention)
        self.Maki = M

        # Convert it to Harvard
        self.Aki2Harvard()

        # All done
        return

    def computeScalarMoment(self):
        '''
        Computes the scalar seismic moment.
        '''

        # check 
        assert hasattr(self, 'Maki'), 'Compute the Moment Tensor first'

        # Get the moment tensor
        M = self.Maki

        # get the norm
        Mo = np.sqrt(0.5 * np.sum(M**2, axis=(0,1)))

        # Store it
        self.Mo = Mo

        # All done
        return Mo

    def computeMagnitude(self, plotHist=None, outputSamp=None):
        '''
        Computes the moment magnitude.
        '''

        # check
        if not hasattr(self, 'Mo'):
            self.computeScalarMoment()

        # Mw
        Mw = 2./3.*(np.log10(self.Mo) - 9.1)

        # Store 
        self.Mw = Mw

        # Plot histogram of magnitudes
        if plotHist is not None:
            assert False, 'Not implemented yet'
            assert isinstance(Mw, np.ndarray), 'cannot make histogram with one value'
            fig = plt.figure(figsize=(14,8))
            ax = fig.add_subplot(111)
            ax.hist(Mw, bins=100)
            ax.grid(True)
            ax.set_xlabel('Moment magnitude', fontsize=18)
            ax.set_ylabel('Normalized count', fontsize=18)
            ax.tick_params(labelsize=18)
            fig.savefig(os.path.join(plotHist, 'momentMagHist.pdf'))
            fig.clf()

        # Write out the samples
        if outputSamp is not None:
            assert False, 'Not implemented yet'
            with open(os.path.join(outputSamp, 'momentMagSamples.dat'), 'w') as ofid:
                self.Mw.tofile(ofid)

        # All done
        return Mw

    def computePotencies(self):
        '''
        Computes the potency of each subSources.
        '''

        self.computeSourcesMoments()
        self.Potencies = np.sqrt(0.5 * np.sum( (self.Moments/self.Mu[:,None,None])**2, axis=(1,2))) 

        # All done
        return

    def Aki2Harvard(self):
        '''
        Transform the patch from the Aki convention to the Harvard convention.
        '''
 
        # Get Maki 
        Maki = self.Maki

        # Transform
        M = self._aki2harvard(Maki)

        # Store it
        self.Mharvard = M

        # All done 
        return

    def _aki2harvard(self, Min):
        '''
        Transform the moment from the Aki convention to the Harvard convention.
        '''

        # Create new tensor
        M = np.zeros_like(Min)

        # Shuffle things around following Aki & Richard, Second edition, pp 113
        M[0,0,...] = Min[2,2,...]
        M[1,0,...] = M[0,1,...] = Min[0,2,...]
        M[2,0,...] = M[0,2,...] = -1.0*Min[1,2,...]
        M[1,1,...] = Min[0,0,...]
        M[2,1,...] = M[1,2,...] = -1.0*Min[1,0,...]
        M[2,2,...] = Min[1,1,...]

        # All done
        return M

    def computeCentroidLonLatDepth(self, plotOutput=None, xyzOutput=None):
        '''
        Computes the equivalent centroid location.
        Take from Theoretical Global Seismology, Dahlen & Tromp. Chapter 5. Section 4. pp. 169
        '''

        # Check
        assert hasattr(self, 'Mharvard'), 'Compute the Moment tensor first'

        # Get the scalar moment
        Mo = self.computeScalarMoment()

        # Get the total Moment
        M = self.Maki

        # Get the moment of each subsource
        dS = self.Moments

        # Get the locations
        x, y, z = self.x, self.y, self.depth

        # Compute the normalized scalar moment density 
        m = 0.5/(Mo**2) * np.sum(M[None,:,:]*dS, axis=(1,2))

        # Centroid location
        xc = np.sum(m*x)
        yc = np.sum(m*y)
        zc = np.sum(m*z)

        # Store the x, y, z locations
        self.centroid = [xc, yc, zc]

        # Convert to lon lat
        lonc, latc = self.xy2ll(xc, yc)
        self.centroidll = [lonc, latc, zc]

        # Plot scatter
        if plotOutput is not None:
            assert False, 'Not implemented yet'
            assert isinstance(xc, np.ndarray), 'cannot make scatter plots with one value'
            fig = plt.figure(figsize=(14,8))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            for ax,datPair,ylabel in [(ax1,(xc,yc),'Northing'), (ax2,(xc,zc),'Depth (km)')]:
                ax.plot(datPair[0], datPair[1], '.b', alpha=0.7)
                ax.set_ylabel(ylabel, fontsize=18)
                ax.set_xlabel('Easting', fontsize=18)
                ax.tick_params(labelsize=18)
                ax.grid(True)
            ax1.plot(self.fault.xf, self.fault.yf, '-r', linewidth=3)
            ax2.set_ylim(ax2.get_ylim()[::-1])
            fig.savefig(os.path.join(plotOutput, 'centroidDists.png'), dpi=400, 
                        bbox_inches='tight')

        # Write points out
        if xyzOutput is not None:
            assert False, 'Not implemented yet'
            fid = open(os.path.join(xyzOutput, 'centroids.xyz'), 'w')
            for lon,lat,z in zip(*self.centroidll):
                fid.write('%15.9f%15.9f%12.6f\n' % (lon, lat, z))
            fid.close()

        return lonc, latc, zc

    def checkSymmetric(self, M):
        '''
        Check if a matrix is symmetric.
        '''

        # Check
        if M.ndim == 2:
            MT = M.T
        else:
            MT = np.transpose(M, (1,0,2))
        assert (M == MT).all(), 'Matrix is not symmetric'

        # all done
        return

    def computeBetaMagnitude(self):
        '''
        Computes the magnitude with a simple approximation.
        '''

        # Initialize moment
        Mo = 0.0

        # Get areas
        S = self.areas*1e6

        # Get slip
        strikeslip, dipslip = self.slip[:2,:,...]
        totalSlip = np.sqrt(strikeslip**2 + dipslip**2)

        # Moment
        Mo = self.Mu * S * totalSlip

        # Compute magnitude
        Mw = 2./3.*(np.log10(Mo) - 9.1)

        # All done
        return Mo, Mw

    def computeMomentAngularDifference(self, Mout, form='harvard'):
        '''
        Computes the difference in angle between the moment Mout and the moment.
            Args:
                Mout: full moment in harvard convention (if form=='aki', moment will be
                        transfered to harvard convention).
        '''

        # Assert
        assert False, 'Not implemented yet'

        # import stuff
        from numpy.linalg import eigh

        # Get Mout in the righ tconvention
        if form is 'aki':
            Mout = self._aki2harvard(Mout)

        # Calculate the Eigenvectors for Mout
        V,S    = eigh(Mout)
        inds   = np.argsort(V)
        S      = S[:,inds]
        S[:,2] = np.cross(S[:,0],S[:,1])
        V1 = copy.deepcopy(S)

        # Angles
        angles = []

        # Loop on the number of Mo
        for i in range(self.Mharvard.shape[2]):
            # Calculate the Eigenvectors 
            V,S    = eigh(self.Mharvard[:,:,i])
            inds   = np.argsort(V)                        
            S      = S[:,inds]         
            S[:,2] = np.cross(S[:,0],S[:,1])
            V2 = copy.deepcopy(S)
            # Calculate theta
            th = np.arccos((np.trace(np.dot(V1,V2.transpose()))-1.)/2.)
            # find the good value
            for j in range(3):
                k       = (j+1)%3
                V3      = copy.deepcopy(V2)
                V3[:,j] = -V3[:,j]
                V3[:,k] = -V3[:,k]
                x       = np.arccos((np.trace(np.dot(V1,V3.transpose()))-1.)/2.) 
                if x < th:
                        th = x

            angles.append(th*180./np.pi)

        # All done
        return angles

    def integrateQuantityAlongProfile(self, lonc, latc, length, azimuth, width, numXBins=100, fault=None, quantity='potency', getDepth=False, method='sum'):
        '''
        Computes the cumulative potency as a function of distance to the profile origin.
        If the potencies were computed with multiple samples (in case of Bayesian exploration), we form histograms
        of potency vs. distance. Otherwise, we just compute a distance profile.

        Args:
            lonc, latc                  : Lon and Lat of the center of the profile
            length                      : Length of the profile (km)
            azimuth                     : Azimuth of the profile (degrees)
            width                       : Width of the profile (km)
            numXBins                    : number of bins to group patches along the profile
            fault                       : If provided, the profile will be referenced at the intersection with teh fault trace.
            quantity                    : Which quantity do we deal with. Can be:
                                            'potency'
                                            'moment'
                                            'strikeslip'
                                            'dipslip'
                                            'tensile'
            method                      : Can be 'sum', 'mean', 'median', 'min' or 'max'
        '''

        # Get the profile
        xc, yc = self.ll2xy(lonc, latc)
        xDis, yDis, BValues, boxll, box, xe1, ye1, xe2, ye2, lon, lat = utils.coord2prof(self, xc, yc, 
                                                                                    length, 
                                                                                    azimuth, 
                                                                                    width)

        # Get the right quantity
        if quantity=='potency':
            self.computePotencies()
            quantities = self.Potencies[BValues]
        elif quantity=='moment':
            self.computeSourcesMoments()
            quantities = self.Moments[BValues,:,:]
        elif quantity=='strikeslip':
            quantities = self.slip[BValues,0]
        elif quantity=='dipslip':
            quantities = self.slip[BValues,1]
        elif quantity=='tensile':
            quantities = self.slip[BValues,2]
        else:
            assert False, 'Unknown quantity'

        # Make bins
        xmin, xmax = xDis.min(), xDis.max()
        xbins = np.linspace(xmin, xmax, numXBins+1)
        binDistances = 0.5 * (xbins[1:] + xbins[:-1])

        # Depth?
        if getDepth:
            depths = self.depth[BValues]

        # Loop over the bin depths
        scalarQuant = [];
        Depth = [];
        for xstart, xend in zip(xbins[:-1], xbins[1:]):

            # Get indexes
            ind = xDis>= xstart
            ind *= xDis<xend
            ind = ind.nonzero()[0]

            # Sum the total potency
            if method=='sum':
                value = np.sum(quantities[ind])
            elif method=='mean':
                value = np.mean(quantities[ind])
            elif method=='median':
                value = np.median(quantities[ind])
            elif method=='min':
                value = np.min(quantities[ind])
            elif method=='max':
                value = np.max(quantities[ind])
            scalarQuant.append(value)

            # Depth?
            if getDepth:
                Depth.append(np.mean(depths[ind]))

        # if fault, set the distances to the fault trace
        if fault is not None:
            binDistances -= utils.intersectProfileFault(xe1, ye1, xe2, ye2, xc, yc, self.fault)

        # All done
        if getDepth:
            return binDistances, np.array(Depth), np.array(scalarQuant)
        else:
            return binDistances, np.array(scalarQuant)

    def integrateQuantityWithDepth(self, plotOutput=None, numDepthBins=5, outputSamp=None, quantity='potency', method='sum'):
        '''
        Computes the cumulative moment with depth by summing the moment per row of
        patches. If the moments were computed with mutiple samples, we form histograms of 
        potency vs. depth. Otherwise, we just compute a depth profile.

        kwargs:
            plotOutput                      output directory for figures
            numDepthBins                    number of bins to group patch depths
            quantity                    : Which quantity do we deal with. Can be:
                                            'potency'
                                            'moment'
                                            'strikeslip'
                                            'dipslip'
                                            'tensile'
            method                      : Can be 'sum', 'mean', 'median'
        '''

        # Collect all sources depths
        sourceDepths = self.depth

        # Determine depth bins for grouping
        zmin, zmax = sourceDepths.min(), sourceDepths.max()
        zbins = np.linspace(zmin, zmax, numDepthBins+1)
        binDepths = 0.5 * (zbins[1:] + zbins[:-1])

        # Get the right quantity
        if quantity=='potency':
            self.computePotencies()
            quantities = self.Potencies
        elif quantity=='moment':
            self.computeSourcesMoments()
            quantities = self.Moments
        elif quantity=='strikeslip':
            quantities = self.slip[:,0]
        elif quantity=='dipslip':
            quantities = self.slip[:,1]
        elif quantity=='tensile':
            quantities = self.slip[:,2]

        # Loop over depth bins
        scalarQuant = []; 
        for i in range(numDepthBins):

            # Get the patch indices that fall in this bin
            zstart, zend = zbins[i], zbins[i+1]
            ind = sourceDepths >= zstart
            ind *= sourceDepths < zend
            ind = ind.nonzero()[0]

            # Sum the total moment for the depth bin
            if method=='sum':
                value = np.sum(quantities[ind])
            elif method=='mean':
                value = np.mean(quantities[ind])
            elif method=='median':
                value = np.median(quantities[ind])
            elif method=='min':
                value = np.min(quantities[ind])
            elif method=='max':
                value = np.max(quantities[ind])
            scalarQuant.append(value)
    
        return binDepths, np.array(scalarQuant)

    def write2GCMT(self, form='full', filename=None):
        '''
        Writes in GCMT style
        Args:
            * form          : format is either 'full' to match with Zacharie binary
                                            or 'line' to match with the option -Sm in GMT

        Example of 'full':
         PDE 2006  1  1  7 11 57.00  31.3900  140.1300  10.0 5.3 5.0 SOUTHEAST OF HONSHU, JAP                
        event name:     200601010711A  
        time shift:     10.4000
        half duration:   1.5000
        latitude:       31.5100
        longitude:     140.0700
        depth:          12.0000
        Mrr:       3.090000e+24
        Mtt:      -2.110000e+24
        Mpp:      -9.740000e+23
        Mrt:      -6.670000e+23
        Mrp:      -5.540000e+23
        Mtp:      -5.260000e+23
        '''

        # Check
        assert hasattr(self,'Mharvard'), 'Compute the Moment tensor first'

        # Get the moment
        M = self.Mharvard

        # Get lon lat
        lon, lat, depth = self.computeCentroidLonLatDepth()

        # Check filename
        if filename is not None:
            fout = open(filename, 'w')
        else:
            fout = sys.stdout

        if form is 'full':
            # Write the BS header
            fout.write(' PDE 1999  1  1  9 99 99.00  99.9900   99.9900  99.0 5.3 5.0 BULLSHIT \n')
            fout.write('event name:    thebigbaoum \n')
            fout.write('time shift:    99.9999     \n')
            fout.write('half duration: 99.9999     \n')
            fout.write('latitude:       {}     \n'.format(lat))
            fout.write('longitude:      {}     \n'.format(lon))
            fout.write('depth:          {}     \n'.format(depth))
            fout.write('Mrr:           {:7e}       \n'.format(M[0,0]*1e7))
            fout.write('Mtt:           {:7e}       \n'.format(M[1,1]*1e7))
            fout.write('Mpp:           {:7e}       \n'.format(M[2,2]*1e7))
            fout.write('Mrt:           {:7e}       \n'.format(M[0,1]*1e7))
            fout.write('Mrp:           {:7e}       \n'.format(M[0,2]*1e7))
            fout.write('Mtp:           {:7e}       \n'.format(M[1,2]*1e7))
        elif form is 'line':
            # get the largest mantissa
            mantissa = 0
            A = [M[0,0], M[1,1], M[2,2], M[0,1], M[0,2], M[1,2]]
            for i in range(6):
                if np.abs(A[i])>0.0:
                    exp = int(np.log10(np.abs(A[i])))
                    if exp > mantissa:
                        mantissa = exp
            mrr = (M[0,0])/10**mantissa
            mtt = (M[1,1])/10**mantissa
            mpp = (M[2,2])/10**mantissa
            mrt = (M[0,1])/10**mantissa
            mrp = (M[0,2])/10**mantissa
            mtp = (M[1,2])/10**mantissa
            fout.write('{} {} {} {:3f} {:3f} {:3f} {:3f} {:3f} {:3f} {:d} \n'.format(
                lon, lat, depth, mrr, mtt, mpp, mrt, mrp, mtp, mantissa+7))

        # Close file
        if filename is not None:
            fout.close()
        else:
            fout.flush()

        # All done
        return

#EOF
