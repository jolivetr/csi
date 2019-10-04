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

# Personals
from .SourceInv import SourceInv

class faultpostproc(SourceInv):

    '''
    A class that allows to compute various things from a fault object.
    
    Args:
        * name          : Name of the InSAR dataset.
        * fault         : Fault object
    
    Kwargs:
        * Mu            : Shear modulus. Default is 24e9 GPa, because it is the PREM value for the upper 15km. Can be a scalar or a list/array of len=len(fault.patch)
        * samplesh5     : file name of h5 file containing samples
        * utmzone       : UTM zone  (optional, default=None)
        * lon0          : Longitude of the center of the UTM zone
        * lat0          : Latitude of the center of the UTM zone
        * ellps         : ellipsoid (optional, default='WGS84')
        * verbose       : Speak to me (default=True)
     
    '''

    def __init__(self, name, fault, Mu=24e9, samplesh5=None, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Base class init
        super(faultpostproc,self).__init__(name,
                                           utmzone = utmzone,
                                           ellps = ellps, 
                                           lon0 = lon0, lat0 = lat0) 

        # Initialize the data set 
        self.name = name
        self.fault = copy.deepcopy(fault) # we don't want to modify fault slip
        self.patchDepths = None
        self.MTs = None

        # Determine number of patches along-strike and along-dip
        self.numPatches = len(self.fault.patch)
        if self.fault.numz is not None:
            self.numDepthPatches = self.fault.numz
            self.numStrikePatches = self.numPatches / self.numDepthPatches
            
        # Assign Mu to each patch
        if len(np.array(Mu).flatten())==1:
            self.Mu = Mu * np.ones((self.numPatches,))
        else:
            assert len(Mu)==self.numPatches, 'length of Mu must be 1 or numPatch'
            self.Mu = np.array(Mu)
            
        # Display
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize Post Processing object {} on fault {}".format(self.name, fault.name))
        self.verbose = verbose

        # Check to see if we're reading in an h5 file for posterior samples
        self.samplesh5 = samplesh5

        # All done
        return

    def h5_init(self, decim=1,indss=None,indds=None):
        '''
        If the attribute self.samplesh5 is not None, we open the h5 file specified by 
        self.samplesh5 and copy the slip values to self.fault.slip (hopefully without loading 
        into memory).

        Kwargs:
            * decim :  decimation factor for skipping samples
            * indss :  tuples (size (2,)) containing desired indices of strike slip in h5File
            * indds :  tuples (size (2,)) containing desired indices of dip slip in h5File

        Returns:
            * None
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

            if indss is None or indds is None:
                nsamples = np.arange(0, samples.shape[0], decim).size
                self.fault.slip = np.zeros((self.numPatches,3,nsamples))
                self.fault.slip[:,0,:] = samples[::decim,:self.numPatches].T
                self.fault.slip[:,1,:] = samples[::decim,self.numPatches:2*self.numPatches].T

            else:
                assert indss[1]-indss[0] == self.numPatches, 'indss[1] - indss[0] different from number of patches'
                assert indss[1]-indss[0] == self.numPatches, 'indds[1] - indds[0] different from number of patches'
                nsamples =  np.arange(0, samples.shape[0], decim).size
                self.fault.slip = np.zeros((self.numPatches,3,nsamples))
                self.fault.slip[:,0,:] = samples[::decim,indss[0]:indss[1]].T
                self.fault.slip[:,1,:] = samples[::decim,indds[0]:indds[1]].T


        return

    def h5_finalize(self):
        '''
        Close the (potentially) open h5 file.

        Returns:
            * None
        '''
        if hasattr(self, 'hfid'):
            self.hfid.close()

        return
            
    def lonlat2xy(self, lon, lat):
        '''
        Uses the transformation in self to convert  lon/lat vector to x/y utm.

        Args:
            * lon           : Longitude array.
            * lat           : Latitude array.

        Returns:
            * None
        '''

        x, y = self.putm(lon,lat)
        x /= 1000.
        y /= 1000.

        return x, y

    def xy2lonlat(self, x, y):
        '''
        Uses the transformation in self to convert x.y vectors to lon/lat.

        Args:
            * x             : Xarray
            * y             : Yarray

        Returns:
            * lon, lat      : 2 float arrays
        '''

        lon, lat = self.putm(x*1000., y*1000., inverse=True)
        return lon, lat

    def patchNormal(self, p):
        '''
        Returns the Normal to a patch.

        Args:
            * p             : Index of the desired patch.

        Returns:
            * unit normal vector
        '''

        if self.fault.patchType == 'triangle':
            normal = self.fault.getpatchgeometry(p, retNormal=True)[-1]
            return normal

        elif self.fault.patchType == 'rectangle':

            # Get the geometry of the patch
            x, y, z, width, length, strike, dip = self.fault.getpatchgeometry(p, center=True)

            # Normal
            n1 = -1.0*np.sin(dip)*np.sin(strike)
            n2 = np.sin(dip)*np.cos(strike)
            n3 = -1.0*np.cos(dip)
            N = np.sqrt(n1**2+ n2**2 + n3**2)

            # All done
            return np.array([n1/N, n2/N, n3/N])

        else:
            assert False, 'unsupported patch type'


    def slipVector(self, p):
        '''
        Returns the slip vector in the cartesian space for the patch p. We do not deal with 
        the opening component. The fault slip may be a 3D array for multiple samples of slip.
        Args:
            * p             : Index of the desired patch.
        '''

        # Get the geometry of the patch
        x, y, z, width, length, strike, dip = self.fault.getpatchgeometry(p, center=True)

        # Get the slip
        strikeslip, dipslip, tensile = self.fault.slip[p,:,...]
        slip = np.sqrt(strikeslip**2 + dipslip**2)

        # Get the rake
        rake = np.arctan2(dipslip, strikeslip)

        # Vectors
        ux = slip*(np.cos(rake)*np.cos(strike) + np.cos(dip)*np.sin(rake)*np.sin(strike))
        uy = slip*(np.cos(rake)*np.sin(strike) - np.cos(dip)*np.sin(rake)*np.cos(strike))
        uz = -1.0*slip*np.sin(rake)*np.sin(dip)

        # All done
        if isinstance(ux, np.ndarray):
            outArr = np.zeros((3,1,ux.size))
            outArr[0,0,:] = ux
            outArr[1,0,:] = uy
            outArr[2,0,:] = uz
            return outArr
        else:
            return np.array([[ux], [uy], [uz]])

    def computePatchMoment(self, p) :
        '''
        Computes the Moment tensor for one patch.
        Args:
            * p             : patch index
        '''

        # Get the normal
        n = self.patchNormal(p).reshape((3,1))

        # Get the slip vector
        u = self.slipVector(p)

        # Compute the moment density
        if u.ndim == 2:
            mt = self.Mu[p] * (np.dot(u, n.T) + np.dot(n, u.T)) 
        elif u.ndim == 3:
            # Careful about tiling - result is already transposed
            nT = np.tile(n, (1,1,u.shape[2]))
            n = np.transpose(nT, (1,0,2))
            uT = np.transpose(u, (1,0,2))
            # Tricky 3D multiplication
            mt = self.Mu[p] * ((u[:,:,None]*nT).sum(axis=1) + (n[:,:,None]*uT).sum(axis=1))

        # Multiply by the area
        mt *= self.fault.area[p]*1000000.

        # All done
        return mt

    def computeMoments(self):
        '''
        Computes the moment tensor for each patch.
        Result is stored in self.Moments
        '''

        # Create the list
        Moments = []

        # Iterate
        for p in range(len(self.fault.patch)):
            Moments.append(self.computePatchMoment(p))
        
        # Save 
        self.Moments = Moments

        # All done
        return

    def computeMomentTensor(self):
        '''
        Computes the full seismic (0-order) moment tensor from the slip distribution.
        '''

        # Compute the area of each patch
        if not hasattr(self.fault, 'area'):
            self.fault.computeArea()

        # Initialize an empty moment
        M = 0.0

        # Compute the tensor for each patch
        self.MTs = []
        for p in range(len(self.fault.patch)):
            # Compute the moment of one patch
            mt = self.computePatchMoment(p)
            self.MTs.append(mt)
            # Add it up to the full tensor
            M += mt
            
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
            with open(os.path.join(outputSamp, 'momentMagSamples.dat'), 'w') as ofid:
                self.Mw.tofile(ofid)

        # All done
        return Mw

    def computePotencies(self):
        '''
        Computes the potencies for all the patches.
        Result is stored in self.Potencies
        '''

        # Compute the patch moments
        self.computeMoments()

        # calculate the potencies
        Potencies = [np.sqrt(0.5*np.sum(M**2, axis=(0,1)))/mu for M,mu in zip(self.Moments,self.Mu)]

        # Save
        self.Potencies = Potencies

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

        # initialize centroid loc.
        xc, yc, zc = 0.0, 0.0, 0.0

        # Loop on the patches
        for p in range(self.numPatches):

            # Get patch info 
            x, y, z, width, length, strike, dip = self.fault.getpatchgeometry(p, center=True)

            # Get the moment tensor
            dS = self.computePatchMoment(p)

            # Compute the normalized scalar moment density
            m = 0.5 / (Mo**2) * np.sum(M * dS, axis=(0,1))

            # Add it up to the centroid location
            xc += m*x
            yc += m*y
            zc += m*z

        # Store the x, y, z locations
        self.centroid = [xc, yc, zc]

        # Convert to lon lat
        lonc, latc = self.putm(xc*1000., yc*1000., inverse=True)
        self.centroidll = [lonc, latc, zc]

        # Plot scatter
        if plotOutput is not None:
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

        # Loop on patches
        for p in range(len(self.fault.patch)):

            # Get area
            S = self.fault.area[p]*1000000.

            # Get slip
            strikeslip, dipslip, tensile = self.fault.slip[p,:,...]

            # Add to moment
            Mo += self.Mu[p] * S * np.sqrt(strikeslip**2 + dipslip**2)

        # Compute magnitude
        Mw = 2./3.*(np.log10(Mo) - 9.1)

        # All done
        return Mo, Mw

    def computeMomentAngularDifference(self, Mout, form='harvard'):
        '''
        Computes the difference in angle between the moment Mout and the moment.
        Mout: full moment in harvard convention.
        '''

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

    def integratedPotencyAlongProfile(self, numXBins=100, outputSamp=None):
        '''
        Computes the cumulative potency as a function of distance to the profile origin.
        If the potencies were computed with multiple samples (in case of Bayesian exploration), we form histograms
        of potency vs. distance. Otherwise, we just compute a distance profile.

        kwargs:
            numXBins                        number of bins to group patches along the profile
        '''

        assert False, 'Not implemented for this kind of fault'

        return

    def integratedPotencyWithDepth(self, plotOutput=None, numDepthBins=5, outputSamp=None):
        '''
        Computes the cumulative moment with depth by summing the moment per row of
        patches. If the moments were computed with mutiple samples, we form histograms of 
        potency vs. depth. Otherwise, we just compute a depth profile.

        kwargs:
            plotOutput                      output directory for figures
            numDepthBins                    number of bins to group patch depths
        '''

        # Collect all patch depths
        patchDepths = np.empty((self.numPatches,))
        for pIndex in range(self.numPatches):
            patchDepths[pIndex] = self.fault.getpatchgeometry(pIndex, center=False)[2]

        # Determine depth bins for grouping
        zmin, zmax = patchDepths.min(), patchDepths.max()
        zbins = np.linspace(zmin, zmax, numDepthBins+1)
        binDepths = 0.5 * (zbins[1:] + zbins[:-1])
        dz = abs(zbins[1] - zbins[0])

        # Loop over depth bins
        potencyDict = {}; scalarPotencyList = []; meanLogPotency = []
        for i in range(numDepthBins):

            # Get the patch indices that fall in this bin
            zstart, zend = zbins[i], zbins[i+1]
            ind = patchDepths >= zstart
            ind *= patchDepths <= zend
            ind = ind.nonzero()[0]
            print(ind.size)

            # Sum the total moment for the depth bin
            M = 0.0
            for patchIndex in ind:
                M += self.computePatchMoment(int(patchIndex)) / self.Mu[patchIndex]
            # Convert to scalar potency
            potency = np.sqrt(0.5 * np.sum(M**2, axis=(0,1)))
            logPotency = np.log10(potency)
            meanLogPotency.append(np.log10(np.mean(potency)))

            # Create and store histogram for current bin
            if self.samplesh5 is not None:
                n, bins = np.histogram(logPotency, bins=100, density=True)
                binCenters = 0.5 * (bins[1:] + bins[:-1])
                zbindict = {}
                zbindict['count'] = n
                zbindict['bins'] = binCenters
                key = 'depthBin_%03d' % (i)
                potencyDict[key] = zbindict
            else:
                scalarPotencyList.append(potency)

        if plotOutput is not None:

            if self.samplesh5 is None:

                fig = plt.figure(figsize=(12,8))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                scalarPotency = np.array(scalarPotencyList)
                logPotency = np.log10(scalarPotency)
                sumLogPotency = np.log10(np.cumsum(scalarPotencyList))
                for ax,dat in [(ax1, logPotency), (ax2, sumLogPotency)]:
                    ax.plot(dat, binDepths, '-o')
                    ax.grid(True)
                    ax.set_xlabel('Log Potency', fontsize=16)
                    ax.set_ylabel('Depth (km)', fontsize=16)
                    ax.tick_params(labelsize=16)
                    ax.set_ylim(ax.get_ylim()[::-1])
                ax1.set_title('Potency vs. depth', fontsize=18)
                ax2.set_title('Integrated Potency vs. depth', fontsize=18)
                fig.savefig(os.path.join(plotOutput, 'depthPotencyDistribution.pdf'))

            else:
      
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111) 
                
                for depthIndex in range(numDepthBins):
                    # Get the histogram for the current depth
                    key = 'depthBin_%03d' % (depthIndex)
                    zbindict = potencyDict[key]
                    nref, bins = zbindict['count'], zbindict['bins']
                    n = nref.copy()
                    # Shift the histogram to the current depth and scale it
                    n /= n.max() / (0.5 * dz)
                    n -= binDepths[depthIndex]
                    # Plot normalized histogram
                    ax.plot(bins, -n)

                # Also draw the means
                ax.plot(meanLogPotency, binDepths, '-ob', linewidth=2)

                ax.set_ylim(ax.get_ylim()[::-1])
                ax.set_xlabel('Log potency', fontsize=18)
                ax.set_ylabel('Depth (km)', fontsize=18)
                ax.tick_params(labelsize=18)
                ax.grid(True)
                fig.savefig(os.path.join(plotOutput, 'depthPotencyDistribution.pdf'))

        # Save histogram for every depth bin
        if outputSamp is not None:
            assert self.samplesh5 is not None, 'cannot output only one sample'
            import h5py
            outfid = h5py.File(os.path.join(outputSamp, 'depthPotencyHistograms.h5'), 'w') 
            for depthIndex in range(numDepthBins):
                # Get the histogram for the current depth
                key = 'depthBin_%03d' % (depthIndex)
                zbindict = potencyDict[key]
                n, bins = zbindict['count'], zbindict['bins']
                # Save to h5
                depthSamp = outfid.create_dataset('depth_%fkm' % (binDepths[depthIndex]), 
                                                  (n.size,3), 'd')
                depthSamp[:,0] = bins
                depthSamp[:,1] = n
                depthSamp[:,2] = meanLogPotency[depthIndex]
            outfid.close()
    
        return


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

    def stressdrop(self,shapefactor=2.44,threshold=0.2,threshold_rand=False,return_Area_Mo_Slip=False):
        '''
        Compute threshold-dependent moment-based average stress-dip (cf., Noda et al., GJI 2013)
        Args:
            * shapefactor: shape factor (e.g., 2.44 for a circular crack,)
            * threshold: Rupture Area = area for slip > threshold * slip_max
            * threashold_rand: if ='log-normal' randomly generate threshold with mean threshold[0] 
                                   and sigma=threshold[1]
                               if ='uniform' randomly generate threshold between threshold[0] 
                                   and threshold[1]
                               if =False: compute stressdrop for a constant threshold
            * return_Area_Mo_Slip: if True, also return Rupture area as well as corresponding 
                                   scalar moment and averaged slip amplitude
        '''

        assert hasattr(self, 'MTs'), 'Compute moment tensor first'
        
        # Slip amplitude
        if self.fault.slip.ndim == 3:
            u       = self.fault.slip[:,:2,:]
            ndim = 3
        else:
            u       = self.fault.slip[:,:2]
            ndim = 2        
        slp     = np.sqrt((u*u).sum(axis=1))
        slp_max = slp.max(axis=0)
        plt.hist(slp_max)
        plt.show()
        if threshold_rand=='log-normal': # Use log-normal distributed thresholds
            th = scipy.random.lognormal(mean=threshold[0],sigma=threshold[1],size=slp_max.size)
        elif threshold_rand=='uniform': # Use uniform distributed thresholds
            th = scipy.random.uniform(low=threshold[0],high=threshold[1],size=slp_max.size)
        else:
            th = threshold * np.ones(slp_max.shape)
        slp_th = th * slp_max

        # Rupture Area and seismic moment
        area = np.zeros(slp_th.shape)
        Mo   = np.zeros(slp_th.shape)
        A = np.array(self.fault.area)
        Slip = np.zeros(slp_th.shape)
        for i in range(len(slp_th)):
            if ndim==3:
                ps = np.where(slp[:,i]>=slp_th[i])[0]
            else:
                ps = np.where(slp>=slp_th[i])[0]
            if ps.size>0:
                area[i] += A[ps].sum()*1000000.
                M = 0.0
                if ndim==3:
                    Slip[i] = slp[ps,i].mean()
                    for p in ps:
                        M += self.MTs[p][:,:,i]                    
                else:
                    Slip[i] = slp[ps].mean()
                    for p in ps:
                        M += self.MTs[p][:,:]
                self.checkSymmetric(M)
                Mo[i] = np.sqrt(0.5 * np.sum(M**2, axis=(0,1)))
        self.rupture_Mo   = Mo
        self.rupture_area = area

        # Scalar moment
        StressDrop = shapefactor * Mo/(area**1.5)
        self.StressDrop = StressDrop 
        
        # All done
        if return_Area_Mo_Slip:
            return area,Mo,Slip,self.StressDrop
        else:
            return self.StressDrop
                   

#EOF
