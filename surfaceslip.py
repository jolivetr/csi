'''
A class that deals with surface slip data

Written by R. Jolivet in 2021
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import matplotlib.path as path
import scipy.spatial.distance as scidis
import copy
import sys, os

# Personals
from .SourceInv import SourceInv
from .geodeticplot import geodeticplot as geoplot
from . import csiutils as utils

class surfaceslip(SourceInv):
    '''
    Args:
        * name      : Name of the surfaceslip dataset

    Kwargs:
        * utmzone   : UTM zone. (optional, default is 10 (Western US))
        * lon0      : Longitude of the utmzone
        * lat0      : Latitude of the utmzone
        * ellps     : ellipsoid (optional, default='WGS84')

    Returns:
        * None
    '''

    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True, lon0=None, lat0=None):

        # Base class init
        super(surfaceslip,self).__init__(name,
                                         utmzone=utmzone,
                                         ellps=ellps,
                                         lon0=lon0,
                                         lat0=lat0)

        # Initialize the data set
        self.dtype = 'surfaceslip'

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize Surface Slip data set {}".format(self.name))
        self.verbose = verbose

        # Initialize some things
        self.vel = None
        self.synth = None
        self.err = None
        self.lon = None
        self.lat = None
        self.Cd = None

        # This is in case surface slip is in the LOS of the satellite
        self.los = None

        # All done
        return

    def checkZeros(self):
        '''
        Checks and remove data points that have Zeros in vel, lon or lat
        '''

        # Check
        if self.vel is not None:
            uVel = np.flatnonzero(self.vel==0.)
        else:
            uVel = np.array([])

        # Reject data
        self.reject(uVel)

        # All done
        return

    def checkNaNs(self):
        '''
        Checks and remove data points that have NaNs in vel, err, lon, lat.
        '''

        # Check
        if self.vel is not None:
            uVel = np.flatnonzero(np.isnan(self.vel))
        else:
            uVel = np.array([])
        if self.err is not None:
            uErr = np.flatnonzero(np.isnan(self.err))
        else:
            uErr = np.array([])
        if self.lon is not None:
            uLon = np.flatnonzero(np.isnan(self.lon))
        else:
            uLon = np.array([])
        if self.lat is not None:
            uLat = np.flatnonzero(np.isnan(self.lat))
        else:
            uLat = np.array([])
        if self.los is not None:
            uLos, toto = np.where(np.isnan(self.los))
            uLos = np.unique(uLos.flatten())
        else:
            uLos = np.array([])

        # Concatenate all these guys
        uRemove = np.concatenate((uVel, uErr, uLon, uLat, uLos)).astype(int)
        uRemove = np.unique(uRemove)

        # Reject pixels
        self.deletePixels(uRemove)

        # All done
        return

    def read_from_binary(self, data, lon, lat, err=None, factor=1.0, downsample=1,
                               step=0.0, los=None, dtype=np.float32):
        '''
        Read from binary file or from array.

        Args:
            * data      : binary array containing the data or binary file
            * lon       : binary arrau containing the longitude or binary file
            * lat       : binary array containing the latitude or binary file

        Kwargs:
            * err           : Uncertainty (array)
            * factor        : multiplication factor (default is 1.0)
            * step          : constant added to the data (default is 0.0)
            * los           : LOS unit vector 3 component array (3-column array)
            * dtype         : data type (default is np.float32 if data is a file)

        Return:
            * None
        '''

        # Get the data
        if type(data) is str:
            vel = np.fromfile(data, dtype=dtype)[::downsample]*factor + step
        else:
            vel = data.flatten()[::downsample]*factor + step
        # Get the lon
        if type(lon) is str:
            lon = np.fromfile(lon, dtype=dtype)[::downsample]
        else:
            lon = lon.flatten()[::downsample]

        # Get the lat
        if type(lat) is str:
            lat = np.fromfile(lat, dtype=dtype)[::downsample]
        else:
            lat = lat.flatten()[::downsample]

        # Check sizes
        assert vel.shape==lon.shape, 'Something wrong with the sizes: {} {} {} '.format(vel.shape, lon.shape, lat.shape)
        assert vel.shape==lat.shape, 'Something wrong with the sizes: {} {} {} '.format(vel.shape, lon.shape, lat.shape)

        # Get the error
        if err is not None:
            if type(err) is str:
                err = np.fromfile(err, dtype=dtype)[::downsample]*np.abs(factor)
            else:
                err = err.flatten()[::downsample]*np.abs(factor)
            assert vel.shape==err.shape, 'Something wrong with the sizes: {} {}'.format(vel.shape, err.shape)

        # Get the LOS
        if los is not None:
            if type(los) is str:
                los = np.fromfile(los, dtype=dtype).reshape((vel.shape[0], 3))
            assert los.shape[0]==vel.shape[0] and los.shape[1]==3, 'Something wrong with the sizes: {} {}'.format(los.shape, vel.shape)

        # Set things in self
        self.vel = vel
        self.err = err
        self.lon = lon
        self.lat = lat
        self.los = los

        # Keep track of factor
        self.factor = factor

        # set lon to (0, 360.)
        self._checkLongitude()

        # compute x, y
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # All done
        return

    def resample(self, nSamples, method='linear', axis='lon'):
        '''
        Linear resampling as a function of longitude or latitude.
        '''
        raise NotImplemented
        return 

    def buildCd(self):
        '''
        Builds a full Covariance matrix from the uncertainties. The Matrix is just a diagonal matrix.
        '''

        # Assert
        assert self.err is not None, 'Need some uncertainties on the LOS displacements...'

        # Get some size
        nd = self.vel.shape[0]

        # Fill Cd
        self.Cd = np.diag(self.err**2)

        # All done
        return

    def distance2point(self, lon, lat):
        '''
        Returns the distance of all pixels to a point.

        Args:
            * lon       : Longitude of a point
            * lat       : Latitude of a point

        Returns:
            * array
        '''

        # Get coordinates
        x = self.x
        y = self.y

        # Get point coordinates
        xp, yp = self.ll2xy(lon, lat)

        # compute distance
        return np.sqrt( (x-xp)**2 + (y-yp)**2 )

    def keepWithin(self, minlon, maxlon, minlat, maxlat):
        '''
        Select the pixels in a box defined by min and max, lat and lon.

        Args:
            * minlon        : Minimum longitude.
            * maxlon        : Maximum longitude.
            * minlat        : Minimum latitude.
            * maxlat        : Maximum latitude.

        Retunrs:
            * None
        '''

        # Store the corners
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat

        # Select on latitude and longitude
        u = np.flatnonzero((self.lat>minlat) & (self.lat<maxlat) & (self.lon>minlon) & (self.lon<maxlon))

        # Do it
        self.keepDatas(u)

        # All done
        return

    def keepDatas(self, u):
        '''
        Keep the datas  indexed u and ditch the other ones

        Args:
            * u         : array of indexes

        Returns:
            * None
        '''

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.vel = self.vel[u]
        if self.err is not None:
            self.err = self.err[u]
        if self.los is not None:
            self.los = self.los[u]
        if self.synth is not None:
            self.synth = self.synth[u]

        # Deal with the covariance matrix
        if self.Cd is not None:
            Cdt = self.Cd[u,:]
            self.Cd = Cdt[:,u]

        # All done
        return

    def deleteDatas(self, u):
        '''
        Delete the datas indicated by index in u.

        Args:
            * u         : array of indexes

        Returns:
            * None  
        '''

        # Select the stations
        self.lon = np.delete(self.lon,u)
        self.lat = np.delete(self.lat,u)
        self.x = np.delete(self.x,u)
        self.y = np.delete(self.y,u)
        self.vel = np.delete(self.vel,u)
        if self.err is not None:
            self.err = np.delete(self.err,u)
        if self.los is not None:
            self.los = np.delete(self.los,u, axis=0)
        if self.synth is not None:
            self.synth = np.delete(self.synth, u)

        # Deal with the covariance matrix
        if self.Cd is not None:
            self.Cd = np.delete(np.delete(Cd ,u, axis=0), u, axis=1)

        # All done
        return

    def getTransformEstimator(self, trans, computeNormFact=True):
        '''
        Returns the Estimator for the transformation to estimate in the surfaceslip data.
        The estimator is only zeros
    
        Args:
            * trans     : useless

        Kwargs:
            * computeNormFact   : useless

        Returns:
            * None
        '''

        # One case
        T = np.zeros((len(self.vel), 1))

        # All done
        return T

    def setGFsInFault(self, fault, G, vertical=True):
        '''
        From a dictionary of Green's functions, sets these correctly into the fault
        object fault for future computation.

        Args:
            * fault     : Instance of Fault
            * G         : Dictionary with 3 entries 'strikeslip', 'dipslip' and 'tensile'. These can be a matrix or None.

        Kwargs:
            * vertical  : Set here for consistency with other data objects, but will always be set to True, whatever you do.

        Returns:
            * None
        '''

        if fault.type == 'Fault':

            # Get the values
            Gss = G['strikeslip']
            Gds = G['dipslip']

            # Here coupling and tensile make no sense
            fault.setGFs(self, strikeslip=[Gss], dipslip=[Gds], tensile=[None],
                        coupling=[None], vertical=True)

        elif fault.type == 'Pressure':

            try:
                GpLOS = G['pressure']
            except:
                GpLOS = None
            try:
                GdvxLOS = G['pressureDVx']
            except:
                GdvxLOS = None
            try:
                GdvyLOS = G['pressureDVy']
            except:
                GdvyLOS = None
            try:
                GdvzLOS = G['pressureDVz']
            except:
                GdvzLOS = None

            fault.setGFs(self, deltapressure=[GpLOS], 
                               GDVx=[GdvxLOS] , GDVy=[GdvyLOS], GDVz =[GdvzLOS], 
                               vertical=True)

        # All done
        return

    def removeTransformation(self, fault, verbose=False, custom=False):
        '''
        Wrapper to ensure consistency between data sets.

        Args:
            * fault     : a fault instance

        Kwargs:
            * verbose   : talk to us
            * custom    : Remove custom GFs

        Returns:
            * None
        '''

        # No transformation is implemented, nothing to do
        # All done
        return

    def removeSynth(self, faults, direction='sd', poly=None, vertical=True, custom=False, computeNormFact=True):
        '''
        Removes the synthetics using the faults and the slip distributions that are in there.

        Args:
            * faults        : List of faults.

        Kwargs:
            * direction         : Direction of slip to use.
            * poly              : if a polynomial function has been estimated, build and/or include
            * vertical          : always True - used here for consistency among data types
            * custom            : if True, uses the fault.custom and fault.G[data.name]['custom'] to correct
            * computeNormFact   : if False, uses TransformNormalizingFactor set with self.setTransformNormalizingFactor

        Returns:
            * None
        '''

        # Build synthetics
        self.buildsynth(faults, direction=direction, poly=poly, 
                                custom=custom, computeNormFact=computeNormFact)

        # Correct
        self.vel -= self.synth

        # All done
        return

    def buildsynth(self, faults, direction='sd', poly=None, vertical=True, custom=False, computeNormFact=True):
        '''
        Computes the synthetic data using either the faults and the associated slip distributions or the pressure sources.

        Args:
            * faults        : List of faults or pressure sources.

        Kwargs:
            * direction         : Direction of slip to use or None for pressure sources.
            * poly              : if a polynomial function has been estimated, build and/or include
            * vertical          : always True. Used here for consistency among data types
            * custom            : if True, uses the fault.custom and fault.G[data.name]['custom'] to correct
            * computeNormFact   : if False, uses TransformNormalizingFactor set with self.setTransformNormalizingFactor

        Returns:
            * None
        '''

        # Check list
        if type(faults) is not list:
            faults = [faults]

        # Number of data
        Nd = self.vel.shape[0]

        # Clean synth
        self.synth = np.zeros((self.vel.shape))

        # Loop on each fault
        for fault in faults:

            if fault.type=="Fault":

                # Get the good part of G
                G = fault.G[self.name]

                if ('s' in direction) and ('strikeslip' in G.keys()):
                    Gs = G['strikeslip']
                    Ss = fault.slip[:,0]
                    synth = np.dot(Gs,Ss)
                    self.synth += synth
                if ('d' in direction) and ('dipslip' in G.keys()):
                    Gd = G['dipslip']
                    Sd = fault.slip[:,1]
                    synth = np.dot(Gd, Sd)
                    self.synth += synth

        # All done
        return

    def getRMS(self):
        '''
        Computes the RMS of the data and if synthetics are computed, the RMS of the residuals

        Returns:
            * float, float
        '''

        # Get the number of points
        N = self.vel.shape[0]

        # RMS of the data
        dataRMS = np.sqrt( 1./N * sum(self.vel**2) )

        # Synthetics
        values = copy.deepcopy(self.vel)
        if self.synth is not None:
            values -= self.synth
        #obsolete if self.orbit is not None:
        #    values -= self.orbit
        synthRMS = np.sqrt( 1./N *sum( (values)**2 ) )

        # All done
        return dataRMS, synthRMS

    def getVariance(self):
        '''
        Computes the Variance of the data and if synthetics are computed, the RMS of the residuals

        Returns:
            * float, float
        '''

        # Get the number of points
        N = self.vel.shape[0]

        # Varianceof the data
        dmean = self.vel.mean()
        dataVariance = ( 1./N * sum((self.vel-dmean)**2) )

        # Synthetics
        values = copy.deepcopy(self.vel)
        if self.synth is not None:
            values -= self.synth
        if self.orbit is not None:
            values -= self.orbit
        synthVariance = ( 1./N *sum( (values - values.mean())**2 ) )

        # All done
        return dataVariance, synthVariance

    def getMisfit(self):
        '''
        Computes the Summed Misfit of the data and if synthetics are computed, the RMS of the residuals

        Returns:
            * float, float
        '''

        # Misfit of the data
        dataMisfit = sum((self.vel))

        # Synthetics
        if self.synth is not None:
            synthMisfit =  sum( (self.vel - self.synth) )
            return dataMisfit, synthMisfit
        else:
            return dataMisfit, 0.

        # All done

    def plot(self, show=True, figsize=None, axis='lon'):
        '''
        Plot the data set, together with fault slip if asked. 

        Kwargs:
            * show              : bool. Show on screen?
            * figsize           : tuple of figure sizes
            * axis              : which quantity to use as x-axis

        Returns:
            * None
        '''

        # X-xaxis
        if axis == 'lon':
            x = self.lon
        elif axis == 'lat':
            x = self.lat
        else:
            print('Unkown axis type: {}'.format(axis))
            return

        # Create a figure
        if figsize is None:
            figsize=(10,3)
        fig,ax = plt.subplots(1,1,figsize=figsize)

        # Plot the data
        u = np.argsort(x)
        if self.err is None:
            ax.plot(x[u], self.vel[u], '.-', color='k', label='Data', markersize=5)
        else:
            ax.fill_between(x[u], self.vel[u]+self.err[u], self.vel[u]-self.err[u], 
                            color='k', alpha=0.3, zorder=1)
            ax.plot(x[u], self.vel[u], '.-', color='k', zorder=2, label='Data')

        # Synthetics
        if self.synth is not None:
            ax.plot(x[u], self.synth[u], '.-', color='r', label='Synthetics', markersize=5, zorder=3)

        ax.legend()

        # Title
        ax.set_title('{}'.format(self.name))

        # Show
        if show: plt.show()
        
        # Save the whole thing
        self.fig = fig
        self.ax = ax

        # All done
        return

    def write2file(self, fname, data='data', outDir='./'):
        '''
        Write to an ascii file

        Args:
            * fname     : Filename

        Kwargs:
            * data      : can be 'data', 'synth' or 'resid'
            * outDir    : output Directory

        Returns:
            * None
        '''

        # Get variables
        x = self.lon
        y = self.lat
        if data=='data':
            z = self.vel
        elif data=='synth':
            z = self.synth
        elif data=='resid':
            z = self.vel - self.synth

        # Write these to a file
        fout = open(os.path.join(outDir, fname), 'w')
        for i in range(x.shape[0]):
            fout.write('{} {} {} \n'.format(x[i], y[i], z[i]))
        fout.close()

        return

    def checkLOS(self, figure=1, factor=100., decim=1):
        '''
        Plots the LOS vectors in a 3D plot.

        Kwargs:
            * figure:   Figure number.
            * factor:   Increases the size of the vectors.
            * decim :   Do not plot all the pixels (takes way too much time)

        Returns:
            * None
        '''

        # Display
        print('Checks the LOS orientation')

        # Create a figure
        fig = plt.figure(figure)

        # Create an axis instance
        ax = fig.add_subplot(111, projection='3d')

        # Loop over the LOS
        for i in range(0,self.vel.shape[0],decim):
            x = [self.x[i], self.x[i]+self.los[i,0]*factor]
            y = [self.y[i], self.y[i]+self.los[i,1]*factor]
            z = [0, self.los[i,2]*factor]
            ax.plot3D(x, y, z, '-k')

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_zlabel('Up')

        # Show it
        plt.show()

        # All done
        return

#EOF
