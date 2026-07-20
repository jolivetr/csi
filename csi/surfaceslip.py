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
from . import eulerPoleUtils as eu

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
        self.dist = None
        self.synth_err = None

        # This is in case surface slip is in the LOS of the satellite
        self.los = None

        # All done
        return

    def deletePixels(self, u):
        '''
        Delete the pixels indicated by index in u.
        
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
            self.Cd = np.delete(np.delete(self.Cd ,u, axis=0), u, axis=1)
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

    def getTransformEstimator(self, transformation, computeNormFact=True):
        '''
        Returns the Estimator for the transformation to estimate in the surfaceslip data.
        The estimator is only zeros
    
        Args:
            * transformation     : useless here but kept for consistency

        Kwargs:
            * computeNormFact   : useless here but kept for consistency

        Returns:
            * Estimator matrix (numpy array)
        '''

        # One case
        T = np.zeros((len(self.vel), 1))

        # All done
        return T

    def setGFsInSource(self, source, G, vertical=True):
        '''
        From a dictionary of Green's functions, sets these correctly into the source 
        object (fault, pressure, block or multiblock) for future computation.

        Args:
            * source    : Instance of Fault, Pressure, Block or MultiBlock
            * G         : Dictionary with entries 'strikeslip', 'dipslip' and 'tensile' for a fault,
            'pressure', 'pressureDVx', 'pressureDVy', 'pressureDVz' for a pressure,
            'rotation', 'intradef' for a block,
            'rotation', 'intradef', 'boundary' for a multiblock. These can be a matrix or None.

        Kwargs:
            * vertical  : Set here for consistency with other data objects, but will always be set to True, whatever you do.

        Returns:
            * None
        '''

        if source.type == 'Fault':

            # Get the values
            Gss = G['strikeslip']
            Gds = G['dipslip']
            Gts = G['tensile']
            
            # Here coupling and tensile make no sense
            source.setGFs(self, strikeslip=[Gss], dipslip=[Gds], tensile=[Gts],
                          coupling=[None], vertical=True)

        elif source.type == 'Pressure':

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

            source.setGFs(self, deltapressure=[GpLOS],
                          GDVx=[GdvxLOS] , GDVy=[GdvyLOS], GDVz =[GdvzLOS],
                          vertical=True)
        
        elif source.type == 'Block':
            
            # Import
            try:
                Grot = G['rotation']
            except:
                Grot = None
                
            try:
                Gint = G['intradef']
            except:
                Gint = None
            
            # Organize
            source.setGFs(self,
                          rotation=[Grot],
                          intradef=[Gint],
                          vertical=True)
        
        elif source.type == 'MultiBlock':
            
            # Import
            try:
                Grot = G['rotation']
            except:
                Grot = None
                
            try:
                Gint = G['intradef']
            except:
                Gint = None
            
            try:
                Gbound = G['boundary']
            except:
                Gbound = None
            
            # Organize
            source.setGFs(self,
                          rotation=[Grot],
                          intradef=[Gint],
                          boundary=[Gbound],
                          vertical=True)
        
        else:
            
            raise NotImplementedError("Source type {} not supported".format(source.type))

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

    def removeSynth(self, sources, direction='sd', blockcomponent='rot+intra+bound', poly=None, vertical=True, custom=False, computeNormFact=True):
        '''
        Removes the synthetics from given deformation sources.

        Args:
            * sources        : list of deformation sources to include.

        Kwargs:
            * direction         : Direction of slip to use.
            * blockcomponent   : for blocks and multiblocks, which components to use
            * poly              : if a polynomial function has been estimated, build and/or include
            * vertical          : always True - used here for consistency among data types
            * custom            : if True, uses the fault.custom and fault.G[data.name]['custom'] to correct
            * computeNormFact   : if False, uses TransformNormalizingFactor set with self.setTransformNormalizingFactor

        Returns:
            * None
        '''

        # Build synthetics
        self.buildsynth(sources, direction=direction, blockcomponent=blockcomponent,
                        poly=poly, custom=custom, computeNormFact=computeNormFact)

        # Correct
        self.vel -= self.synth

        # All done
        return

    def buildsynth(self, sources, direction='sd', blockcomponent='rot+intra+bound', poly=None, vertical=True, custom=False, computeNormFact=True):
        '''
        Computes the synthetic data using either the faults and the associated slip distributions or the pressure sources.

        Args:
            * sources        : List of deformation sources to include.

        Kwargs:
            * direction         : Direction of slip to use or None for pressure sources.
            * blockcomponent   : string or dictionary
                Which block components to consider.
                
                - If blockcomponent is a string, all blocks will have the same components considered.
                   Options are 'rotation', 'intradef', 'boundary' or any combination (e.g. 'rotation+intradef').
                   
                - If blockcomponent is a dictionary, each block can have different components considered (only valid for a multiblock)
                  The dictonary should be of the form {csi.Block.name: 'rotation+intradef+boundary'}
                  For example, if you want to consider only rotation for block A and rotation + intradef for block B, you can set
                  blockcomponent = {'A': 'rotation', 'B': 'rotation+intradef'}.
            * poly              : if a polynomial function has been estimated, build and/or include
            * vertical          : always True. Used here for consistency among data types
            * custom            : if True, uses the fault.custom and fault.G[data.name]['custom'] to correct
            * computeNormFact   : if False, uses TransformNormalizingFactor set with self.setTransformNormalizingFactor

        Returns:
            * None
        '''

        # Check list
        if type(sources) is not list:
            sources = [sources]

        # Number of data
        Nd = self.vel.shape[0]

        # Clean synth
        self.synth = np.zeros((self.vel.shape))

        # Loop on each source
        for source in sources:
            
            # Get the good part of G
            G = source.G[self.name]

            # Fault
            if source.type == "Fault":

                if ('s' in direction) and ('strikeslip' in G.keys()):
                    Gs = G['strikeslip']
                    Ss = source.slip[:, 0]
                    synth = np.dot(Gs, Ss)
                    self.synth += synth
                    
                if ('d' in direction) and ('dipslip' in G.keys()):
                    Gd = G['dipslip']
                    Sd = source.slip[:, 1]
                    synth = np.dot(Gd, Sd)
                    self.synth += synth
                
                if ('t' in direction) and ('tensile' in G.keys()):
                    Gt = G['tensile']
                    St = source.slip[:, 2]
                    synth = np.dot(Gt, St)
                    self.synth += synth
            
            # Block
            elif source.type == "Block":
                
                # Block rotation
                if ('rotation' in blockcomponent) and ('rotation' in G.keys()):
                    
                    Grot = G['rotation']
                    rot = np.array([source.omega_x, source.omega_y, source.omega_z]) / eu.MAS2RAD if source.omega_x is not None else np.zeros(3)
                    self.synth += Grot @ rot
                
                # Intra block deformation
                if ('intradef' in blockcomponent) and ('intradef' in G.keys()):
                
                    Gint = G['intradef']
                    eps = np.array([source.eps_lonlon, source.eps_lonlat, source.eps_latlat]) if source.eps_lonlon is not None else np.zeros(3)
                    self.synth += Gint @ eps
            
            # Multiblock
            elif source.type == "MultiBlock":

                # Get the block components to consider
                if type(blockcomponent) is str:
                    source.blockcomponent = {block.name: blockcomponent for block in source.blocks}
                else:
                    source.blockcomponent = blockcomponent
                
                synth = np.zeros((self.vel.shape))
                
                for iblock, block in enumerate(source.blocks):
                
                    # Block rotation
                    if ('rotation' in source.blockcomponent[block.name]) and ('rotation' in G.keys()):

                        Grot = G['rotation'][:, iblock*3:(iblock+1)*3]
                        rot = np.array([block.omega_x, block.omega_y, block.omega_z]) / eu.MAS2RAD if block.omega_x is not None else np.zeros(3)
                        synth_rot = Grot @ rot
                        
                        synth += synth_rot
                    
                    # Block boundary deformation
                    if ('boundary' in source.blockcomponent[block.name]) and ('boundary' in G.keys()):
                    
                        Gbound = G['boundary'][:, iblock*3:(iblock+1)*3]
                        rot = np.array([block.omega_x, block.omega_y, block.omega_z]) / eu.MAS2RAD if block.omega_x is not None else np.zeros(3)
                        synth_bound = Gbound @ rot
                        
                        synth += synth_bound
                    
                    # Intra block deformation
                    if ('intradef' in source.blockcomponent[block.name]) and ('intradef' in G.keys()):
                    
                        Gint = G['intradef'][:, iblock*3:(iblock+1)*3]
                        eps = np.array([block.eps_lonlon, block.eps_lonlat, block.eps_latlat]) if block.eps_lonlon is not None else np.zeros(3)
                        synth_intra = Gint @ eps
                        
                        synth += synth_intra
                
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
    
    def GetDistanceAlongFault(self, fault, discretize=True, coord='ll'):
        '''
        Computes the distance along fault for each data point.

        Args:
            * fault     : instance of Fault
            
        Kwargs:
            * discretize : bool. If True, use the discretized fault trace
            * coord      : coordinate system of lon, lat ('ll' / 'lonlat' or 'xy'/ 'utm')

        Returns:
            * array of distances
        '''
        
        # Check coordinates
        if coord in ('ll', 'lonlat'):
            x, y = self.lon, self.lat
        elif coord in ('xy', 'utm'):
            x, y = self.x, self.y

        # Loop over points
        Dalong = []
        
        for x_, y_ in zip(x, y):
            dalong_, _ = fault.distance2trace(lon=x_, lat=y_, discretized=discretize, coord=coord)
            Dalong.append(dalong_)
        
        self.dist = np.array(Dalong)

        # All done
        return 


    def plot(self, data=['data'], show=True, figsize=None, axis='lon', error=True, color=['k']):
        '''
        Plot the data set, together with fault slip if asked. 

        Kwargs:
            * data              : list of data to plot (can be 'data', 'synth', 'res' or 'transformation')
            * show              : bool. Show on screen?
            * figsize           : tuple of figure sizes
            * axis              : which quantity to use as x-axis, can be 'lon', 'lat', 'x', 'y' or 'distance'
            * error             : bool. If True, plot error bars
            * color             : list of colors to use for the different data types (data, synth, res)

        Returns:
            * None
        '''
        
        # Check data
        if (type(data) is not list) and (type(data) is str):
            data = [data]

        # X-xaxis
        if axis == 'lon':
            x = self.lon
            xlabel = 'Longitude (deg)'
        elif axis == 'lat':
            x = self.lat
            xlabel = 'Latitude (deg)'
        elif axis == 'x':
            x = self.x
            xlabel = 'Easting (km)'
        elif axis == 'y':
            x = self.y
            xlabel = 'Northing (km)'
        elif axis == 'dist':
            x = self.dist
            xlabel = 'Distance along fault (km)'
        else:
            print('Unkown axis type: {}'.format(axis))
            return
        
        # Make the dictionary of things to plot
        Data = {}
        for dtype,col in zip(data, color):
            Data[dtype] = {}
            
            # Add the values and colors
            if dtype == 'data':
                Data[dtype]['Values'] = self.vel
                Data[dtype]['Color'] = col
            elif dtype == 'synth':
                if self.synth is not None:
                    Data[dtype]['Values'] = self.synth
                    Data[dtype]['Color'] = col
            elif dtype == 'res':
                if self.synth is not None:
                    Data[dtype]['Values'] = self.vel - self.synth
                    Data[dtype]['Color'] = col
            
            # Add the errors
            if dtype == 'data' and np.isfinite(self.err).all():
                Data[dtype]['Error'] = self.err
            if dtype == 'synth' and self.synth_err is not None and np.isfinite(self.synth_err).all():
                Data[dtype]['Error'] = self.synth_err
            if dtype == 'res' and np.isfinite(self.err).all() and self.synth_err is not None and np.isfinite(self.synth_err).all():
                Data[dtype]['Error'] = self.err + self.synth_err

        # Create a figure
        if figsize is None:
            figsize = (10, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(xlabel)

        # Plot the data
        u = np.argsort(x)
        
        for dName in Data:
            values = Data[dName]['Values']
            c = Data[dName]['Color']
            
            ax.plot(x[u], values[u], '.-', color=c, label=dName, markersize=5)
            
            if 'Error' in Data[dName] and error:
                ax.fill_between(x[u], values[u]+Data[dName]['Error'][u], values[u]-Data[dName]['Error'][u], 
                                color=c, alpha=0.3, zorder=1)

        ax.legend()

        # Title
        ax.set_title('{}'.format(self.name))

        # Show
        if show:
            plt.show()
        
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

    def getBlockList(self, blocks, faultboundary=None):
        '''
        For each point, get the blocks in which it is located, or if at a block boundary, get the two blocks at this boundary.
        Otherwise, set the block to None.

        Args:
            * blocks    : list of block objects
            
        Kwargs:
            * faultboundary  : if not None, must be a csi Fault object which is also a block boundary. -
            You need to create a MultiBlock object first and use setBlockFaultBoundaries to set this properly.

        Returns:
            * None
        '''

        # Get the blocks
        self.block_list = []
        
        # If the data are not at a block boundary
        if faultboundary is None:
            
            # Iterate over the data points
            for k, (lon, lat) in enumerate(zip(self.lon, self.lat)):

                for block in blocks:
                
                    if block.PointInBlock(lon, lat):
                        self.block_list.append(block.name)
                        break
            
                # If no block found
                if len(self.block_list) != k+1:
                    self.block_list.append(None)
                
        # If the data are at a block boundary
        else:
            
            if faultboundary.patchType in ('rectangle', 'triangle'):
                
                # We take the bounding blocks from the closest patch at the surface
                
                if not hasattr(faultboundary, 'patchblock'):
                    raise ValueError('The faultboundary object must have the attribute patchblock, please use setBlockFaultBoundaries on your MultiBlock object.')
        
                # Get the indexes of the patches that are at the surface
                zeroD = np.where([0 in p[:, 2] for p in faultboundary.patch])[0]
                if len(zeroD) == 0:
                    print('No surface patches.')

                # Patch center geometry
                patch_geometry = np.array([faultboundary.getpatchgeometry(p, center=True) for p in faultboundary.patch])
                x = patch_geometry[zeroD, 0]
                y = patch_geometry[zeroD, 1]
                z = patch_geometry[zeroD, 2]
                
                # Iterate over the data points
                for k, (lon, lat) in enumerate(zip(self.lon, self.lat)):
                
                    # Get the closest patch which is at the surface
                    xd, yd = faultboundary.ll2xy(lon, lat)
                    zd = 0.
                    dist = np.sqrt((x-xd)**2 + (y-yd)**2 + (z-zd)**2)
                    idx_patch = np.argmin(dist)
                    
                    self.block_list.append([faultboundary.patchblock[zeroD[idx_patch]][0].name,
                                            faultboundary.patchblock[zeroD[idx_patch]][1].name])
                
                # If no block found
                if len(self.block_list) != k+1:
                    self.block_list.append([None, None])
            
            elif faultboundary.patchType in ('triangletent'):
                
                # We take the bounding block from the closest tent
                
                if not hasattr(faultboundary, 'tentblock'):
                    raise ValueError('The faultboundary object must have the attribute tentblock, please use setBlockFaultBoundaries on your MultiBlock object.')
                
                # Get the indexes of the tents that are at the surface
                zeroD = np.where([t[2] == 0 for t in faultboundary.tent])[0]
                if len(zeroD) == 0:
                    print('No surface tents.')

                # Tent geometry
                x = np.array(faultboundary.tent)[zeroD, 0]
                y = np.array(faultboundary.tent)[zeroD, 1]
                z = np.array(faultboundary.tent)[zeroD, 2]
                
                # Iterate over the data points
                for k, (lon, lat) in enumerate(zip(self.lon, self.lat)):
                
                    # Get the closest tent which is at the surface
                    xd, yd = faultboundary.ll2xy(lon, lat)
                    zd = 0.
                    dist = np.sqrt((x-xd)**2 + (y-yd)**2 + (z-zd)**2)
                    idx_tent = np.argmin(dist)
                    
                    self.block_list.append([faultboundary.tentblock[zeroD[idx_tent]][0].name,
                                            faultboundary.tentblock[zeroD[idx_tent]][1].name])
                
                # If no block found
                if len(self.block_list) != k+1:
                    self.block_list.append([None, None])
            
        # Save
        self.block_list = np.array(self.block_list)
        self.blockboundary = faultboundary
        
        # Check if there are points that are not in any block
        if None in self.block_list:
            print(f"Warning: {np.any(self.block_list == None, axis=1).sum()}/{len(self.lon)} points are not in any block.")

        # All done
        return

#EOF
