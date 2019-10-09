'''
A class that deals with InSAR data, after decimation using VarRes.

Written by R. Jolivet, B. Riel and Z. Duputel, April 2013.
Edited by T. Shreve, June 2019. Edited buildsynth to include pressure sources.
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

class insar(SourceInv):
    '''
    Args:
        * name      : Name of the InSAR dataset.

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
        super(insar,self).__init__(name,
                                   utmzone=utmzone,
                                   ellps=ellps,
                                   lon0=lon0,
                                   lat0=lat0)

        # Initialize the data set
        self.dtype = 'insar'

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize InSAR data set {}".format(self.name))
        self.verbose = verbose

        # Initialize some things
        self.vel = None
        self.synth = None
        self.err = None
        self.lon = None
        self.lat = None
        self.los = None
        self.corner = None
        self.xycorner = None
        self.Cd = None

        # All done
        return

    def mergeInsar(self, sar):
        '''
        Combine the existing data set with another insar object.

        Args:
            * sar     : Instance of the insar class

        Returns:
            * None
        '''

        # Assert we have the same geographic transformation
        assert self.utmzone==sar.utmzone, 'Objects do not have the same \
                geographic transform'
        assert self.lon0==sar.lon0, 'Objects do not have the same \
                geographic transform'
        assert self.lat0==sar.lat0, 'Objects do not have the same \
                geographic transform'

        # Assert everything exists
        if self.err is None:
            self.err = np.array([])
        if self.vel is None:
            self.vel = np.array([])
        if self.lat is None:
            self.lat = np.array([])
        if self.lon is None:
            self.lon = np.array([])
        if self.los is None:
            self.los = np.array([])

        # Assert everything exists in the slave
        assert sar.vel is not None, 'Nothing to merge in...'

        # Add things
        self.err = np.array(self.err.tolist()+sar.err.tolist())
        self.vel = np.array(self.vel.tolist()+sar.vel.tolist())
        self.lat = np.array(self.lat.tolist()+sar.lat.tolist())
        self.lon = np.array(self.lon.tolist()+sar.lon.tolist())
        self.los = np.array(self.los.tolist()+sar.los.tolist())

        # Convert to xy
        self.x, self.y = self.ll2xy(self.lon, self.lat)

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

        # Reject pixels
        self.reject_pixel(uVel)

        # All done
        return

    def checkNaNs(self):
        '''
        Checks and remove data points that have NaNs in vel, err, lon, lat or los.
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
        uRemove = np.concatenate((uVel, uErr, uLon, uLat, uLos))

        # Reject pixels
        self.reject_pixel(uRemove)

        # All done
        return

    def read_from_ascii_simple(self, filename, factor=1.0, step=0.0, header=0, los=None):
        '''
        Read the InSAR data from an ascii file with 3 cols.

        Args:
            * filename      : Name of the input file (format is Lon, Lat. data)

        Kwargs:
            * factor        : Factor to multiply the LOS velocity.
            * step          : Add a value to the velocity.
            * header        : Size of the header.
            * los           : LOS unit vector (3 column array)

        Returns:
            * None
        '''

        # Open the file
        fin = open(filename, 'r')

        # Read it all
        Lines = fin.readlines()
        fin.close()

        # Initialize the business
        self.vel = []
        self.err = []
        self.lon = []
        self.lat = []

        # Loop over yje lines
        for i in range(len(Lines)):
            # Get values
            line = Lines[i].split()
            # Fill in the values
            self.lon.append(np.float(line[0]))
            self.lat.append(np.float(line[1]))
            self.vel.append(np.float(line[2]))
            if len(line)>3:
                self.err.append(np.float(line[3]))
            else:
                self.err.append(0.0)

        # Make arrays
        self.vel = (np.array(self.vel)+step)*factor
        self.err = np.array(self.err)*factor
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)

        # set lon to (0, 360.)
        self._checkLongitude()

        # Compute lon lat to utm
        self.x, self.y = self.ll2xy(self.lon,self.lat)

        # store the factor
        self.factor = factor

        # LOS
        if los is not None:
            self.los = []
            fin = open(los, 'r')
            Lines = fin.readlines()
            fin.close()
            for line in Lines:
                line = line.split()
                self.los.append([float(line[0]), float(line[1]), float(line[2])])
            self.los = np.array(self.los)
        else:
            self.los = None

        # All done
        return

    def read_from_ascii(self, filename, factor=1.0, step=0.0, header=0):
        '''
        Read the InSAR data from an ascii file.

        Args:
            * filename      : Name of the input file. Format is Lon, Lat, data, uncertainty, los E, los N, los U.

        Kwargs:
            * factor        : Factor to multiply the LOS velocity.
            * step          : Add a value to the velocity.
            * header        : Size of the header.

        Returns:
            * None
        '''

        # Open the file
        fin = open(filename, 'r')

        # Read it all
        Lines = fin.readlines()
        fin.close()

        # Initialize the business
        self.vel = []
        self.lon = []
        self.lat = []
        self.err = []
        self.los = []
        self.corner = []

        # Loop over yje lines
        for i in range(header,len(Lines)):
            # Get values
            line = Lines[i].split()
            # Fill in the values
            self.lon.append(np.float(line[0]))
            self.lat.append(np.float(line[1]))
            self.vel.append(np.float(line[2]))
            self.err.append(np.float(line[3]))
            self.los.append([np.float(line[4]), np.float(line[5]), np.float(line[6])])

        # Make arrays
        self.vel = (np.array(self.vel)+step)*factor
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.err = np.array(self.err)*factor
        self.los = np.array(self.los)

        # set lon to (0, 360.)
        self._checkLongitude()

        # Compute lon lat to utm
        self.x, self.y = self.ll2xy(self.lon,self.lat)

        # store the factor
        self.factor = factor

        # All done
        return

    def read_from_varres(self,filename, factor=1.0, step=0.0, header=2, cov=False):
        '''
        Read the InSAR LOS rates from the VarRes output.

        Args:
            * filename      : Name of the input file. Two files are opened filename.txt and filename.rsp.

        Kwargs:
            * factor        : Factor to multiply the LOS velocity.
            * step          : Add a value to the velocity.
            * header        : Size of the header.
            * cov           : Read an additional covariance file (binary float32, Nd*Nd elements).

        Returns:
            * None
        '''

        if self.verbose:
            print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename+'.txt','r')
        fsp = open(filename+'.rsp','r')

        # Read it all
        A = fin.readlines()
        B = fsp.readlines()

        # Initialize the business
        self.vel = []
        self.lon = []
        self.lat = []
        self.err = []
        self.los = []
        self.corner = []

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            tmp = A[i].split()
            self.vel.append(np.float(tmp[5]))
            self.lon.append(np.float(tmp[3]))
            self.lat.append(np.float(tmp[4]))
            self.err.append(np.float(tmp[6]))
            self.los.append([np.float(tmp[8]), np.float(tmp[9]), np.float(tmp[10])])
            tmp = B[i].split()
            self.corner.append([np.float(tmp[6]), np.float(tmp[7]), np.float(tmp[8]), np.float(tmp[9])])

        # Make arrays
        self.vel = (np.array(self.vel)+step)*factor
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.err = np.array(self.err)*np.abs(factor)
        self.los = np.array(self.los)
        self.corner = np.array(self.corner)

        # Close file
        fin.close()
        fsp.close()

        # set lon to (0, 360.)
        #self._checkLongitude()

        # Compute lon lat to utm
        self.x, self.y = self.ll2xy(self.lon,self.lat)

        # Compute corner to xy
        self.xycorner = np.zeros(self.corner.shape)
        x, y = self.ll2xy(self.corner[:,0], self.corner[:,1])
        self.xycorner[:,0] = x
        self.xycorner[:,1] = y
        x, y = self.ll2xy(self.corner[:,2], self.corner[:,3])
        self.xycorner[:,2] = x
        self.xycorner[:,3] = y

        # Read the covariance
        if cov:
            nd = self.vel.size
            self.Cd = np.fromfile(filename+'.cov', dtype=np.float32).reshape((nd, nd))*factor*factor

        # Store the factor
        self.factor = factor

        # All done
        return

    def read_from_binary(self, data, lon, lat, err=None, factor=1.0,
                               step=0.0, incidence=None, heading=None, azimuth=None, los=None,
                               dtype=np.float32, remove_nan=True, downsample=1,
                               remove_zeros=True):
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
            * incidence     : incidence angle (degree)
            * heading       : heading angle (degree)
            * azimuth       : Azimuth angle (degree)
            * los           : LOS unit vector 3 component array (3-column array)
            * dtype         : data type (default is np.float32 if data is a file)
            * remove_nan    : True/False
            * downsample    : default is 1 (take one pixel out of those)
            * remove_zeros  : True/False

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
            lon = lon[::downsample]

        # Get the lat
        if type(lat) is str:
            lat = np.fromfile(lat, dtype=dtype)[::downsample]
        else:
            lat = lat[::downsample]

        # Check sizes
        assert vel.shape==lon.shape, 'Something wrong with the sizes: {} {} {} '.format(vel.shape, lon.shape, lat.shape)
        assert vel.shape==lat.shape, 'Something wrong with the sizes: {} {} {} '.format(vel.shape, lon.shape, lat.shape)

        # Get the error
        if err is not None:
            if type(err) is str:
                err = np.fromfile(err, dtype=dtype)[::downsample]
            err = err * np.abs(factor)
            assert vel.shape==err.shape, 'Something wrong with the sizes: {} {} {} '.format(vel.shape, lon.shape, lat.shape)

        # If zeros
        if remove_zeros:
            iZeros = np.flatnonzero(np.logical_or(vel!=0.,
                                                  lon!=0.,
                                                  lat!=0.))
        else:
            iZeros = range(len(vel))

        # Check NaNs
        if remove_nan:
            iFinite = np.flatnonzero(np.isfinite(vel))
        else:
            iFinite = range(len(vel))

        # Compute the LOS
        if heading is not None:
            if type(incidence) is np.ndarray:
                self.inchd2los(incidence, heading, origin='binaryfloat')
                self.los = self.los[::downsample,:]
            elif type(incidence) in (float, np.float):
                self.inchd2los(incidence, heading, origin='float')
            elif type(incidence) is str:
                self.inchd2los(incidence, heading, origin='binary')
                self.los = self.los[::downsample,:]
        elif azimuth is not None:
            if type(incidence) is np.ndarray:
                self.incaz2los(incidence, azimuth, origin='binaryfloat',
                        dtype=dtype)
                self.los = self.los[::downsample,:]
            elif type(incidence) in (float, np.float):
                self.incaz2los(incidence, azimuth, origin='float')
            elif type(incidence) is str:
                self.incaz2los(incidence, azimuth, origin='binary',
                                dtype=dtype)
                self.los = self.los[::downsample,:]
        elif los is not None:
            if type(los) is np.ndarray:
                self.los = los[::downsample,:]
            elif type(los) is str:
                self.los = np.fromfile(los, 'f').reshape((len(vel), 3))
        else:
            self.los = None

        # Who to keep
        iKeep = np.intersect1d(iZeros, iFinite)

        # Remove unwanted pixels
        vel = vel[iKeep]
        if err is not None:
            err = err[iKeep]
        lon = lon[iKeep]
        lat = lat[iKeep]
        if self.los is not None:
            self.los = self.los[iKeep,:]

        # Set things in self
        self.vel = vel
        if err is not None:
            self.err = err
        else:
            self.err = None
        self.lon = lon
        self.lat = lat

        # Keep track of factor
        self.factor = factor

        # set lon to (0, 360.)
        self._checkLongitude()

        # compute x, y
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # All done
        return

    def read_from_mat(self, filename, factor=1.0, step=0.0, incidence=35.88, heading=-13.115):
        '''
        Reads velocity map from a mat file.

        Args:
            * filename  : Name of the input matlab file

        Kwargs:
            * factor    : scale by a factor.
            * step      : add a step.
            * incidence : incidence angle (degree)
            * heading   : heading angle (degree)

        Returns:
            * None
        '''

        # Initialize values
        self.vel = []
        self.lon = []
        self.lat = []
        self.err = []
        self.los = []

        # Open the input file
        import scipy.io as scio
        A = scio.loadmat(filename)

        # Get the phase values
        self.vel = (A['velo'].flatten()+ step)*factor
        self.err = A['verr'].flatten()
        self.err[np.where(np.isnan(self.vel))] = np.nan
        self.vel[np.where(np.isnan(self.err))] = np.nan

        # Deal with lon/lat
        Lon = A['posx'].flatten()
        Lat = A['posy'].flatten()
        Lon,Lat = np.meshgrid(Lon,Lat)
        w,l = Lon.shape
        self.lon = Lon.reshape((w*l,)).flatten()
        self.lat = Lat.reshape((w*l,)).flatten()

        # Keep the non-nan pixels
        u = np.flatnonzero(np.isfinite(self.vel))
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.vel = self.vel[u]
        self.err = self.err[u]

        # set lon to (0, 360.)
        self._checkLongitude()

        # Convert to utm
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # Deal with the LOS
        self.inchd2los(incidence, heading)

        # Store the factor
        self.factor = factor

        # All done
        return

    def incaz2los(self, incidence, azimuth, origin='onefloat', dtype=np.float32):
        '''
        From the incidence and the heading, defines the LOS vector.

        Args:
            * incidence : Incidence angle.
            * azimuth   : Azimuth angle of the LOS

        Kwargs:
            * origin    : What are these numbers
                - onefloat      : One number
                - grd           : grd files
                - binary        : Binary files
                - binaryfloat   : Arrays of float
            * dtype     : Data type (default is np.float32)

        Returns:
            * None

        '''

        # Save values
        self.incidence = incidence
        self.azimuth = azimuth

        # Read the files if needed
        if origin in ('grd', 'GRD'):
            try:
                from netCDF4 import Dataset as netcdf
                fincidence = netcdf(incidence, 'r', format='NETCDF4')
                fazimuth = netcdf(azimuth, 'r', format='NETCDF4')
            except:
                import scipy.io.netcdf as netcdf
                fincidence = netcdf.netcdf_file(incidence)
                fazimuth = netcdf.netcdf_file(azimuth)
            incidence = np.array(fincidence.variables['z'][:]).flatten()
            azimuth = np.array(fazimuth.variables['z'][:]).flatten()
            self.origininchd = origin
        elif origin in ('binary', 'bin'):
            incidence = np.fromfile(incidence, dtype=dtype)
            azimuth = np.fromfile(azimuth, dtype=dtype)
            self.origininchd = origin
        elif origin in ('binaryfloat'):
            self.origininchd = origin

        self.Incidence = incidence
        self.Azimuth = azimuth

        # Convert angles
        alpha = -1.0*azimuth*np.pi/180.
        phi = incidence*np.pi/180.

        # Compute LOS
        Se = np.sin(alpha) * np.sin(phi)
        Sn = np.cos(alpha) * np.sin(phi)
        Su = np.cos(phi)

        # Store it
        if origin in ('grd', 'GRD', 'binary', 'bin', 'binaryfloat'):
            self.los = np.ones((alpha.shape[0],3))
        else:
            self.los = np.ones((self.lon.shape[0],3))
        self.los[:,0] *= Se
        self.los[:,1] *= Sn
        self.los[:,2] *= Su

        # all done
        return

    def inchd2los(self, incidence, heading, origin='onefloat'):
        '''
        From the incidence and the heading, defines the LOS vector.

        Args:
            * incidence : Incidence angle.
            * heading   : Heading angle.

        Kwargs:
            * origin    : What are these numbers
                - onefloat      : One number
                - grd           : grd files
                - binary        : Binary files
                - binaryfloat   : Arrays of float

        Returns:
            * None
        '''

        # Save values
        self.incidence = incidence
        self.heading = heading

        # Read the files if needed
        if origin in ('grd', 'GRD'):
            try:
                from netCDF4 import Dataset as netcdf
                fincidence = netcdf(incidence, 'r', format='NETCDF4')
                fheading = netcdf(heading, 'r', format='NETCDF4')
            except:
                import scipy.io.netcdf as netcdf
                fincidence = netcdf.netcdf_file(incidence)
                fheading = netcdf.netcdf_file(heading)
            incidence = np.array(fincidence.variables['z'][:]).flatten()
            heading = np.array(fheading.variables['z'][:]).flatten()
            self.origininchd = origin
        elif origin in ('binary', 'bin'):
            incidence = np.fromfile(incidence, dtype=np.float32)
            heading = np.fromfile(heading, dtype=np.float32)
            self.origininchd = origin
        elif origin in ('binaryfloat'):
            self.origininchd = origin

        self.Incidence = incidence
        self.Heading = heading

        # Convert angles
        alpha = (heading+90.)*np.pi/180.
        phi = incidence *np.pi/180.

        # Compute LOS
        Se = -1.0 * np.sin(alpha) * np.sin(phi)
        Sn = -1.0 * np.cos(alpha) * np.sin(phi)
        Su = np.cos(phi)

        # Store it
        if origin in ('grd', 'GRD', 'binary', 'bin', 'binaryfloat'):
            self.los = np.ones((alpha.shape[0],3))
        else:
            self.los = np.ones((self.lon.shape[0],3))
        self.los[:,0] *= Se
        self.los[:,1] *= Sn
        self.los[:,2] *= Su

        # all done
        return

    def read_from_grd(self, filename, factor=1.0, step=0.0, incidence=None, heading=None,
                      los=None, keepnans=False):
        '''
        Reads velocity map from a grd file.

        Args:
            * filename  : Name of the input file

        Kwargs:
            * factor    : scale by a factor
            * step      : add a value.
            * incidence : incidence angle (degree)
            * heading   : heading angle (degree)
            * los       : LOS unit vector (3 column array)
            * keepnans  : True/False

        Returns:
            * None
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Initialize values
        self.vel = []
        self.lon = []
        self.lat = []
        self.err = []
        self.los = []

        # Open the input file
        try:
            from netCDF4 import Dataset as netcdf
            fin = netcdf(filename, 'r', format='NETCDF4')
        except ImportError:
            import scipy.io.netcdf as netcdf
            fin = netcdf.netcdf_file(filename)

        # Get the values
        if len(fin.variables['z'].shape)==1:
            self.vel = (np.array(fin.variables['z'][:]) + step) * factor
        else:
            self.vel = (np.array(fin.variables['z'][:,:]).flatten() + step)*factor
        self.err = np.zeros((self.vel.shape))
        self.err[np.where(np.isnan(self.vel))] = np.nan
        self.vel[np.where(np.isnan(self.err))] = np.nan

        # Deal with lon/lat
        if 'x' in fin.variables.keys():
            Lon = fin.variables['x'][:]
            Lat = fin.variables['y'][:]
        elif 'lon' in fin.variables.keys():
            Lon = fin.variables['lon'][:]
            Lat = fin.variables['lat'][:]
        else:
            Nlon, Nlat = fin.variables['dimension'][:]
            Lon = np.linspace(fin.variables['x_range'][0], fin.variables['x_range'][1], Nlon)
            Lat = np.linspace(fin.variables['y_range'][1], fin.variables['y_range'][0], Nlat)
        self.lonarr = Lon.copy()
        self.latarr = Lat.copy()
        Lon, Lat = np.meshgrid(Lon,Lat)
        w, l = Lon.shape
        self.lon = Lon.reshape((w*l,)).flatten()
        self.lat = Lat.reshape((w*l,)).flatten()
        self.grd_shape = Lon.shape

        # Keep the non-nan pixels only
        if not keepnans:
            u = np.flatnonzero(np.isfinite(self.vel))
            self.lon = self.lon[u]
            self.lat = self.lat[u]
            self.vel = self.vel[u]
            self.err = self.err[u]

        # set lon to (0, 360.)
        self._checkLongitude()

        # Convert to utm
        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # Deal with the LOS
        if heading is not None and incidence is not None and los is None:
            if type(heading) is str:
                ori = 'grd'
            else:
                ori = 'float'
            self.inchd2los(incidence, heading, origin=ori)
            if not keepnans and self.los.shape[0]!=self.lon.shape[0]:
                self.los = self.los[u,:]
        elif los is not None:
            # If strings, they are meant to be grd files
            if type(los[0]) is str:
                if los[0][-4:] not in ('.grd'):
                    print('LOS input files do not seem to be grds as the displacement file')
                    print('There might be some issues...')
                    print('      Input files: {}, {} and {}'.format(los[0], los[1], los[2]))
                try:
                    from netCDF4 import Dataset
                    finx = Dataset(los[0], 'r', format='NETCDF4')
                    finy = Dataset(los[1], 'r', format='NETCDF4')
                    finz = Dataset(los[2], 'r', format='NETCDF4')
                except ImportError:
                    import scipy.io.netcdf as netcdf
                    finx = netcdf.netcdf_file(los[0])
                    finy = netcdf.netcdf_file(los[1])
                    finz = netcdf.netcdf_file(los[2])
                losx = np.array(finx.variables['z'][:,:]).flatten()
                losy = np.array(finy.variables['z'][:,:]).flatten()
                losz = np.array(finz.variables['z'][:,:]).flatten()
                # Remove NaNs?
                if not keepnans:
                    losx = losx[u]
                    losy = losy[u]
                    losz = losz[u]
                # Do as if binary
                losList = [losx, losy, losz]
            # Store these guys
            self.los = np.zeros((len(losx),3))
            self.los[:,0] = losx
            self.los[:,1] = losy
            self.los[:,2] = losz

        else:
            print('Warning: not enough information to compute LOS')
            print('LOS will be set to 1,0,0')
            self.los = np.zeros((len(self.vel),3))
            self.los[:,0] = 1.0
            self.los[:,1] = 0.0
            self.los[:,2] = 0.0

        # Store the factor
        self.factor = factor

        # All done
        return

    def ModelResolutionDownsampling(self, faults, threshold, damping, startingsize=10., minimumsize=0.5, tolerance=0.1, plot=False):
        '''
        Downsampling algorythm based on Lohman & Simons, 2005, G3.

        Args:
            * faults        : List of faults, these need to have a buildGFs routine (ex: for RectangularPatches, it will be Okada).
            * threshold     : Resolution threshold, if above threshold, keep dividing.
            * damping       : Damping parameter. Damping is enforced through the addition of a identity matrix.

        Kwargs:
            * startingsize  : Starting size of the downsampling boxes.
            * minimumsize   : Minimum window size (km)
            * tolerance     : Tolerance on the window size calculation
            * plot          : True/False

        Returns:
            * None
        '''

        # If needed
        from .imagedownsampling import imagedownsampling

        # Check if faults have patches and builGFs routine
        for fault in faults:
            assert (hasattr(fault, 'builGFs')), 'Fault object {} does not have a buildGFs attribute...'.format(fault.name)

        # Create the insar downsampling object
        downsampler = imagedownsampling('Downsampler {}'.format(self.name), self, faults)

        # Initialize the downsampling starting point
        downsampler.initialstate(startingsize, minimumsize, tolerance=tolerance)

        # Iterate until done
        downsampler.ResolutionBasedIterations(threshold, damping, plot=False)

        # Plot
        if plot:
            downsampler.plot()

        # Write outputs
        downsampler.writeDownsampled2File(self.name, rsp=True)

        # All done
        return

    def buildDiagCd(self):
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

    def buildCd(self, sigma, lam, function='exp', diagonalVar=False,
                normalizebystd=False):
        '''
        Builds the full Covariance matrix from values of sigma and lambda.

        If function='exp':

            :math:`C_d(i,j) = \sigma^2  e^{-\\frac{d[i,j]}{\lambda}}`

        elif function='gauss':

            :math:`C_d(i,j) = \sigma^2 e^{-\\frac{d_{i,j}^2}{2*\lambda}}`

        Args:
            * sigma             : Sigma term of the covariance
            * lam               : Caracteristic length of the covariance

        Kwargs:
            * function          : Can be 'gauss' or 'exp'
            * diagonalVar       : Substitute the diagonal by the standard deviation of the measurement squared
            * normalizebystd    : Weird option to normalize the covariance matrix by the std deviation

        Returns:
            * None
        '''

        # Assert
        assert function in ('exp', 'gauss'), \
                'Unknown functional form for Covariance matrix'

        # Check something
        if normalizebystd:
            diagonalVar = True

        # Get some size
        nd = self.vel.shape[0]

        # positions
        x = self.x
        y = self.y
        distance = np.sqrt( (x[:,None] - x[None,:])**2 + (y[:,None] - y[None,:])**2)

        # Compute Cd
        if function is 'exp':
            self.Cd = sigma*sigma*np.exp(-1.0*distance/lam)
        elif function is 'gauss':
            self.Cd = sigma*sigma*np.exp(-1.0*distance*distance/(2*lam))

        # Normalize
        if normalizebystd:
            for i in range(nd):
                for j in range(i,nd):
                    self.Cd[j,i] *= self.err[j]*self.err[i]/(sigma*sigma)
                    self.Cd[i,j] *= self.err[j]*self.err[i]/(sigma*sigma)

        # Substitute variance?
        if diagonalVar:
            for i in range(nd):
                self.Cd[i,i] = self.err[i]**2

        # All done
        return

    def distancePixel2Pixel(self, i, j):
        '''
        Returns the distance in km between two pixels.

        Args:
            * i     : index of a pixel
            * h     : index of a pixel

        Returns:
            * float
        '''

        # Get values
        x1 = self.x[i]
        y1 = self.y[i]
        x2 = self.x[j]
        y2 = self.y[j]

        # Compute the distance
        d = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # All done
        return d

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

    def returnAverageNearPoint(self, lon, lat, distance):
        '''
        Returns the phase value, the los and the errors averaged over a distance
        from a point (lon, lat).

        Args:
            * lon       : longitude of the point
            * lat       : latitude of the point
            * distance  : distance around the point

        Returns:
            * float, float, tuple
        '''

        # Get the distances
        distances = self.distance2point(lon, lat)

        # Get the indexes
        u = np.flatnonzero(distances<=distance)

        # return the values
        if len(u)>0:
            vel = np.nanmean(self.vel[u])
            err = np.nanstd(self.vel[u])
            los = np.nanmean(self.los[u,:], axis=0)
            los /= np.linalg.norm(los)
            return vel, err, los
        else:
            return None, None, None

    def extractAroundGPS(self, gps, distance, doprojection=True):
        '''
        Returns a gps object with values projected along the LOS around the
        gps stations included in gps. In addition, it projects the gps displacements
        along the LOS

        Args:
            * gps           : gps or gpstimeseries object
            * distance      : distance to consider around the stations

        Kwargs:
            * doprojection  : Projects the gps enu disp into the los as well

        Retunrs:
            * gps instance
        '''

        # Create a gps object
        out = copy.deepcopy(gps)

        # Create a holder in the new gps object
        out.vel_los = []
        out.err_los = []
        out.los = []

        # Iterate over the stations
        for lon, lat in zip(out.lon, out.lat):
            vel, err, los = self.returnAverageNearPoint(lon, lat, distance)
            out.vel_los.append(vel)
            out.err_los.append(err)
            out.los.append(los)

        # Convert to arrays
        out.vel_los = np.array(out.vel_los)
        out.err_los = np.array(out.err_los)
        out.los = np.array(out.los)

        # Do a projection
        if doprojection:
            gps.project2InSAR(out.los)

        # All done
        return out

    def select_pixels(self, minlon, maxlon, minlat, maxlat):
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
        self.keepPixels(u)

        # All done
        return

    def keepPixels(self, u):
        '''
        Keep the pixels indexed u and ditch the other ones

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
        if self.corner is not None:
            self.corner = self.corner[u,:]
            self.xycorner = self.xycorner[u,:]

        # Deal with the covariance matrix
        if self.Cd is not None:
            Cdt = self.Cd[u,:]
            self.Cd = Cdt[:,u]

        # All done
        return

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
        if fault.type is "Fault":
            # Get the values
            try:
                GssLOS = G['strikeslip']
            except:
                GssLOS = None
            try:
                GdsLOS = G['dipslip']
            except:
                GdsLOS = None
            try:
                GtsLOS = G['tensile']
            except:
                GtsLOS = None
            try:
                GcpLOS = G['coupling']
            except:
                GcpLOS = None

            # set the GFs
            fault.setGFs(self, strikeslip=[GssLOS], dipslip=[GdsLOS], tensile=[GtsLOS],
                        coupling=[GcpLOS], vertical=True)
        elif fault.type is "Pressure":
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

            fault.setGFs(self, deltapressure=[GpLOS], GDVx=[GdvxLOS] , GDVy=[GdvyLOS], GDVz =[GdvzLOS], vertical=True)

        # All done
        return


    def setTransformNormalizingFactor(self, x0, y0, normX, normY, base):
        '''
        Set orbit normalizing factors in insar object.

        Args:
            * x0        : Reference point x coordinate
            * y0        : Reference point y coordinate
            * normX     : Normalization distance along x
            * normY     : Normalization distance along y
            * base      : baseline

        Returns:
            * None
        '''

        self.TransformNormalizingFactor = {}
        self.TransformNormalizingFactor['x'] = normX
        self.TransformNormalizingFactor['y'] = normY
        self.TransformNormalizingFactor['ref'] = [x0, y0]
        self.TransformNormalizingFactor['base'] = base

        # All done
        return


    def computeTransformNormalizingFactor(self):
        '''
        Compute orbit normalizing factors and store them in insar object.

        Returns:
            * None
        '''

        x0 = np.mean(self.x)
        y0 = np.mean(self.y)
        normX = np.abs(self.x - x0).max()
        normY = np.abs(self.y - y0).max()
        base_max = np.max([normX, normY])
        #print(self.x,self.y)
        print('normalizing factors are ', x0,y0,normX,normY)
        self.TransformNormalizingFactor = {}
        self.TransformNormalizingFactor['x'] = normX
        self.TransformNormalizingFactor['y'] = normY
        self.TransformNormalizingFactor['ref'] = [x0, y0]
        self.TransformNormalizingFactor['base'] = base_max
        # All done
        return

    def getTransformEstimator(self, trans, computeNormFact=True):
        '''
        Returns the Estimator for the transformation to estimate in the InSAR data.

        Args:
            * trans     : Transformation type
                - 1: constant offset to the data
                - 3: constant and linear function of x and y
                - 4: constant, linear term and cross term.
                - strain: Estimates an aerial strain tensor

        Kwargs:
            * computeNormFact   : Recompute the normalization factor

        Returns:
            * None
        '''

        # Several cases
        if type(trans) is int:
            T = self.getPolyEstimator(trans, computeNormFact=computeNormFact)
        elif type(trans) is str:
            T = self.get2DstrainEst(computeNormFact=computeNormFact)

        # All done
        return T

    def get2DstrainEst(self, computeNormFact=True):
        '''
        Returns the matrix to estimate the 2d aerial strain tensor. When building the estimator, third column is the Epsilon_xx component, Fourth column is the Epsilon_xy component, fifth column is the Epsilon_yy component.

        Kwargs:
            * computeNormFact       : Recompute the normalization factor

        Returns:
            * None
        '''

        # Get the number of gps stations
        ns = self.vel.shape[0]

        # Parameter size
        nc = 3

        # Get the reference
        if computeNormFact:
            self.computeTransformNormalizingFactor()
        x0,y0 = self.TransformNormalizingFactor['ref']
        base = self.TransformNormalizingFactor['base']

        # Compute the baselines
        base_x = (self.x - x0)/base
        base_y = (self.y - y0)/base

        # Store the normalizing factor
        self.StrainNormalizingFactor = base

        # Allocate a Base
        H = np.zeros((2,nc))

        # Allocate the 2 full matrix
        Hfe = np.zeros((ns,nc))
        Hfn = np.zeros((ns,nc))

        # Fill in
        Hfe[:,0] = base_x
        Hfe[:,1] = 0.5*base_y
        Hfn[:,1] = 0.5*base_x
        Hfn[:,2] = base_y

        # multiply by the los
        Hout = self.los[:,0][:,np.newaxis]*Hfe + self.los[:,1][:,np.newaxis]*Hfn

        # All done
        return Hout

    def getPolyEstimator(self, ptype, computeNormFact=True):
        '''
        Returns the Estimator for the polynomial form to estimate in the InSAR data.

        Args:
            * ptype: integer
                - 1: constant offset to the data
                - 3: constant and linear function of x and y
                - 4: constant, linear term and cross term.

        Kwargs:
            * computeNormFact : bool

        Returns:
            * None
        '''

        # number of data points
        nd = self.vel.shape[0]

        # Create the Estimator
        orb = np.zeros((nd, ptype))
        if ptype > 0.0:
            orb[:,0] = 1.0

        if ptype >= 3:
            # Compute normalizing factors
            if computeNormFact:
                self.computeTransformNormalizingFactor()
            else:
                assert hasattr(self, 'TransformNormalizingFactor'), 'You must set TransformNormalizingFactor first'

            normX = self.TransformNormalizingFactor['x']
            normY = self.TransformNormalizingFactor['y']
            x0, y0 = self.TransformNormalizingFactor['ref']

            # Fill in functionals
            orb[:,1] = (self.x - x0) / normX
            orb[:,2] = (self.y - y0) / normY

        if ptype == 4:
            orb[:,3] = orb[:,1] * orb[:,2]

        # Scale everything by the data factor
        orb *= self.factor

        # All done
        return orb

    def computePoly(self, fault, computeNormFact=True):
        '''
        Computes the orbital bias estimated in fault

        Args:
            * fault : Fault object that has a polysol structure.

        Kwargs:
            * computeNormFact   : Recompute the norm factors

        Returns:
            * None
        '''

        # Get the polynomial type
        ptype = fault.poly[self.name]

        # Get the parameters
        params = fault.polysol[self.name]
        if type(params) is dict:
            params = params[ptype]

        # Get the estimator
        Horb = self.getPolyEstimator(ptype, computeNormFact=computeNormFact)

        # Compute the polynomial
        self.orbit = np.dot(Horb, params)

        # All done
        return

    def computeCustom(self, fault):
        '''
        Computes the displacements associated with the custom green's functions.

        Args:
            * fault : A fault instance

        Returns:
            * None
        '''

        # Get GFs and parameters
        G = fault.G[self.name]['custom']
        custom = fault.custom[self.name]

        # Compute
        self.custompred = np.dot(G,custom)

        # All done
        return

    def removePoly(self, fault, verbose=False, custom=False ,computeNormFact=True):
        '''
        Removes a polynomial from the parameters that are in a fault.

        Args:
            * fault     : a fault instance

        Kwargs:
            * verbose           : Show us stuff
            * custom            : include custom green's functions
            * computeNormFact   : recompute norm factors

        Returns:
            * None
        '''

        # compute the polynomial
        self.computePoly(fault,computeNormFact=computeNormFact)

        # Print Something
        if verbose:
            print('Correcting insar {} from polynomial function'.format(self.name))
        # Correct
        self.vel -= self.orbit
        # Correct Custom
        if custom:
            self.computeCustom(fault)
            self.vel -= self.custompred

        # All done
        return

    def removeTransformation(self, fault, verbose=False, custom=False):
        '''
        Wrapper of removePoly to ensure consistency between data sets.

        Args:
            * fault     : a fault instance

        Kwargs:
            * verbose   : talk to us
            * custom    : Remove custom GFs

        Returns:
            * None
        '''

        self.removePoly(fault, verbose=verbose, custom=custom)

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
        self.buildsynth(faults, direction=direction, poly=poly, custom=custom, computeNormFact=computeNormFact)

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
            if fault.type is "Fault":
                # Get the good part of G
                G = fault.G[self.name]

                if ('s' in direction) and ('strikeslip' in G.keys()):
                    Gs = G['strikeslip']
                    Ss = fault.slip[:,0]
                    losss_synth = np.dot(Gs,Ss)
                    self.synth += losss_synth
                if ('d' in direction) and ('dipslip' in G.keys()):
                    Gd = G['dipslip']
                    Sd = fault.slip[:,1]
                    losds_synth = np.dot(Gd, Sd)
                    self.synth += losds_synth
                if ('t' in direction) and ('tensile' in G.keys()):
                    Gt = G['tensile']
                    St = fault.slip[:,2]
                    losop_synth = np.dot(Gt, St)
                    self.synth += losop_synth
                if ('c' in direction) and ('coupling' in G.keys()):
                    Gc = G['coupling']
                    Sc = fault.coupling
                    losdc_synth = np.dot(Gc,Sc)
                    self.synth += losdc_synth

                if custom:
                    Gc = G['custom']
                    Sc = fault.custom[self.name]
                    losdc_synth = np.dot(Gc, Sc)
                    self.synth += losdc_synth

                if poly is not None:
                    # Compute the polynomial
                    self.computePoly(fault,computeNormFact=computeNormFact)
                    if poly is 'include':
                        self.removePoly(fault, computeNormFact=computeNormFact)
                    else:
                        self.synth += self.orbit

            #Loop on each pressure source
            elif fault.type is "Pressure":

                # Get the good part of G
                G = fault.G[self.name]
                if fault.source in {"Mogi", "Yang"}:
                    Gp = G['pressure']
                    Sp = fault.deltapressure/fault.mu
                    print("Scaling by pressure", fault.deltapressure )
                    losdp_synth = Gp*Sp
                    self.synth += losdp_synth

                elif fault.source is ("pCDM"):
                    Gdx = G['pressureDVx']
                    Sxp = fault.DVx/fault.scale
                    lossx_synth = np.dot(Gdx,Sxp)
                    self.synth += lossx_synth
                    Gdy = G['pressureDVy']
                    Syp = fault.DVy/fault.scale
                    lossy_synth = np.dot(Gdy, Syp)
                    self.synth += lossy_synth
                    Gdz = G['pressureDVz']
                    Szp = fault.DVz/fault.scale
                    lossz_synth = np.dot(Gdz, Szp)
                    self.synth += lossz_synth

                elif fault.source is ("CDM"):
                    Gp = G['pressure']
                    Sp = fault.deltaopening
                    print("Scaling by opening")
                    losdp_synth = Gp*Sp
                    self.synth += losdp_synth


                if custom:
                    Gc = G['custom']
                    Sc = fault.custom[self.name]
                    losdc_synth = np.dot(Gc, Sc)
                    self.synth += losdc_synth

                if poly is not None:
                    # Compute the polynomial
                    self.computePoly(fault,computeNormFact=computeNormFact)
                    if poly is 'include':
                        self.removePoly(fault, computeNormFact=computeNormFact)
                    else:
                        self.synth += self.orbit


        # All done
        return

    def writeEDKSdata(self):
        '''
        ***Obsolete***
        '''

        # Get the x and y positions
        x = self.x
        y = self.y

        # Get LOS informations
        los = self.los

        # Open the file
        datname = self.name.replace(' ','_')
        filename = 'edks_{}.idEN'.format(datname)
        fout = open(filename, 'w')

        # Write a header
        fout.write("id E N E_los N_los U_los\n")

        # Loop over the data locations
        for i in range(len(x)):
            string = '{:5d} {} {} {} {} {} \n'.format(i, x[i], y[i], los[i,0], los[i,1], los[i,2])
            fout.write(string)

        # Close the file
        fout.close()

        # All done
        return datname,filename

    def reject_pixel(self, u):
        '''
        Reject pixels.

        Args:
            * u         : Index of the pixel to reject.

        Returns:
            * None
        '''

        self.lon = np.delete(self.lon, u)
        self.lat = np.delete(self.lat, u)
        self.x = np.delete(self.x, u)
        self.y = np.delete(self.y, u)
        if self.err is not None:
            self.err = np.delete(self.err, u)
        self.los = np.delete(self.los, u, axis=0)
        self.vel = np.delete(self.vel, u)

        if self.Cd is not None:
            self.Cd = np.delete(self.Cd, u, axis=0)
            self.Cd = np.delete(self.Cd, u, axis=1)

        if self.corner is not None:
            self.corner = np.delete(self.corner, u, axis=0)
            self.xycorner = np.delete(self.xycorner, u, axis=0)

        if self.synth is not None:
            self.synth = np.delete(self.synth, u, axis=0)

        # All done
        return

    def reject_pixels_fault(self, dis, faults):
        '''
        Rejects the pixels that are {dis} km close to the fault.

        Args:
            * dis       : Threshold distance.
            * faults    : list of fault objects.

        Returns:
            * None
        '''

        # Variables to trim are  self.corner,
        # self.xycorner, self.Cd, (self.synth)

        # Check something
        if faults.__class__ is not list:
            faults = [faults]

        # Build a line object with the faults
        fl = []
        for flt in faults:
            f = [[x, y] for x,y in np.vstack((flt.xf, flt.yf)).T.tolist()]
            fl = fl + f

        # Get all the positions
        pp = [[x, y] for x,y in zip(self.x, self.y)]

        # Get distances
        D = scidis.cdist(pp, fl)

        # Get minimums
        d = np.min(D, axis=1)
        del D

        # Find the close ones
        if dis>0.:
            u = np.where(d<=dis)[0]
        else:
            u = np.where(d>=(-1.0*dis))[0]

        # Reject
        self.reject_pixel(u)

        # All done
        return u

    def getprofile(self, name, loncenter, latcenter, length, azimuth, width):
        '''
        Project the SAR velocities onto a profile. Works on the lat/lon coordinates system. Profile is stored in a dictionary called {self}.profile

        Args:
            * name              : Name of the profile.
            * loncenter         : Profile origin along longitude.
            * latcenter         : Profile origin along latitude.
            * length            : Length of profile.
            * azimuth           : Azimuth in degrees.
            * width             : Width of the profile.

        Returns:
            * None
        '''

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.ll2xy(loncenter, latcenter)

        # Get the profile
        Dalong, Dacros, Bol, boxll, box, xe1, ye1, xe2, ye2, \
                lon, lat = utils.coord2prof(self, xc, yc, length, azimuth, width)

        # Get values
        vel = self.vel[Bol]
        if self.synth is not None:
            synth = self.synth[Bol]
        else:
            synth = None
        if self.err is not None:
            err = self.err[Bol]
        else:
            err = None
        if self.los is not None:
            los = self.los[Bol]
        else:
            los = None

        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name]
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['Lon'] = lon
        dic['Lat'] = lat
        dic['LOS Velocity'] = vel
        dic['LOS Synthetics'] = synth
        dic['LOS Error'] = err
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]
        lone1, late1 = self.putm(xe1*1000., ye1*1000., inverse=True)
        lone2, late2 = self.putm(xe2*1000., ye2*1000., inverse=True)
        dic['EndPointsLL'] = [[lone1, late1],
                              [lone2, late2]]
        dic['LOS vector'] = los

        # All done
        return

    def getprofileAlongCurve(self, name, lon, lat, width, widthDir):
        '''
        Project the SAR velocities onto a profile. Works on the lat/lon coordinates system. Has not been tested in a long time...

        Args:
            * name              : Name of the profile.
            * lon               : Longitude of the Line around which we do the profile
            * lat               : Latitude of the Line around which we do the profile
            * width             : Width of the zone around the line.
            * widthDir          : Direction to of the width.

        Returns:
            * None
        '''

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}

        # lonlat2xy
        xl = []
        yl = []
        for i in range(len(lon)):
            x, y = self.ll2xy(lon[i], lat[i])
            xl.append(x)
            yl.append(y)

        # Get the profile
        Dalong, vel, err, Dacros, boxll, xc, yc, xe1, ye1, xe2, ye2, length = self.curve2prof(xl, yl, width, widthDir)

        # get lon lat center
        loncenter, latcenter = self.xy2ll(xc, yc)

        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name]
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['LOS Velocity'] = vel
        dic['LOS Error'] = err
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]
        lone1, late1 = self.putm(xe1*1000., ye1*1000., inverse=True)
        lone2, late2 = self.putm(xe2*1000., ye2*1000., inverse=True)
        dic['EndPointsLL'] = [[lone1, late1],
                              [lone2, late2]]

        # All done
        return

    def referenceProfile(self, name, xmin, xmax, method='mean'):
        '''
        Removes the mean value of points between xmin and xmax. Optionally, can remove the linear best fit between these 2 values

        Args:
            * name      : Name of the profile
            * xmin      : minimum value along X-axis to consider
            * xmax      : Maximum value along X-axis to consider

        Kwargs:
            * method    : 'mean' or 'linear'

        Returns:
            * None
        '''

        # Get the profile
        profile = self.profiles[name]

        # Get the indexes
        ii = self._getindexXlimProfile(name, xmin, xmax)

        # Check
        if len(ii)==0:
            return

        # Get average value
        if method=='mean':
            reference = profile['LOS Velocity'][ii].mean()
        elif method=='linear':
            y = profile['LOS Velocity'][ii]
            x = profile['Distance'][ii]
            P = np.polyfit(x, y, 1)
            reference = profile['Distance']*P[0] + P[1]

        # Set the reference
        profile['LOS Velocity'] -= reference

        # all done
        return

    def cleanProfile(self, name, xlim=None, zlim=None):
        '''
        Cleans a specified profile.

        Args:
            * name      : name of the profile to work with

        Kwargs:
            * xlim      : tuple (xmin, xmax). Removes pixels outside of this box
            * zlim      : tuple (zmin, zmax). Removes pixels outside of this box

        Returns:
            * None
        '''

        # Get profile
        profile = self.profiles[name]

        # Distance cleanup
        if xlim is not None:
            ii = self._getindexXlimProfile(name, xlim[0], xlim[1])
            profile['Distance'] = profile['Distance'][ii]
            profile['LOS Velocity'] = profile['LOS Velocity'][ii]
            profile['Normal Distance'] = profile['Normal Distance'][ii]
            if profile['LOS Error'] is not None:
                profile['LOS Error'] = profile['LOS Error'][ii]

        # Amplitude cleanup
        if zlim is not None:
            ii = self._getindexZlimProfile(name, zlim[0], zlim[1])
            profile['Distance'] = profile['Distance'][ii]
            profile['LOS Velocity'] = profile['LOS Velocity'][ii]
            profile['Normal Distance'] = profile['Normal Distance'][ii]
            if profile['LOS Error'] is not None:
                profile['LOS Error'] = profile['LOS Error'][ii]

        return

    def smoothProfile(self, name, window, method='mean'):
        '''
        Computes smoothed profile but running a mean or median window on the profile.

        Args:
            * name      : Name of the profile to work on
            * window    : Width of the window (km)

        Kwargs:
            * method    : 'mean' or 'median'

        Returns:
            * None
        '''

        # Get profile
        dis = self.profiles[name]['Distance']
        vel = self.profiles[name]['LOS Velocity']
        los = self.profiles[name]['LOS vector']

        # Create the bins
        bins = np.arange(dis.min(), dis.max(), window)
        indexes = np.digitize(dis, bins)

        # Create Lists
        outvel = []
        outerr = []
        outdis = []
        if los is not None:
            outlos = []

        # Run a runing average on it
        for i in range(len(bins)-1):

            # Find the guys inside this bin
            uu = np.flatnonzero(indexes==i)

            # If there is points in this bin
            if len(uu)>0:

                # Get the mean
                if method in ('mean'):
                    m = vel[uu].mean()
                elif method in ('median'):
                    m = np.median(vel[uu])

                # Get the mean distance
                d = dis[uu].mean()

                # Get the error
                e = vel[uu].std()

                # Set it
                outvel.append(m)
                outerr.append(e)
                outdis.append(d)

                # Get the LOS
                if los is not None:
                    l = los[uu,:].mean(axis=0)
                    outlos.append(l)

        # Copy the old profile and modify it
        newName = 'Smoothed {}'.format(name)
        self.profiles[newName] = copy.deepcopy(self.profiles[name])
        self.profiles[newName]['LOS Velocity'] = np.array(outvel)
        self.profiles[newName]['LOS Error'] = np.array(outerr)
        self.profiles[newName]['Distance'] = np.array(outdis)
        if los is not None:
            self.profiles[newName]['LOS vector'] = np.array(outlos)

        # All done
        return

    def _getindexXlimProfile(self, name, xmin, xmax):
        '''
        Returns the index of the points that are in between xmin & xmax.
        '''

        # Get the distance array
        distance = self.profiles[name]['Distance']

        # Get the indexes
        ii = np.flatnonzero(distance>=xmin)
        jj = np.flatnonzero(distance<=xmax)
        uu = np.intersect1d(ii,jj)

        # All done
        return uu

    def _getindexZlimProfile(self, name, zmin, zmax):
        '''
        Returns the index of the points that are in between zmin & zmax.
        '''

        # Get the velocity
        velocity = self.profiles[name]['LOS Velocity']

        # Get the indexes
        ii = np.flatnonzero(velocity>=zmin)
        jj = np.flatnonzero(velocity<=zmax)
        uu = np.intersect1d(ii,jj)

        # All done
        return uu

    def curve2prof(self, xl, yl, width, widthDir):
        '''
        Routine returning the profile along a curve. !!!! Not tested in a long time !!!!

        Args:
            * xl                : List of the x coordinates of the line.
            * yl                : List of the y coordinates of the line.
            * width             : Width of the zone around the line.
            * widthDir          : Direction to of the width.

        Returns:
            * None
        '''

        # If not list
        if type(xl) is not list:
            xl = xl.tolist()
            yl = yl.tolist()

        # Get the widthDir into radians
        alpha = widthDir*np.pi/180.

        # Get the endpoints
        xe1 = xl[0]
        ye1 = yl[0]
        xe2 = xl[-1]
        ye2 = yl[-1]

        # Convert the endpoints
        elon1, elat1 = self.xy2ll(xe1, ye1)
        elon2, elat2 = self.xy2ll(xe2, ye2)

        # Translate the line into the withDir direction on both sides
        transx = np.sin(alpha)*width/2.
        transy = np.cos(alpha)*width/2.

        # Make a box with that
        box = []
        pts = zip(xl, yl)
        for x, y in pts:
            box.append([x+transx, y+transy])
        box.append([xe2, ye2])
        pts.reverse()
        for x, y in pts:
            box.append([x-transx, y-transy])
        box.append([xe1, ye1])

        # Convert the box into lon lat to save it for further purpose
        boxll = []
        for b in box:
            boxll.append(self.xy2ll(b[0], b[1]))

        # vector perpendicular to the curve
        vec = np.array([xe1-box[-2][0], ye1-box[-2][1]])

        # Get the InSAR points inside this box
        SARXY = np.vstack((self.x, self.y)).T
        rect = path.Path(box, closed=False)
        Bol = rect.contains_points(SARXY)
        xg = self.x[Bol]
        yg = self.y[Bol]
        vel = self.vel[Bol]
        if self.err is not None:
            err = self.err[Bol]
        else:
            err = None

        # Compute the cumulative distance along the line
        dis = np.zeros((len(xl),))
        for i in range(1, len(xl)):
            d = np.sqrt((xl[i] - xl[i-1])**2 + (yl[i] - yl[i-1])**2)
            dis[i] = dis[i-1] + d

        # Sign of the position across
        sarxy = np.vstack((np.array(xg-xe1), np.array(yg-ye1))).T
        sign = np.sign(np.dot(sarxy, vec))

        # Get their position along and across the line
        Dalong = []
        Dacross = []
        for x, y, s in zip(xg.tolist(), yg.tolist(), sign.tolist()):
            d = scidis.cdist([[x, y]], [[xli, yli] for xli, yli in zip(xl, yl)])[0]
            imin1 = d.argmin()
            dmin1 = d[imin1]
            d[imin1] = 99999999.
            imin2 = d.argmin()
            dmin2 = d[imin2]
            # Put it along the fault
            dtot = dmin1+dmin2
            xcd = (xl[imin1]*dmin1 + xl[imin2]*dmin2)/dtot
            ycd = (yl[imin1]*dmin1 + yl[imin2]*dmin2)/dtot
            # Distance
            if dmin1<dmin2:
                jm = imin1
            else:
                jm = imin2
            # Append
            Dalong.append(dis[jm] + np.sqrt( (xcd-xl[jm])**2 + (ycd-yl[jm])**2) )
            Dacross.append(s*np.sqrt( (xcd-x)**2 + (ycd-y)**2 ))

        # Remove NaNs
        jj = np.flatnonzero(np.isfinite(vel)).tolist()
        vel = vel[jj]
        Dalong = np.array(Dalong)[jj]
        Dacross = np.array(Dacross)[jj]
        if err is not None:
            err = err[jj]

        # Length
        length = dis[-1]

        # Center
        uu = np.argmin(np.abs(dis-length/2.))
        xc = xl[uu]
        yc = yl[uu]

        # All done
        return Dalong, vel, err, Dacross, boxll, xc, yc, xe1, ye1, xe2, ye2, length

    def getAlongStrikeOffset(self, name, fault, interpolation=None, width=1.0,
            length=10.0, faultwidth=1.0, tolerance=0.2, azimuthpad=2.0):

        '''
        Runs along a fault to determine variations of the phase offset in the
        along strike direction. !!!! Not tested in a long time !!!!

        Args:
            * name              : name of the results stored in AlongStrikeOffsets
            * fault             : a fault object.

        Kwargs:
            * interpolation     : interpolation distance
            * width             : width of the profiles used
            * length            : length of the profiles used
            * faultwidth        : width of the fault zone.
            * tolerance         : ??
            * azimuthpad        : ??

        Returns:
            * None
        '''

        # the Along strike measurements are in a dictionary
        if not hasattr(self, 'AlongStrikeOffsets'):
            self.AlongStrikeOffsets = {}

        # Interpolate the fault object if asked
        if interpolation is not None:
            fault.discretize(every=interpolation, tol=tolerance)
            xf = fault.xi
            yf = fault.yi
        else:
            xf = fault.xf
            yf = fault.yf

        # Initialize some lists
        ASprof = []
        ASx = []
        ASy = []
        ASazi = []

        # Loop
        for i in range(len(xf)):

            # Write something
            sys.stdout.write('\r Fault point {}/{}'.format(i,len(xf)))
            sys.stdout.flush()

            # Get coordinates
            xp = xf[i]
            yp = yf[i]

            # get the local profile and fault azimuth
            Az, pAz = self._getazimuth(xf, yf, i, pad=azimuthpad)

            # If there is something
            if np.isfinite(Az):

                # Get the profile
                norm, dis, Bol = utils.coord2prof(xp, yp,
                        length, pAz, width)[0:3]
                vel = self.vel[Bol]
                err = self.err[Bol]

                # Keep only the non NaN values
                pts = np.flatnonzero(np.isfinite(vel))
                dis = np.array(dis)[pts]
                ptspos = np.flatnonzero(dis>0.0)
                ptsneg = np.flatnonzero(dis<0.0)

                # If there is enough points, on both sides, get the offset value
                if (len(pts)>20 and len(ptspos)>10 and len(ptsneg)>10):

                    # Select the points
                    vel = vel[pts]
                    err = err[pts]
                    norm = np.array(norm)[pts]

                    # Symmetrize the profile
                    mindis = np.min(dis)
                    maxdis = np.max(dis)
                    if np.abs(mindis)>np.abs(maxdis):
                       pts = np.flatnonzero(dis>-1.0*maxdis)
                    else:
                        pts = np.flatnonzero(dis<=-1.0*mindis)

                    # Get the points
                    dis = dis[pts]
                    ptsneg = np.flatnonzero(dis>0.0)
                    ptspos = np.flatnonzero(dis<0.0)

                    # If we still have enough points on both sides
                    if (len(pts)>20 and len(ptspos)>10 and len(ptsneg)>10 and np.abs(mindis)>(10*faultwidth/2)):

                        # Get the values
                        vel = vel[pts]
                        err = err[pts]
                        norm = norm[pts]

                        # Get offset
                        off = self._getoffset(dis, vel, faultwidth, plot=False)

                        # Store things in the lists
                        ASprof.append(off)
                        ASx.append(xp)
                        ASy.append(yp)
                        ASazi.append(Az)

                    else:

                        # Store some NaNs
                        ASprof.append(np.nan)
                        ASx.append(xp)
                        ASy.append(yp)
                        ASazi.append(Az)

                else:

                    # Store some NaNs
                    ASprof.append(np.nan)
                    ASx.append(xp)
                    ASy.append(yp)
                    ASazi.append(Az)
            else:

                # Store some NaNs
                ASprof.append(np.nan)
                ASx.append(xp)
                ASy.append(yp)
                ASazi.append(Az)

        ASprof = np.array(ASprof)
        ASx = np.array(ASx)
        ASy = np.array(ASy)
        ASazi = np.array(ASazi)

        # Store things
        self.AlongStrikeOffsets[name] = {}
        dic = self.AlongStrikeOffsets[name]
        dic['xpos'] = ASx
        dic['ypos'] = ASy
        lon, lat = self.xy2ll(ASx, ASy)
        dic['lon'] = lon
        dic['lat'] = lat
        dic['offset'] = ASprof
        dic['azimuth'] = ASazi

        # Compute along strike cumulative distance
        if interpolation is not None:
            disc = True
        dic['distance'] = fault.cumdistance(discretized=disc)

        # Clean screen
        sys.stdout.write('\n')
        sys.stdout.flush()

        # all done
        return

    def writeAlongStrikeOffsets2File(self, name, filename):
        '''
        Write the variations of the offset along strike in a file.

        Args:
            * name      : name of the profile to work on
            * filename  : output file name

        Returns:
            * None
        '''

        # Open a file
        fout = open(filename, 'w')

        # Write the header
        fout.write('# Distance (km) || Offset || Azimuth (rad) || Lon || Lat \n')

        # Get the values from the dictionary
        x = self.AlongStrikeOffsets[name]['distance']
        y = self.AlongStrikeOffsets[name]['offset']
        azi = self.AlongStrikeOffsets[name]['azimuth']
        lon = self.AlongStrikeOffsets[name]['lon']
        lat = self.AlongStrikeOffsets[name]['lat']

        # Write to file
        for i in range(len(x)):
            fout.write('{} {} {} {} {} \n'.format(x[i], y[i], azi[i], lon[i], lat[i]))

        # Close file
        fout.close()

    def writeProfile2File(self, name, filename, fault=None):
        '''
        Writes the profile named 'name' to the ascii file filename.

        Args:
            * name      : Name of the profile to work with
            * filename  : Output file name

        Kwargs:
            * fault     : Instance of fault. Can be a list of faults. Adds the intersection with the fault in the header of the file.

        Returns:
            * None
        '''

        # open a file
        fout = open(filename, 'w')

        # Get the dictionary
        dic = self.profiles[name]

        # Write the header
        fout.write('#---------------------------------------------------\n')
        fout.write('# Profile Generated with CSI\n')
        fout.write('# Center: {} {} \n'.format(dic['Center'][0], dic['Center'][1]))
        fout.write('# Endpoints: \n')
        fout.write('#           {} {} \n'.format(dic['EndPointsLL'][0][0], dic['EndPointsLL'][0][1]))
        fout.write('#           {} {} \n'.format(dic['EndPointsLL'][1][0], dic['EndPointsLL'][1][1]))
        fout.write('# Box Points: \n')
        fout.write('#           {} {} \n'.format(dic['Box'][0][0],dic['Box'][0][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][1][0],dic['Box'][1][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][2][0],dic['Box'][2][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][3][0],dic['Box'][3][1]))

        # Place faults in the header
        if fault is not None:
            if fault.__class__ is not list:
                fault = [fault]
            fout.write('# Fault Positions: \n')
            for f in fault:
                d = self.intersectProfileFault(name, f)
                fout.write('# {}           {} \n'.format(f.name, d))

        fout.write('#---------------------------------------------------\n')

        # Write the values
        for i in range(len(dic['Distance'])):
            d = dic['Distance'][i]
            Vp = dic['LOS Velocity'][i]
            if dic['LOS Error'] is not None:
                Ep = dic['LOS Error'][i]
            else:
                Ep = None
            Lon = dic['Lon'][i]
            Lat = dic['Lat'][i]
            if np.isfinite(Vp):
                fout.write('{} {} {} {} {} \n'.format(d, Vp, Ep, Lon, Lat))

        # Close the file
        fout.close()

        # all done
        return

    def plotprofile(self, name, legendscale=10., fault=None, norm=None, ref='utm', synth=False):
        '''
        Plot profile.

        Args:
            * name      : Name of the profile.

        Kwargs:
            * legendscale: Length of the legend arrow.
            * fault     : Fault object
            * norm      : Colorscale limits
            * ref       : utm or lonlat
            * synth     : Plot synthetics (True/False).

        Returns:
            * None
        '''

        # Check the profile
        x = self.profiles[name]['Distance']
        assert len(x)>5, 'There is less than 5 points in your profile...'

        # Plot the insar
        self.plot(faults=fault, norm=norm, show=False)

        # plot the box on the map
        b = self.profiles[name]['Box']
        bb = np.zeros((len(b)+1, 2))
        for i in range(len(b)):
            x = b[i,0]
            if x<0.:
                x += 360.
            bb[i,0] = x
            bb[i,1] = b[i,1]
        bb[-1,0] = bb[0,0]
        bb[-1,1] = bb[0,1]
        self.fig.carte.plot(bb[:,0], bb[:,1], '-k', zorder=0)

        # open a figure
        fig = plt.figure()
        prof = fig.add_subplot(111)

        # plot the profile
        x = self.profiles[name]['Distance']
        y = self.profiles[name]['LOS Velocity']
        ey = self.profiles[name]['LOS Error']
        try:
            p = prof.errorbar(x, y, yerr=ey, label='LOS', fmt='o')
        except:
            p = prof.plot(x, y, label='LOS', marker='.')
        if synth:
            sy = self.profiles[name]['LOS Synthetics']
            s = prof.plot(x, sy, '-r', label='synthetics')

        # If a fault is here, plot it
        if fault is not None:
            # If there is only one fault
            if fault.__class__ is not list:
                fault = [fault]
            # Loop on the faults
            for f in fault:
                # Get the distance
                d = self.intersectProfileFault(name, f)
                if d is not None:
                    ymin, ymax = prof.get_ylim()
                    prof.plot([d, d], [ymin, ymax], '--', label=f.name)

        # plot the legend
        prof.legend()

        # Show to screen
        self.fig.show(showFig=['map'])

        # All done
        return

    def intersectProfileFault(self, name, fault):
        '''
        Gets the distance between the fault/profile intersection and the profile center.

        Args:
            * name      : name of the profile.
            * fault     : fault object from verticalfault.

        Returns:
            * float
        '''

        # Import shapely
        import shapely.geometry as geom

        # Grab the fault trace
        xf = fault.xf
        yf = fault.yf

        # Grab the profile
        prof = self.profiles[name]

        # import shapely
        import shapely.geometry as geom

        # Build a linestring with the profile center
        Lp = geom.LineString(prof['EndPoints'])

        # Build a linestring with the fault
        ff = []
        for i in range(len(xf)):
            ff.append([xf[i], yf[i]])
        Lf = geom.LineString(ff)

        # Get the intersection
        if Lp.crosses(Lf):
            Pi = Lp.intersection(Lf)
            if type(Pi) is geom.point.Point:
                p = Pi.coords[0]
            else:
                return None
        else:
            return None

        # Get the center
        lonc, latc = prof['Center']
        xc, yc = self.ll2xy(lonc, latc)

        # Get the sign
        xa,ya = prof['EndPoints'][0]
        vec1 = [xa-xc, ya-yc]
        vec2 = [p[0]-xc, p[1]-yc]
        sign = np.sign(np.dot(vec1, vec2))

        # Compute the distance to the center
        d = np.sqrt( (xc-p[0])**2 + (yc-p[1])**2)*sign

        # All done
        return d

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
        if self.synth is not None:
            synthRMS = np.sqrt( 1./N *sum( (self.vel - self.synth)**2 ) )
            return dataRMS, synthRMS
        else:
            return dataRMS, 0.

        # All done

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
        if self.synth is not None:
            rmean = (self.vel - self.synth).mean()
            synthVariance = ( 1./N *sum( (self.vel - self.synth - rmean)**2 ) )
            return dataVariance, synthVariance
        else:
            return dataVariance, 0.

        # All done

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

    def plot(self, faults=None, figure=None, gps=None, decim=False, norm=None, data='data', show=True, drawCoastlines=True, expand=0.2, edgewidth=1, figsize=[None, None]):
        '''
        Plot the data set, together with a fault, if asked.

        Kwargs:
            * faults            : list of fault objects.
            * figure            : number of the figure.
            * gps               : list of gps objects.
            * decim             : plot the insar following the decimation process of varres.
            * norm              : colorbar limits
            * data              : 'data', 'synth' or 'res'
            * show              : bool. Show on screen?
            * drawCoastlines    : bool. default is True
            * expand            : default expand around the limits covered by the data
            * edgewidth         : width of the edges of the decimation process patches
            * figsize           : tuple of figure sizes

        Returns:
            * None
        '''

        # Get lons lats
        lonmin = self.lon.min()-expand
        if lonmin<0.:
            lonmin += 360.
        lonmax = self.lon.max()+expand
        if lonmax<0.:
            lonmax += 360.
        latmin = self.lat.min()-expand
        latmax = self.lat.max()+expand

        # Create a figure
        fig = geoplot(figure=figure, lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax, figsize=figsize)

        # Draw the coastlines
        if drawCoastlines:
            fig.drawCoastlines(drawLand=True, parallels=5, meridians=5, drawOnFault=True)

        # Plot the gps data if asked
        if gps is not None:
            if type(gps) is not list:
                gps = [gps]
            for g in gps:
                fig.gps(g)

        # Plot the decimation process, if asked
        if decim:
            fig.insar(self, norm=norm, colorbar=True, data=data, plotType='decimate', edgewidth=edgewidth)

        # Plot the insar
        if not decim:
            fig.insar(self, norm=norm, colorbar=True, data=data, plotType='scatter')

        # Plot the fault trace if asked
        if faults is not None:
            if type(faults) is not list:
                faults = [faults]
            for fault in faults:
                if fault.type is "Fault":
                    fig.faulttrace(fault)

        # Show
        if show:
            fig.show(showFig=['map'])
        else:
            self.fig = fig

        # All done
        return

    def write2grd(self, fname, oversample=1, data='data', interp=100, cmd='surface',
                        tension=None, useGMT=False, verbose=False, outDir='./'):
        '''
        Write to a grd file.

        Args:
            * fname     : Filename

        Kwargs:
            * oversample    : Oversampling factor.
            * data          : can be 'data' or 'synth'.
            * interp        : Number of points along lon and lat.
            * cmd           : command used for the conversion( i.e., surface or xyz2grd)
            * tension       : Tension in the gmt command
            * useGMT        : use surface or xyz2grd or the direct wrapper to netcdf
            * verbose       : Talk to us
            * outDir        : output directory

        Returns:
            * None
        '''

        # Filename
        fname = os.path.join(outDir, fname)

        # Get variables
        x = self.lon
        y = self.lat
        if data == 'data':
            z = self.vel
        elif data == 'synth':
            z = self.synth
        elif data == 'poly':
            z = self.orbit
        elif data == 'res':
            z = self.vel - self.synth

        if not useGMT:

            utils.write2netCDF(fname, x, y, z, nSamples=interp, verbose=verbose)

        else:

            # Write these to a dummy file
            fout = open('xyz.xyz', 'w')
            for i in range(x.shape[0]):
                fout.write('{} {} {} \n'.format(x[i], y[i], z[i]))
            fout.close()

            # Import subprocess
            import subprocess as subp

            # Get Rmin/Rmax/Rmin/Rmax
            lonmin = x.min()
            lonmax = x.max()
            latmin = y.min()
            latmax = y.max()
            R = '-R{}/{}/{}/{}'.format(lonmin, lonmax, latmin, latmax)

            # Create the -I string
            if type(interp)!=list:
                Nlon = int(interp)*int(oversample)
                Nlat = Nlon
            else:
                Nlon = int(interp[0])
                Nlat = int(interp[1])
            I = '-I{}+/{}+'.format(Nlon,Nlat)

            # Create the G string
            G = '-G'+fname

            # Create the command
            com = [cmd, R, I, G]

            # Add tension
            if tension is not None and cmd in ('surface'):
                T = '-T{}'.format(tension)

            # open stdin and stdout
            fin = open('xyz.xyz', 'r')

            # Execute command
            subp.call(com, stdin=fin, shell=False)

            # CLose the files
            fin.close()

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
        if data is 'data':
            z = self.vel
        elif data is 'synth':
            z = self.synth
        elif data is 'poly':
            z = self.orbit
        elif data is 'resid':
            z = self.vel - self.synth

        # Write these to a file
        fout = open(os.path.join(outDir, fname), 'w')
        for i in range(x.shape[0]):
            fout.write('{} {} {} \n'.format(x[i], y[i], z[i]))
        fout.close()

        return

    def writeDecim2file(self, filename, data='data', outDir='./'):
        '''
        Writes the decimation scheme to a file plottable by GMT psxy command.

        Args:
            * filename  : Name of the output file (ascii file)

        Kwargs:
            * data      : Add the value with a -Z option for each rectangle. Can be 'data', 'synth', 'res', 'transformation'
            * outDir    : output directory

        Returns:
            * None
        '''

        # Open the file
        fout = open(os.path.join(outDir, filename), 'w')

        # Which data do we add as colors
        if data in ('data', 'd', 'dat', 'Data'):
            values = self.vel
        elif data in ('synth', 's', 'synt', 'Synth'):
            values = self.synth
        elif data in ('res', 'resid', 'residuals', 'r'):
            values = self.vel - self.synth
        elif data in ('transformation', 'trans', 't'):
            values = self.transformation

        # Iterate over the data and corner
        for corner, d in zip(self.corner, values):

            # Make a line
            string = '> -Z{} \n'.format(d)
            fout.write(string)

            # Write the corners
            xmin, ymin, xmax, ymax = corner
            fout.write('{} {} \n'.format(xmin, ymin))
            fout.write('{} {} \n'.format(xmin, ymax))
            fout.write('{} {} \n'.format(xmax, ymax))
            fout.write('{} {} \n'.format(xmax, ymin))
            fout.write('{} {} \n'.format(xmin, ymin))

        # Close the file
        fout.close()

        # All done
        return

    def _getazimuth(self, x, y, i, pad=2):
        '''
        Get the azimuth of a line.
        Args:
            * x,y       : x,y values of the line.
            * i         : index of the position of interest.
            * pad       : number of points to take into account.
        '''
        # Compute distances along trace
        dis = np.sqrt((x-x[i])**2 + (y-y[i])**2)
        # Get points that are close than pad/2.
        pts = np.where(dis<=pad/2)
        # Get the azimuth if there is more than 2 points
        if len(pts[0])>=2:
                d = y[pts]
                G = np.vstack((np.ones(d.shape),x[pts])).T
                m,res,rank,s = np.linalg.lstsq(G,d)
                Az = np.arctan(m[1])
                pAz= Az+np.pi/2
        else:
                Az = np.nan
                pAz = np.nan
        # All done
        return Az*180./np.pi,pAz*180./np.pi

    def _getoffset(self, x, y, w, plot=True):
        '''
        Computes the offset around zero along a profile.
        Args:
            * x         : X-axis of the profile
            * y         : Y-axis of the profile
            * w         : Width of the zero zone.
        '''

        # Initialize plot
        if plot:
            plt.figure(1213)
            plt.clf()
            plt.plot(x,y,'.k')

        # Define function
        G = np.vstack((np.ones(y.shape),x)).T

        # fit a function on the negative side
        pts = np.where(x<=-1.*w/2.)
        dneg = y[pts]
        Gneg = np.squeeze(G[pts,:])
        mneg,res,rank,s = np.linalg.lstsq(Gneg,dneg)
        if plot:
            plt.plot(x,np.dot(G,mneg),'-r')

        # fit a function on the positive side
        pts = np.where(x>=w/2)
        dpos = y[pts]
        Gpos = np.squeeze(G[pts,:])
        mpos,res,rank,s = np.linalg.lstsq(Gpos,dpos)
        if plot:
            plt.plot(x,np.dot(G,mpos),'-g')

        # Offset
        off = mpos[0] - mneg[0]

        # plot
        if plot:
            print('Measured offset: {}'.format(off))
            plt.show()

        # all done
        return off

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
