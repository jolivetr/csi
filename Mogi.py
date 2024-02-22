'''
A Mogi sub-class

Written by T. Shreve, June 2019
'''

# Import Externals stuff
import numpy as np
from argparse import Namespace
import warnings

# Personals
from . import mogifull
from .Pressure import Pressure

#sub-class Mogi
class Mogi(Pressure):

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Sub-class implementing Mogi pressure objects.

        Args:
            * name          : Name of the pressure source.
            * utmzone       : UTM zone  (optional, default=None)
            * ellps         : ellipsoid (optional, default='WGS84')

        Kwargs:
            * x0, y0       : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * ax           : semi-axes of the Mogi source, should be equal.
            * ellps        : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(Mogi,self).__init__(name, utmzone=utmzone, ellps=ellps, 
                                        lon0=lon0, lat0=lat0, verbose=verbose)
        self.source = "Mogi"
        self.deltapressure  = None  # Dimensionless pressure
        self.deltavolume = None
        return


    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Set volume and pressure changes
    def setVolume(self, deltaVolume):
        '''
        Set deltapressure given deltavolume.

        Returns:
            * deltavolume             : Volume change
        '''

        self.deltavolume = deltaVolume
        self.volume2pressure()

        # All done
        return

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Set volume and pressure changes
    def setPressure(self, deltaPressure):
        '''
        Set deltavolume given deltapressure.

        Returns:
            * deltapressure             : Pressure change
        '''
        #Check if deltavolume already defined
        self.deltapressure = deltaPressure
        self.pressure2volume()


        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Convert pressure change to volume change for Mogi
    def pressure2volume(self):
        '''
        Converts pressure change to volume change (m3) for Mogi.

        Uses formulation (eq. 15) from:
        Amoruso and Crescentini, 2009, Shape and volume change of pressurized ellipsoidal cavities from deformation and seismic data

        Assuming radius (a) << depth (z0), formulation is:
        deltaV = (3/4)*(V*deltaP/mu)

        i.e. point source approximation for eq. 19 in:
        Battaglia, Maurizio, Cervelli, P.F., and Murray, J.R., 2013, Modeling crustal deformation near active faults and volcanic centers

        If no assumptions on radius vs. depth:
        deltaV = (pi*a^3)*(deltaP/mu)*[1+(a/z0)^4]


        Returns:
            * deltavolume             : Volume change
        '''
        #Check if deltapressure already defined
        if self.deltapressure is None:
                raise ValueError("Need to set self.deltapressure with self.setPressure.")
        self.volume = self.computeVolume()
        self.deltavolume = (3./4.)*self.volume*self.deltapressure/self.mu


        # All done
        return self.deltavolume

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Convert volume change to pressure change for Mogi
    def volume2pressure(self):
        '''
        Converts volume change (m3) to pressure change for Mogi.

        Uses formulation (eq. 15) from:
        Amoruso and Crescentini, 2009, Shape and volume change of pressurized ellipsoidal cavities from deformation and seismic data

        Assuming radius (a) << depth (z0), formulation is:
        deltaP = (4/3)*(mu/V)*deltaV

        Returns:
            * deltapressure             : Pressure change
        '''
        #Check if deltavolume already defined
        if self.deltavolume is None:
                raise ValueError("Need to set self.deltavolume with self.setVolume.")

        self.volume = self.computeVolume()
        self.deltapressure = (4./3.)*self.mu*self.deltavolume/self.volume

        # All done
        return self.deltapressure
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Find volume of ellipsoidal cavity
    def computeVolume(self):
        '''
        Computes volume (m3) of spheroidal cavity, given the semimajor axis.

        Returns:
            * volume             : Volume of cavity
        '''
        a = self.ellipshape['ax']
        self.volume = (4./3.)*np.pi*a**3             #Volume of sphere = 4/3*pi*a^3

        # All done
        return self.volume


    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Some building routines that can be touched... I guess
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def pressure2dis(self, data, delta="pressure", volume=None):
        '''
        Computes the surface displacement at the data location using Mogi formulations.

        Args:
            data (Data): Data object from GPS or InSAR.
            delta (str): Unit pressure is assumed. Default is "pressure".
            volume (float): Volume change. If None, the volume change is calculated based on the pressure change.

        Returns:
            numpy.ndarray: Array of x, y, and z displacements.

        Raises:
            Exception: If the shape of the spheroid is not defined.

        Notes:
            This method uses Mogi's equations to compute the surface displacement at the data location. 
            The pressure change or volume change can be specified to calculate the displacement.
        '''

        # Set the volume change
        if volume is None:
            self.volume2pressure()
            DP = self.deltapressure
        else:
            # Set the pressure value
            if delta=="pressure":
                DP = self.mu                             #Dimensionless unit pressure
            elif delta=="volume":
                print("Converting to pressure for Mogi Green's function calculations")
                DP = self.mu

        # Set the shear modulus and poisson's ratio
        if self.mu is None:
            self.mu        = 30e9
        if self.nu is None:
            self.nu        = 0.25

        # Get parameters
        if self.ellipshape is None:
            raise Exception("Need to define shape of spheroid (run self.createShape)")
        #define dictionary entries as variables
        ellipse = Namespace(**self.ellipshape)

        # Get data position -- in m
        x = data.x*1000
        y = data.y*1000
        z = np.zeros(x.shape)   # Data are at the surface

        if (DP!=0.0):
            # x0, y0 need to be in utm and meters
            Ux,Uy,Uz = mogifull.displacement(x, y, z, ellipse.x0m*1000, ellipse.y0m*1000, ellipse.z0, ellipse.ax, DP/self.mu, self.nu)
        else:
            dp_dis = np.zeros((len(x), 3))
        # All done
        # concatenate into one array
        u = np.vstack([Ux, Uy, Uz]).T

        return u

    # ----------------------------------------------------------------------
    # Define the shape of the spheroid
    #
    # ----------------------------------------------------------------------

    def createShape(self, x, y, z0, a, latlon=True):
        '''
        Defines the shape of the mogi pressure source.

        Args:
            * x, y         : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * a            : Radius (m)

        Returns:
            None
        '''
        if latlon is True:
            self.lon, self.lat = x, y
            lon1, lat1 = x, y
            self.pressure2xy()
            x1, y1 = self.xf, self.yf
        else:
            self.xf, self.yf = x, y
            x1, y1 = x, y
            self.pressure2ll()
            lon1, lat1 = self.lon, self.lat
        self.ellipshape = {'x0': lon1,'x0m': x1,'y0': lat1, 'y0m': y1, 'z0': z0,'ax': a, 'ay': a, 'az': a, 'strike': 0.0, 'dip': 0.0, 'plunge': 0.0}
        #if radius is not much smaller than depth, mogi equations don't hold
        if float(a)/float(z0)> 0.1 :
            warnings.warn('Results may be inaccurate if depth is not much greater than radius',Warning)
        return
