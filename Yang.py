'''
A Yang sub-class

Written by T. Shreve, June 2019
'''

# Import Externals stuff
import numpy as np
import sys
import os
from argparse import Namespace

# Personals
from . import yangfull
from .Pressure import Pressure
from .EDKSmp import sum_layered
from .EDKSmp import dropSourcesInPatches as Patches2Sources

#sub-class Yang
class Yang(Pressure):

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, x0=None, y0=None, z0=None, ax=None, ay=None, az=None, dip=None, strike=None, plunge=None, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Sub-class implementing Yang pressure objects.

        Args:
            * name          : Name of the pressure source.
            * utmzone       : UTM zone  (optional, default=None)
            * ellps         : ellipsoid (optional, default='WGS84')

        Kwargs:
            * x0, y0       : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * ax, ay, az   : Semi-axes of the CDM along the x, y and z axes respectively, before applying rotations. ax=ay for Yang.
            * dip          : Clockwise around N-S (Y) axis; dip = 90 means vertically elongated source
            * strike       : Clockwise from N; strike = 0 means source is oriented N-S
            * plunge       : 0 for Yang
            * ellps        : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(Yang,self).__init__(name, x0=x0, y0=y0, z0=z0, ax=ax, ay=ay, az=az, dip=dip, strike=strike, plunge=plunge, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=True)
        self.source = "Yang"
        self.deltapressure  = None  # Dimensionless pressure
        self.deltavolume = None
        if None not in {x0, y0, z0, ax, ay, az, dip, strike}:
            self.createShape(x0, y0, z0, ax, ay, az, dip, strike, latlon=True)
            print("Defining source parameters")

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
        self.deltapressure = deltaPressure
        self.pressure2volume()


        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Convert pressure change to volume change for Yang
    def pressure2volume(self):
        '''
        Converts pressure change to volume change (m3) for Yang.

        Uses empirical formulation from:
        Battaglia, Maurizio, Cervelli, P.F., and Murray, J.R., 2013, Modeling crustal deformation near active faults and volcanic centers

        Rigorous formulation:
        deltaV = ((1-2v)/(2*(1+v)))*V*(deltaP/mu)*((p^T/deltaP)-3),
        where V is the volume of the ellipsoidal cavity and p^T is the trace of the stress inside the ellipsoidal cavity.

        Empirical formulation:
        deltaV = V*(deltaP/mu)((A^2/3)-0.7A+1.37)

        Returns:
            * deltavolume             : Volume change
        '''
        #Check if deltapressure already defined
        if self.deltapressure is None:
                raise ValueError("Need to set self.deltapressure with self.setPressure.")
        self.volume = self.computeVolume()
        A = self.ellipshape['ax']/self.ellipshape['az']
        self.deltavolume = (self.volume)*(self.deltapressure/self.mu)*(((A**2)/3.)-(0.7*A)+1.37)



        # All done
        return self.deltavolume

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Convert volume change to pressure change for Yang
    def volume2pressure(self):
        '''
        Converts volume change (m3) to pressure change for Yang.

        Uses empirical formulation from:
        Battaglia, Maurizio, Cervelli, P.F., and Murray, J.R., 2013, Modeling crustal deformation near active faults and volcanic centers

        Empirical formulation:
        deltaP = (deltaV/V)*(mu/((A^2/3)-0.7A+1.37))

        Returns:
            * deltapressure             : Pressure change
        '''
        #Check if deltavolume already defined
        if self.deltavolume is None:
                raise ValueError("Need to set self.deltavolume with self.setVolume.")
        self.volume = self.computeVolume()
        A = self.ellipshape['ax']/self.ellipshape['az']
        self.deltapressure = (self.deltavolume/self.volume)*(self.mu/(((A**2)/3.)-(0.7*A)+1.37))

        # All done
        return self.deltapressure
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Find volume of ellipsoidal cavity
    def computeVolume(self):
        '''
        Computes volume (m3) of ellipsoidal cavity, given the semimajor axis.

        Returns:
            * volume             : Volume of cavity
        '''
        a = self.ellipshape['az']
        A = self.ellipshape['ax']/self.ellipshape['az']
        self.volume = (4./3.)*np.pi*a*((a*A)**2)             #Volume of ellipsoid = 4/3*pi*a*b*c

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
        Computes the surface displacement at the data location using yang. ~~~ This is where the good stuff happens ~~

        Args:
            * data          : Data object from gps or insar.
            * delta         : Unit pressure is assumed.

        Returns:
            * u             : x, y, and z displacements
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
                print("Converting to pressure for Yang Green's function calculations")
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
        strike = ellipse.strike + 90

        # Run it for yang pressure source
        if (DP!=0.0):
            # x0, y0 need to be in utm and meters
            Ux,Uy,Uz = yangfull.displacement(x, y, z, ellipse.x0m*1000, ellipse.y0m*1000, ellipse.z0, ellipse.az, ellipse.ax/ellipse.az, ellipse.dip, strike, DP/self.mu, self.nu)
        else:
            dp_dis = np.zeros((len(x), 3))

        # All done
        # concatenate into one array
        u = np.vstack([Ux, Uy, Uz]).T
    
        return u

    # ----------------------------------------------------------------------
    # Define the shape of the ellipse
    #
    # ----------------------------------------------------------------------

    def createShape(self, x, y, z0, ax, ay, az, dip, strike, latlon=True):
        '''
        Defines the shape of the mogi pressure source.

        Args:
            * x, y         : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * (ax, ay, az) : Principle semi-axes (m) before rotations applied. az will be the semi-major axis, while ax = ay.
            * dip          : Plunge angle (dip=90 is vertical source)
            * strike       : Azimuth (azimuth=0 is aligned North)

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
        ### before rotation, so az = semi-major axis and ax = ay
        c, b, a = np.sort([ax,ay,az])
        A = b/a
        if A == 1:
            print('If semi-minor and semi-major axes are equal, more efficient to use Mogi.py')
        ### Double check this
        if float(z0) < (float(A)*float(a))**2/float(a):
            print('WARNING: radius of curvature has to be less than the depth, will output "null" shape')
            #... this model is still taken into account in Bayesian inverison, but should be rejected.')
            self.ellipshape = {'x0': 0.,'x0m': 0.,'y0': 0.,'y0m': 0.,'z0': 0.,'a': 0.,'A': 0.,'dip': 0.,'strike': 0.}
        else:
            print('Using CDM conventions for rotation - dip = 90 is vertical, rotation clockwise around Y-axis (N-S). dip = 0, strike = 0 source elongated N-S')
            self.ellipshape = {'x0': lon1,'x0m': x1,'y0': lat1,'y0m': y1,'z0': z0,'ax': b,'ay': c,'az': a,'dip': dip,'strike': strike,'plunge': 0.0}

        return
