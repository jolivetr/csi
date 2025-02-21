'''
A pointCDM (pCDM) sub-class. Once we have the "finite" pCDM, this should end up being the parent pressure class

Written by T. Shreve, June 2019
'''

# Import Externals stuff
import numpy as np
from argparse import Namespace

# Personals
from . import pCDMfull
from .Pressure import Pressure

#sub-class pCDM
class pCDM(Pressure):

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, x0=None, y0=None, z0=None, ax=None, ay=None, az=None, dip=None, strike=None, plunge=None, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Sub-class implementing pCDM pressure objects.

        Args:
            * name          : Name of the pressure source.
            * utmzone       : UTM zone  (optional, default=None)
            * ellps         : ellipsoid (optional, default='WGS84')

        Kwargs:
            * x0, y0       : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * ax, ay, az   : None for pCDM
            * dip          : Clockwise around N-S (Y) axis; dip = 90 means vertically elongated source
            * strike       : Clockwise from N; strike = 0 means source is oriented N-S
            * plunge       : Clockwise along E-W (X) axis
            * ellps        : ellipsoid (optional, default='WGS84')
        '''

        # Attribute only for pCDM and CDM
        self.deltapotency   = None
        # Unit potency based on semi-axes
        self.DVtot = None
        self.DVx = None
        self.DVy = None
        self.DVz = None
        self.A = None
        self.B = None
        self.scale = 1000000

        # Base class init
        super(pCDM,self).__init__(name, x0=x0, y0=y0, z0=z0, ax=ax, ay=ay, az=az, dip=dip, strike=strike, plunge=plunge, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=True)
        self.source = "pCDM"
        if None not in {x0, y0, z0, dip, strike, plunge}:
            self.createShape(x0, y0, z0, dip, strike, plunge)
            print("Defining source parameters")


        return

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Find unit potency change (opening=1)
    def computeTotalpotency(self):
        '''
        Computes the total potency and ratios A and B

        Args:
            * A                   : Ratio of horizontal potency (volume) to total potency (volume) variation
            * B                   : Ratio of vertical potency (volume) variation
            * self.DV             : Change in potency
        '''
        if None in {self.DVz, self.DVx, self.DVy}:
            self.DVz = self.ellipshape['az']
            self.DVy = self.ellipshape['ay']
            self.DVx = self.ellipshape['ax']

        self.A = self.DVz/(self.DVz+self.DVy+self.DVx)
        self.B = self.DVy/(self.DVy+self.DVx)
        self.DVtot = (self.DVz+self.DVy+self.DVx)

        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Some building routines that can be touched... I guess
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def pressure2dis(self, data, delta="volume", volume=None):
        '''
        Computes the surface displacement at the data location using pCDM. ~~~ This is where the good stuff happens ~~

        Args:
            * data          : data object from gps or insar.
            * delta         : only total potency, in units of volume, in the pCDM case
        Returns:
            * u             : x, y, and z displacements
        '''
        # Get parameters
        if self.ellipshape is None:
            raise Exception("Error: Need to define shape of source (run self.createShape)")
        #define dictionary entries as variables
        ellipse = Namespace(**self.ellipshape)

        # Set the volume change
        if volume is None:
            DVx = ellipse.ax
            DVy = ellipse.ay
            DVz = ellipse.az
            self.computeTotalpotency()
        else:
            # Set the potency value
            if delta=="volume":
                DVx, DVy, DVz = self.scale, self.scale, self.scale
            if delta=="pressure":
                raise Exception("pCDM requires potencies, not pressure")
        # Set poisson's ratio
        if self.nu is None:
            self.nu        = 0.25

        # Get data position -- in m
        x = data.x*1000
        y = data.y*1000
        z = np.zeros(x.shape)   # Data are at the surface
        # Run it for pCDM pressure source

        # Convert from dip, strike, and plunge to omegaX, omegaY, omegaZ - verify they have same convention as Yang
        omegaX = ellipse.plunge
        omegaY = 90. - ellipse.dip
        omegaZ = ellipse.strike

        if any(v != 0.0 for v in [DVx, DVy, DVz]):
            #### x0, y0 need to be in utm and meters
            Ux1, Ux2, Ux3, Uy1, Uy2, Uy3, Uz1, Uz2, Uz3, Ux, Uy, Uz = pCDMfull.displacement(x, y, z, ellipse.x0m*1000, ellipse.y0m*1000, ellipse.z0, omegaX, omegaY, omegaZ, DVx, DVy, DVz, self.nu)
        else:
            u1 = np.zeros((len(x), 3))
            u2 = np.zeros((len(x), 3))
            u3 = np.zeros((len(x), 3))

        # All done
        # concatenate into one array
        u1 = np.vstack([Ux1, Uy1, Uz1]).T
        u2 = np.vstack([Ux2, Uy2, Uz2]).T
        u3 = np.vstack([Ux3, Uy3, Uz3]).T

        return u1, u2, u3

    # ----------------------------------------------------------------------
    # Define the shape of the CDM
    #
    # ----------------------------------------------------------------------

    def createShape(self, x, y, z0, dip, strike, plunge, latlon=True):
        """"
        Defines the shape of the pCDM pressure source.

        Args:
            * x0, y0       : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * A            : Horizontal divided by total volume variation ratio, A = DVz/(DVx+DVy+DVz) --- ??? or potency ???
            * B            : Vertical volume variation ratio, B : DVy/(DVx+DVy) --- ??? or potency ???
            * dip          : Clockwise around N-S (Y) axis ??? To verify
            * strike       : Clockwise from N ??? To verify
            * plunge       : Clockwise along E-W (X) axis??? To verify


            Examples:
            A = 1, any B : horizontal sill
            A = 0, B = 0.5 : vertical pipe
            A = 0, B = 0 or 1 : vertical dyke
            A = 1/3, B = 0.5 : isotrope source
            A > 0, B > 0 : dyke + sill


        """
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

        ##c, b, a = np.sort([float(ax),float(ay),float(az)])
        ##if b == c:
        ##    if a == b == c:
        ##        raise Exception('All axes are equal, use Mogi.py')
        ##    else:
        ##        raise Exception('Semi-minor axes are equal, use Yang.py')
        #A, B, self.DV = self.computePotency()

        # DVx = 4*ay*az                          #unit potency (assume opening = 1 m )
        # DVy = 4*ax*az
        # DVz = 4*ax*ay
        # A = DVz / (DVx + DVy + DVz)
        # B = DVy / (DVx + DVy)
        # self.DV = (DVx + DVy + DVz)

        self.ellipshape = {'x0': lon1,'x0m': x1,'y0': lat1, 'y0m': y1,'z0': z0, 'ax': 1.0, 'ay': 1.0, 'az': 1.0,'dip': dip, 'strike': strike, 'plunge': plunge}

        return
