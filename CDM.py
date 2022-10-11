'''
A Compound Dislocation Model (CDM) sub-class.

Written by T. Shreve, June 2019
'''

# Import Externals stuff
import numpy as np
import sys
import os
from argparse import Namespace

# Personals
from . import CDMfull
from .Pressure import Pressure
from .EDKSmp import sum_layered
from .EDKSmp import dropSourcesInPatches as Patches2Sources

#sub-class CDM
class CDM(Pressure):

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, x0=None, y0=None, z0=None, ax=None, ay=None, az=None, dip=None, strike=None, plunge=None, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Sub-class implementing CDM pressure objects.

        Args:
            * name          : Name of the pressure source.
            * utmzone      : UTM zone  (optional, default=None)


        Kwargs:
            * x0, y0       : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * ax, ay, az   : Semi-axes of the CDM along the x, y and z axes respectively, before applying rotations.
            * dip          : Clockwise around N-S (Y) axis; dip = 90 means vertically elongated source
            * strike       : Clockwise from N; strike = 0 means source is oriented N-S
            * plunge       : Clockwise along E-W (X) axis
            * ellps        : ellipsoid (optional, default='WGS84')
        '''

        # Attributes only for CDM and pCDM

        # Potency given semi-axes and unit opening
        self.DV = None

        # Final potency change scaled by self.DV
        self.deltapotency   = None

        # Tensile component on each rectangular dislocation
        self.deltaopening   = None

        # Base class init
        super(CDM,self).__init__(name, x0=x0, y0=y0, z0=z0, ax=ax, ay=ay, az=az, dip=dip, strike=strike, plunge=plunge, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=True)
        self.source = "CDM"
        if None not in {x0, y0, z0, ax, ay, az, dip, strike, plunge}:
            self.createShape(x0, y0, z0, ax, ay, az, dip, strike, plunge)
            print("Defining source parameters")

        return


    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Find unit potency change (opening=1)
    def computePotency(self):
        '''
        Computes potency change (m3) of three orthogonal point tensile dislocations, given the semimajor axes.

        Returns:
            * A                   : Ratio of horizontal potency (volume) to total potency (volume) variation
            * B                   : Ratio of vertical potency (volume) variation
            * self.DV             : Change in potency
        '''
        DVx = 4*self.ellipshape['ay']*self.ellipshape['az']                          #unit potency (assume opening = 1 m )
        DVy = 4*self.ellipshape['ax']*self.ellipshape['az']
        DVz = 4*self.ellipshape['ax']*self.ellipshape['ay']
        A = DVz / (DVx + DVy + DVz)
        B = DVy / (DVx + DVy)
        self.DV = (DVx + DVy + DVz)

        return A, B, self.DV


    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Some building routines that can be touched... I guess
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------
    # Convert openign to change in potency
    def opening2potency(self):
        '''
        Converts opening (m) to potency change (m3) for CDM.

        Uses formula from :
        Nikkhoo, M., Walter, T. R., Lundgren, P. R., and Prats-Iraola, P. (2017). Compound dislocation models(CDMs) for volcano deformation analyses.Geophysical Journal International, 208(2):877–894


        Returns:
            * deltapotency             : strength of dislocation source, or volume available to host fluids intruding into cavity from the outside
        '''

        ax, ay, az = self.ellipshape['ax'], self.ellipshape['ay'], self.ellipshape['az']
        self.deltapotency = (4*(ax*ay + ay*az + ax*az)*self.deltaopening)

        # All done
        return self.deltapotency


    def pressure2dis(self, data, delta="volume", volume=None):
        '''
        Computes the surface displacement at the data location using pCDM. ~~~ This is where the good stuff happens ~~

        Args:
            * data          : data object from gps or insar.
            * delta         : total potency change, in units of volume
        Returns:
            * u             : x, y, and z displacements
        '''

        # Get parameters
        if self.ellipshape is None:
            raise Exception("Error: Need to define shape of source (run self.createShape)")
        #define dictionary entries as variables
        ellipse = Namespace(**self.ellipshape)

        # Set the potency change
        if volume is None:
            self.DV = (4*(ellipse.ax*ellipse.ay + ellipse.ax*ellipse.az + ellipse.ay*ellipse.az))
            opening = self.deltapotency/self.DV
            print("Opening is", opening)
        else:
            # Set the potency value
            if delta=="volume":
                opening = 1.0
                self.DV = (4*(ellipse.ax*ellipse.ay + ellipse.ax*ellipse.az + ellipse.ay*ellipse.az))
            if delta=="pressure":
                raise Exception("Error: CDM requires potencies, not pressure")

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

        if (opening!=0.0):
            #### x0, y0 need to be in utm and meters
            Ux,Uy,Uz,Dv = CDMfull.displacement(x, y, z, ellipse.x0m*1000, ellipse.y0m*1000, ellipse.z0, omegaX, omegaY, omegaZ, ellipse.ax, ellipse.ay, ellipse.az, opening, self.nu)
        else:
            dp_dis = np.zeros((len(x), 3))

        # All done
        # concatenate into one array
        u = np.vstack([Ux, Uy, Uz]).T


        return u

    # ----------------------------------------------------------------------
    # Define the shape of the CDM
    #
    # ----------------------------------------------------------------------

    def createShape(self, x, y, z0, ax, ay, az, dip, strike, plunge, latlon=True):
        """"
        Defines the shape of the CDM pressure source.

        Args:
            * x0, y0       : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * ax, ay, az   : Semi-axes of the CDM along the x, y and z axes respectively, before applying rotations.
            * dip          : Clockwise around N-S (Y) axis; dip = 90 means vertical source
            * strike       : Clockwise from N; strike = 0 means source is oriented N-S
            * plunge       : Clockwise along E-W (X) axis

        Returns:
            None

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

        self.ellipshape = {'x0': lon1,'x0m': x1,'y0': lat1, 'y0m': y1, 'z0': z0,'ax': ax,'ay': ay, 'az': az, 'dip': dip,'strike': strike, 'plunge': plunge}

        return
