'''
A class that deals with 3D faults.

Written by R. Jolivet, B. Riel and Z. Duputel April 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys
import os

# Rectangular patches Fault class
from .RectangularPatches import RectangularPatches

# Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok

class fault3D(RectangularPatches):

    '''
    A class that handles faults in 3D. It inherits from RectangularPatches but
    allows to build the fault in 3D using tying points.

    Args:
        * name      : Name of the fault.

    Kwargs:
        * utmzone   : UTM zone  (optional, default=None)
        * lon0      : Longitude of the center of the UTM zone
        * lat0      : Latitude of the center of the UTM zone
        * ellps     : ellipsoid (optional, default='WGS84')
        * verbose   : Speak to me (default=True)

    '''

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Base class init
        super(fault3D,self).__init__(name,
                                     utmzone = utmzone,
                                     ellps = ellps, 
                                     lon0 = lon0, 
                                     lat0 = lat0, 
                                     verbose = verbose)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Computes dip angle at depth z
    def dipatZ(self, interp, z):
        '''
        Uses the interpolator to return the dip angle evolution along strike at depth z.
        The interpolation scheme is piecewise linear.

        Args:

            * interp        : Dip interpolation function
            * z             : Depth.
        '''

        # Create a structure
        self.dip = []

        # Set a distance counter
        dis = 0

        # Set the previous x,y
        xp = self.xi[0]
        yp = self.yi[0]

        # Loop along the discretized fault trace
        for i in range(self.xi.shape[0]):

            # Update the distance
            dis += np.sqrt( (self.xi[i]-xp)**2 + (self.yi[i]-yp)**2 )

            # get the dip
            d = interp(dis, z[i])

            # store it
            self.dip.append(d)

            # Update previous xp, yp
            xp = self.xi[i]; yp = self.yi[i]

        # Array it
        self.dip = np.array(self.dip)

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Build rectangular patches
    def buildPatches(self, dip=None, dipdirection=None, every=10, 
                           minpatchsize=0.00001, trace_tol=0.1, trace_fracstep=0.2, 
                           trace_xaxis='x', trace_cum_error=True):
        '''
        Builds a dipping fault given a certain dip angle evolution. Dip angle is linearly
        interpolated between tying points given as arguments.

        Args:
            * dip               : Dip angle tying points 

                [[alongstrike, depth, dip], [alongstrike, depth, dip], ..., [alongstrike, depth, dip]]
                                   
            * dipdirection      : Direction towards which the fault dips.
            * every             : patch length for the along trace discretization
            * minpatchsize      : minimum patch size
            * trace_tol         : tolerance for the along trace patch discretization optimization
            * trace_fracstep    : fractional step in x for the patch discretization optimization
            * trace_xaxis       : x axis for the discretization ('x' or 'y')
            * trace_cum_error   : if True, account for accumulated error to define the x axis bound for the last patch

        Example: dip = [[0, 0, 20], [10, 10, 30], [80, 10, 90]] means that from the origin point of the fault (self.xi[0], self.yi[0]), the dip is 20 deg at 0 km and 0 km depth, 30 deg at km 10 and 10 km-depth and 90 deg at km 80 and 10 km-depth. The routine starts by discretizing the surface trace, then defines a dip evolution as a function of distance from the fault origin and drapes the fault down to depth.
        '''

        # Initialize the structures
        self.patch = []
        self.patchll = []
        self.slip = []
        self.patchdip = []

        # Build a 2d dip interpolator
        if dip is not None:
            import scipy.interpolate as sciint
            xy = np.array([ [dip[i][0], dip[i][1]] for i in range(len(dip))])
            dips = np.array([dip[i][2] for i in range(len(dip))])
            dipinterpolator = sciint.LinearNDInterpolator(xy, dips, fill_value=90.)      # If the points are not inside the area provided by the user, the dip will be 90 deg (vertical)
        else:
            def dipinterpolator(d, z):
                return 90.

        # Discretize the surface trace of the fault
        self.discretize(every,trace_tol,trace_fracstep,trace_xaxis,trace_cum_error)

        # degree to rad
        if dipdirection is not None:
            dipdirection_rad = dipdirection*np.pi/180.
            sdr = np.sin(dipdirection_rad)
            cdr = np.cos(dipdirection_rad)
        else:
            sdr = 0.
            cdr = 0.

        # initialize the depth of the top row
        self.zi = np.ones((self.xi.shape))*self.top

        # set a marker
        D = [self.top]

        # Loop over the depths
        for i in range(self.numz):

            # Get the depth of the top of the row
            zt = self.zi

            # Compute the dips for this row (it updates xi and yi at the same time)
            self.dipatZ(dipinterpolator, zt)
            self.dip *= np.pi/180.

            # Get the top of the row
            xt = self.xi
            yt = self.yi
            lont, latt = self.putm(xt*1000., yt*1000., inverse=True)
            zt = self.zi.round(decimals=10)

            # Compute the bottom row
            xb = xt + self.width*np.cos(self.dip)*sdr
            yb = yt + self.width*np.cos(self.dip)*cdr
            lonb, latb = self.putm(xb*1000., yb*1000., inverse=True)
            zb = zt + self.width*np.sin(self.dip)

            # fill D
            D.append(zb.max())

            # Build the patches by linking the points together
            for j in range(xt.shape[0]-1):
                # 1st corner
                x1 = xt[j]
                y1 = yt[j]
                z1 = zt[j]
                lon1 = lont[j]
                lat1 = latt[j]
                # 2nd corner
                x2 = xt[j+1]
                y2 = yt[j+1]
                z2 = zt[j+1]
                lon2 = lont[j+1]
                lat2 = latt[j+1]
                # 3rd corner
                x3 = xb[j+1]
                y3 = yb[j+1]
                z3 = zb[j+1]
                lon3 = lonb[j+1]
                lat3 = latb[j+1]
                # 4th corner 
                x4 = xb[j]
                y4 = yb[j]
                z4 = zb[j]
                lon4 = lonb[j]
                lat4 = latb[j]
                # Set points
                if y1>y2:
                    p2 = [x1, y1, z1]; p2ll = [lon1, lat1, z1]
                    p1 = [x2, y2, z2]; p1ll = [lon2, lat2, z2]
                    p4 = [x3, y3, z3]; p4ll = [lon3, lat3, z3]
                    p3 = [x4, y4, z4]; p3ll = [lon4, lat4, z4]
                else:
                    p1 = [x1, y1, z1]; p1ll = [lon1, lat1, z1]
                    p2 = [x2, y2, z2]; p2ll = [lon2, lat2, z2]
                    p3 = [x3, y3, z3]; p3ll = [lon3, lat3, z3]
                    p4 = [x4, y4, z4]; p4ll = [lon4, lat4, z4]
                # Store these
                psize = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
                if psize<minpatchsize:           # Increase the size of the previous patch
                    continue                # Breaks the loop and trashes the patch
                p = [p1, p2, p3, p4]
                pll = [p1ll, p2ll, p3ll, p4ll]
                p = np.array(p)
                pll = np.array(pll)
                # fill in the lists
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])
                self.patchdip.append(dip)

            # upgrade xi
            self.xi = xb
            self.yi = yb
            self.zi = zb

        # set depth
        D = np.array(D)
        self.z_patches = D
        self.depth = D.max()

        # Translate slip into an array
        self.slip = np.array(self.slip)

        # Re-discretoze to get the original fault
        self.discretize(every,trace_tol,trace_fracstep,trace_xaxis,trace_cum_error)

        # Compute the equivalent rectangles
        self.computeEquivRectangle()
    
        # All done
        return
    # ----------------------------------------------------------------------

#EOF
