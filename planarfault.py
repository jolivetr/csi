'''
A class that deals with simple vertical faults.

Written by Z. Duputel and R. Jolivet, January 2014
'''


# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys

# Rectangular patches Fault class
from .RectangularPatches import RectangularPatches

# Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok

class planarfault(RectangularPatches):

    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True, lon0=None, lat0=None):
        '''
        Args:
            * name          : Name of the fault.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''

        # Parent class init
        super(planarfault,self).__init__(name,
                                         utmzone=utmzone,
                                         ellps=ellps,
                                         lon0=lon0,
                                         lat0=lat0,
                                         verbose=verbose)
        
        # All done
        return

    def discretize(self, lon, lat, strike, length, n_strike):
        '''
        Define the discretized trace of the fault
        Args:
            * lat,lon: coordinates at the center of the top edge of the fault
            * strike: strike angle in degrees (from North)
            * length: length of the fault (i.e., along strike)
            * n_strike: number of patches along strike
        '''

        strike_rad = strike*np.pi/180.
        
        # Transpose lat/lon into the UTM reference
        xc, yc = self.ll2xy(lon,lat)
        
        # half-length
        half_length = 0.5*length

        # set x0, y0 (i.e., coordinates at one of the top corner of the fault)
        x0 = xc - half_length * np.sin(strike_rad)
        y0 = yc - half_length * np.cos(strike_rad)

        # set patch corners along strike
        dist_strike = np.linspace(0,length,n_strike+1)
        self.xi = x0 + dist_strike * np.sin(strike_rad)
        self.yi = y0 + dist_strike * np.cos(strike_rad)
        self.loni,self.lati = self.xy2ll(self.xi,self.yi)
        
        # Set trace attributes
        self.trace(self.loni,self.lati)

        # All done
        return

    def buildPatches(self, lon, lat, dep, strike, dip, f_length, f_width, n_strike, n_dip, verbose=True):
        '''
        Builds a dipping fault.
        Args:
            * lat,lon,dep: coordinates at the center of the top edge of the fault
            * strike: strike angle in degrees (from North)
            * f_length: length of the fault (i.e., along strike)
            * f_width: width of the fault (i.e., along dip)        
            * n_strike: number of patches along strike
            * n_dip: number of patches along dip
        '''
        
        # Print
        if verbose:
            print("Building a dipping fault")
            print("         Lat, Lon, Dep : {} deg, {} deg, {} km ".format(lat,lon,dep))
            print("         Strike Angle    : {} degrees".format(strike))
            print("         Dip Angle       : {} degrees".format(dip))
            print("         Dip Direction   : {} degrees".format(strike+90.))
            print("         Length          : {} km".format(f_length))
            print("         Width           : {} km".format(f_width))
            print("         {} patches along strike".format(n_strike))
            print("         {} patches along dip".format(n_dip))

        # Set depth patch attributes
        p_width = f_width/float(n_dip)
        self.setdepth(nump=n_dip,top=dep)
        
        # Initialize the structures
        self.patch        = []        
        self.patchll      = []
        self.equivpatch   = []
        self.equivpatchll = []
        self.slip         = []
        self.patchdip     = []
        
        # Discretize the surface trace of the fault
        self.discretize(lon,lat,strike,f_length,n_strike)
        
        # degree to rad
        dip_rad = dip*np.pi/180.
        dipdirection_rad = ((strike + 90)%360) * np.pi/180.#(-1.0*dipdirection+90)*np.pi/180.
        
        # initialize the depth of the top row
        self.zi = np.ones((self.xi.shape))*self.top
        
        # set a marker
        D = [self.top]
        
        # Loop over the depths
        for i in range(self.numz):
            
            # Get the top of the row
            xt = self.xi
            yt = self.yi
            zt = self.zi
            lont = self.loni
            latt = self.lati
            
            # Compute the bottom row
            xb = xt + p_width * np.cos(dip_rad) * np.sin(dipdirection_rad)
            yb = yt + p_width * np.cos(dip_rad) * np.cos(dipdirection_rad)
            lonb, latb = self.xy2ll(xb, yb)
            zb = zt + p_width*np.sin(dip_rad)
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
                p = [p1, p2, p3, p4]
                pll = [p1ll, p2ll, p3ll, p4ll]
                p = np.array(p)
                pll = np.array(pll)
                # fill in the lists
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])
                self.patchdip.append(dip_rad)
                # No equivalent patch calculation (patches are already rectangular)
                self.equivpatch.append(p)
                self.equivpatchll.append(pll)
                
            # upgrade top patches coordinates
            self.xi = xb
            self.yi = yb
            self.zi = zb
            self.loni, self.lati = self.xy2ll(xb,yb)            

        # set depth
        D = np.array(D)
        self.z_patches = D
        self.depth = D.max()
        
        # Translate slip into an array
        self.slip = np.array(self.slip)
        self.zi = np.ones((self.xi.shape))*self.top
        
        # Re-discretize to get the original fault
        self.discretize(lon,lat,strike,f_length,n_strike)

        # All done
        return


    def buildPatchesVarResolution(self, lon, lat, dep, strike, dip, f_length, f_width, 
                                  patch_lengths, patch_widths, interpolation='linear', verbose=True):
        '''
        Builds a dipping fault.
        Args:
            * lat,lon,dep: coordinates at the center of the top edge of the fault
            * strike: strike angle in degrees (from North)
            * f_length: length of the fault (i.e., along strike)
            * f_width: width of the fault (i.e., along dip)        
            * n_strike: number of patches along strike
            * n_dip: number of patches along dip
        '''
        
        # Print
        if verbose:
            print("Building a dipping fault")
            print("         Lat, Lon, Dep : {} deg, {} deg, {} km ".format(lat,lon,dep))
            print("         Strike Angle    : {} degrees".format(strike))
            print("         Dip Angle       : {} degrees".format(dip))
            print("         Dip Direction   : {} degrees".format(strike+90.))
            print("         Length          : {} km".format(f_length))
            print("         Width           : {} km".format(f_width))
        
        # Initialize the structures
        self.patch = []        
        self.patchll = []
        self.equivpatch = []
        self.equivpatchll = []
        self.slip = []
        self.patchdip = []

        # Top of the fault
        self.setdepth(nump=0, top=dep)

        # Dip direction - conversion to rad
        dip_rad = dip*np.pi/180.
        dipdirection_rad = ((strike + 90)%360) * np.pi/180.#(-1.0*dipdirection+90)*np.pi/180.

        # Interpolant function instantiation
        min_z  = self.top 
        max_z  = self.top + f_width     * np.sin(dip_rad)
        z_points    = np.array([min_z,max_z])
        fint_width  = sciint.interp1d(z_points, patch_widths , kind=interpolation)
        fint_length = sciint.interp1d(z_points, patch_lengths, kind=interpolation)

        # Loop over depths
        width     = 0.
        self.numz = 0 
        D = [self.top]
        while width < f_width:
            # Set depth patch attributes            
            patch_width  = fint_width(D[-1])
            patch_length = fint_length(D[-1])
            n_strike     = int(np.round(f_length/patch_length))
            
            # Discretize the surface trace of the fault
            self.discretize(lon,lat,strike,f_length,n_strike)
        
            # initialize the depth at the top of the fault
            self.zi = np.ones((self.xi.shape))*self.top
        
            # Get the top of the row
            xt = self.xi + width * np.cos(dip_rad) * np.sin(dipdirection_rad)
            yt = self.yi + width * np.cos(dip_rad) * np.cos(dipdirection_rad)
            zt = self.zi + width * np.sin(dip_rad)
            lont,latt = self.xy2ll(xt,yt)

            # Update the total fault width
            width += patch_width
                        
            # Compute the bottom row
            xb = self.xi + width * np.cos(dip_rad) * np.sin(dipdirection_rad)
            yb = self.yi + width * np.cos(dip_rad) * np.cos(dipdirection_rad)
            lonb, latb = self.xy2ll(xb, yb)
            zb = self.zi + width * np.sin(dip_rad)

            # fill D and update patches count
            D.append(zb.max())
            self.numz += 1
            
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
                p = [p1, p2, p3, p4]
                pll = [p1ll, p2ll, p3ll, p4ll]
                p = np.array(p)
                pll = np.array(pll)
                # fill in the lists
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])
                self.patchdip.append(dip_rad)
                # No equivalent patch calculation (patches are already rectangular)
                self.equivpatch.append(p)
                self.equivpatchll.append(pll)
                
        # set depth
        D = np.array(D)
        self.z_patches = D
        self.depth = D.max()
        
        # Translate slip into an array
        self.slip = np.array(self.slip)
        
        # Re-discretize to get the original fault
        patch_length = fint_length(min_z)
        n_strike     = int(np.round(f_length/patch_length))
        self.discretize(lon,lat,strike,f_length,n_strike)
        
        # All done
        return

    def moveFault(self, dx, dy, dz):
        '''
        Translates all the patches by dx, dy, dz.
        '''

        # Check if the fault will not fly in the air
        zmin = np.min(self.z_patches)
        if zmin+dz < 0.:
            dz = zmin

        # Move the fault
        for i in range(len(self.patch)):
            self.patch[i] += np.array([ [dx, dy, dz],
                                        [dx, dy, dz],
                                        [dx, dy, dz],
                                        [dx, dy, dz] ])

        # All done
        return

#EOF
