'''
A class that deals with blocks.

Written by E. Denise, March 2025
'''

# Import external packages
from math import dist
import os
import numpy as np
import copy
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
from shapely.validation import explain_validity
import pyproj

# Import internal packages
from .SourceInv import SourceInv
from .eulerPoleUtils import ERADIUS, MASYR2DEGMYR, MAS2RAD, llh2xyz, xyz2llh
from .geodeticplot import geodeticplot as geoplot

#class Block
class Block(SourceInv):

    '''
        Class implementing a Block object, based on the approach of
        B.J. Meade and J.P. Loveless (2009).

        You can specify either an official utm zone number or provide
        longitude and latitude for a custom zone (may be wrong for
        large blocks, but not used for block rotation and internal
        block deformation anyway).

        Args:
            * name          : Name of the block.
            * utmzone       : UTM zone  (optional, default=None)
            * lon0          : Longitude defining the center of the custom utm zone
            * lat0          : Latitude defining the center of the custom utm zone
            * ellps         : ellipsoid (optional, default='WGS84')
    '''

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Base class init
        super(Block, self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)

        # Initialize the block
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing block {}".format(self.name))
        self.verbose = verbose
        
        self.type = "Block"

        # Allocate block boundary attributes
        self.xf   = None # original non-regularly spaced coordinates (UTM)
        self.yf   = None
        self.xi   = None # regularly spaced coordinates (UTM)
        self.yi   = None
        self.loni = None # regularly spaced coordinates (geographical)
        self.lati = None
        self.lon  = None
        self.lat  = None
        
        # Reference points
        self.lonref = None
        self.latref = None
        self.xref = None
        self.yref = None
        
        # Allocate block rotation attributes
        self.pole_lon = None # Euler pole longitude (degrees)
        self.pole_lat = None # Euler pole latitude (degrees)
        self.pole_omega = None # Euler pole rotation rate (rad/yr)
        self.pole_x = None # Euler pole x coordinate (UTM) (km)
        self.pole_y = None # Euler pole y coordinate (UTM) (km)
        self.omega_x = None # rotation rate around x axis (rad/yr)
        self.omega_y = None # rotation rate around y axis (rad/yr)
        self.omega_z = None # rotation rate around z axis (rad/yr)
        
        # Allocate block internal deformation attributes
        # (horizontal strain rate tensor)
        self.eps_lonlon = None # horizontal strain rate lon/lon component (nstrain/time unit)
        self.eps_latlat = None # horizontal strain rate lon/lat component (nstrain/time unit)
        self.eps_lonlat = None # horizontal strain rate lat/lat component (nstrain/time unit)
        
        # A priori covariance matrix
        self.Cm = None

        # Create a dictionnary for the polysol
        self.polysol = {}

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set up what is needed for a block with no rotation
    def initializeEmptyBlock(self):
        '''
        Initializes what is required for a block with no rotation
        and no internal deformation.

        Returns: None
        '''

        # Initialize 
        self.omega_x = 0.
        self.omega_y = 0.
        self.omega_z = 0.
        
        self.pole_lon = 0.
        self.pole_lat = 0.
        self.pole_omega = 0.
        self.pole_x = 0.
        self.pole_y = 0.
        
        self.eps_lonlon = 0.
        self.eps_latlat = 0.
        self.eps_lonlat = 0.

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Returns a copy of the block
    def duplicateBlock(self):
        '''
        Returns a full copy (copy.deepcopy) of the block object.

        Return:
            * block         : block object
        '''

        # All done
        return copy.deepcopy(self)
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Convert the block boundary from lat/lon to UTM
    def boundary2xy(self):
        '''
        Transpose the block boundary coordinates in lon/lat into UTM.
        UTM coordinates are stored in self.xf and self.yf in km.

        Returns:
            * None
        '''

        # do it
        self.xf, self.yf = self.ll2xy(self.lon, self.lat)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Convert the block boundary from UTM to lat/lon
    def boundary2ll(self):
        '''
        Transpose the block boundary coordinates in UTM into lon/lat.
        Lon/lat coordinates are stored in self.lon and self.lat in degrees/

        Returns:
            * None
        '''

        # do it
        self.lon, self.lat = self.xy2ll(self.xf, self.yf)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the block boundary
    def setBoundary(self, lon=None, lat=None, x=None, y=None):
        '''
        Set the block boundary coordinates from lon/lat or UTM coordinates
        Block boundary is stored in self.xf, self.yf (UTM) and
        self.lon, self.lat (lon/lat).
        lon and lat must be provided if x and y are not and vice versa.

        Args:
            * lon           : Array/List containing the lon points (in degrees).
            * lat           : Array/List containing the lat points (in degrees).
            * x             : Array/List containing the UTM x points (in km).
            * y             : Array/List containing the UTM y points (in km).

        Returns:
            * None
        '''

        # Set lon and lat
        if lon is not None and lat is not None:
            
            # check if the boundary is closed
            if lon[0] != lon[-1] or lat[0] != lat[-1]:
                print("Warning: Block boundary is not closed. Adding a closing point.")
                lon = np.append(lon, lon[0])
                lat = np.append(lat, lat[0])
            
            # Check if there are successive duplicate points
            idx_delete = []
            for k in range (1, len(lon)):
                if lon[k] == lon[k-1] and lat[k] == lat[k-1]:
                    idx_delete.append(k)
            if len(idx_delete) > 0:
                print("Warning: Block boundary contains successive duplicate points. Removing them.")
                lon = np.delete(lon, idx_delete)
                lat = np.delete(lat, idx_delete)
            
            self.lon = np.array(lon)
            self.lat = np.array(lat)
            self.boundary2xy()
        
        elif x is not None and y is not None:
            
            # check if the boundary is closed
            if x[0] != x[-1] or y[0] != y[-1]:
                print("Warning: Block boundary is not closed. Adding a closing point.")
                x = np.append(x, x[0])
                y = np.append(y, y[0])
            
            # Check if there are successive duplicate points
            idx_delete = []
            for k in range (1, len(x)):
                if x[k] == x[k-1] and y[k] == y[k-1]:
                    idx_delete.append(k)
            if len(idx_delete) > 0:
                print("Warning: Block boundary contains successive duplicate points. Removing them.")
                x = np.delete(x, idx_delete)
                y = np.delete(y, idx_delete)
            
            self.xf = np.array(x) / 1000.
            self.yf = np.array(y) / 1000.
            self.boundary2ll()
        
        else:
            raise ValueError("Please provide either lon/lat or x/y coordinates")

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the block reference point
    def setRefPoint(self, lonref=None, latref=None, xref=None, yref=None, discretized=False):
        '''
        Set the block reference point from lon/lat or UTM coordinates
        If neither lonref/latref nor xref/yref are provided, the block
        reference point will be defined as the centroid of the block.
        
        The computation of the centroid is based on the following paper:
        Brock, J. E. (March 1, 1975). "The Inertia Tensor for a Spherical Triangle."
        ASME. J. Appl. Mech. March 1975; 42(1): 239. https://doi.org/10.1115/1.3423535
        The code is based on the following stackoverflow post:
        https://stackoverflow.com/a/38201499/21454544

        Args:
            * lonref        : float, longitude of the reference point (in degrees)
            * latref        : float, latitude of the reference point (in degrees)
            * xref          : float, UTM x coordinate of the reference point (in km)
            * yref          : float, UTM y coordinate of the reference point (in km)
            * discretized    : If True, use the discretized block boundary
        Returns:
            * None
        '''
        
        # Set the reference point (lon/lat)
        if lonref is not None and latref is not None:
            self.lonref = lonref
            self.latref = latref
            self.xref, self.yref = self.ll2xy(lonref, latref)
        
        # Set the reference point (UTM)
        elif xref is not None and yref is not None:
            self.xref = xref / 1000.
            self.yref = yref / 1000.
            self.lonref, self.latref = self.xy2ll(xref, yref)
        
        # If no reference point is provided, compute the centroid of the polygon
        else:
            
            def angle_between_unit_vectors(a, b):
                """
                Compute the angle between two unit vectors in radians.

                Args:
                    a (1D ndarray): vector of size (3,)
                    b (1D ndarray): vector of size (3,)

                Returns:
                    float: angle in radians between a and b
                """
                diff = a - b
                summ = a + b
                if np.dot(a, b) < 0:
                    return np.pi - 2 * np.arcsin(np.linalg.norm(summ) / 2)
                else:
                    return 2 * np.arcsin(np.linalg.norm(diff) / 2)
            
            # Use the discretized block boundary or not ?
            if discretized:
                lont = self.loni
                latt = self.lati
            else:
                lont = self.lon
                latt = self.lat
            
            # Compute coordinates of the polygon onto the unit sphere
            vertices = np.array([llh2xyz(lon=lon_,
                                         lat=lat_,
                                         height=0,
                                         earth_radius=1) for lon_, lat_ in zip(lont, latt)])

            # Compute the moment
            moment = np.zeros(3)
            n = len(vertices)
            for i in range(n-1):
                a = vertices[i]
                b = vertices[i+1]
                normal = np.cross(a, b)
                normal /= np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else 1
                angle = angle_between_unit_vectors(a, b) / 2
                moment += normal * angle
            
            # Compute the centroid
            direction = moment / np.linalg.norm(moment)
            lonref, latref, href = xyz2llh(x=direction[0],
                                           y=direction[1],
                                           z=direction[2],
                                           earth_radius=1)
        
            # Save the centroid
            self.lonref = lonref
            self.latref = latref
            
            # Convert to UTM
            self.xref, self.yref = self.ll2xy(self.lonref, self.latref) 
        
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the block boundary from a file
    def file2boundary(self, filename, utm=False, header=0):
        '''
        Reads the block boundary from a text file (ascii 2 columns)
            - If utm is False, format is lon lat (in degrees)
            - If utm is True, format is x y (in km)

        Args:
            * filename      : Name of the block boundary file.

        Kwargs:
            * utm           : Specify nature of coordinates
            * header        : Number of lines to skip at the beginning of the file

        Returns:
            * None
        '''
        
        # open file
        with open(filename, 'r') as f:
            
            lines = f.readlines()

            # store coordinates
            coord0, coord1 = [], []
            
            for k in range(header, len(lines)):
                coord0.append(float(lines[k].split()[0]))
                coord1.append(float(lines[k].split()[1]))

        # Create the boundary
        if utm:
            self.setBoundary(x=coord0, y=coord1)
        else:
            self.setBoundary(lon=coord0, lat=coord1)
        
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Discretize the block boundary
    def discretize(self, every=2.):
        '''
        Refine the block boundary by setting a constant distance between
        each point.
        Descretized boundary is stored in self.xi, self.yi,
        self.loni, and self.lati.

        Kwargs:
            * every         : Spacing between each point (in km)

        Returns:
            * None
        '''

        geod = pyproj.Geod(ellps="WGS84")

        def interpolate_geodesic_points(p1, p2, step_km):
            """
            Interpolates points along the geodesic between p1 and p2 every step_km kilometers.
            """
            lon1, lat1 = p1
            lon2, lat2 = p2

            # Compute total distance and azimuth between the two points
            az12, az21, dist_m = geod.inv(lon1, lat1, lon2, lat2)
            dist_km = dist_m * 1e-3

            # Number of intermediate points (not including endpoints)
            n_steps = int(dist_km / step_km)
            
            if n_steps == 0:
                return [p1, p2]
            else:
                # Generate points at step_km intervals
                coords = geod.npts(lon1, lat1, lon2, lat2, n_steps, terminus_idx=0)
                lons, lats = zip(*coords)
                return [(lon1, lat1)] + list(zip(lons, lats))

        # Create polygon from the block boundary
        polygon_block = Polygon(np.column_stack((self.lon, self.lat)))
        
        # Refine the polygon
        refined_coords = []
        coords = list(polygon_block.exterior.coords)

        for i in range(len(coords) - 1):
            seg_points = interpolate_geodesic_points(coords[i], coords[i+1], every)
            refined_coords.extend(seg_points[:-1])  # exclude last point to avoid duplication

        refined_coords.append(coords[-1])  # close the polygon

        # Save
        loni, lati = zip(*refined_coords)
        self.loni = np.array(loni)
        self.lati = np.array(lati)
        
        # Convert to UTM
        self.xi, self.yi = self.ll2xy(self.loni, self.lati)
        
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Compute the strike along the block boundary
    def strikeOfBoundary(self, discretized=True, npoints=4):
        '''
        Computes the strike of the block boundary from the discretized (default) 
        block boundary.

        Kwargs:
            * discretized       : Use the discretized block boundary (self.xi and self.yi) (default True)
            * npoints           : Number of points to average strike

        Return:
            * None. Stores the strike in self.strike
        '''

        # Get the block boundary
        if discretized:
            xt = self.xi
            yt = self.yi
        else:
            xt = self.xf
            yt = self.yf

        # Iterate over these guys
        strike = []
        for i in range(len(xt)):
            s = []
            for n in range(1, npoints):
                istart = np.max((0, i-n))
                iend = np.min((len(xt)-1, i+n))
                s.append(np.pi/2.-np.arctan2(yt[iend]-yt[istart],xt[iend]-xt[istart]))
            strike.append(np.mean(s))

        # Save strike 
        self.strike = np.array(strike)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Compute the area of the block
    def getAreaPerimeter(self, discretized=False):
        '''
        Computes the area of the block in km^2.

        Kwargs:
            * discretized       : Use the discretized block boundary (default False)

        Returns:
            * area              : Area of the block (km^2)
            * perimeter         : Perimeter of the block (km)
        '''

        # Get the block boundary
        if discretized:
            lont = self.loni
            latt = self.lati
        else:
            lont = self.lon
            latt = self.lat

        # Create polygon from the block boundary
        polygon_block = Polygon(np.column_stack((lont, latt)))

        # Compute area
        geod = pyproj.Geod(ellps='WGS84')
        poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon_block)
        area = poly_area / 1e6
        perimeter = poly_perimeter / 1e3
        
        # Save
        self.area = area
        self.perimeter = perimeter
        
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Compute the distance between a point and the block boundary
    def distance2boundary(self, lon, lat, discretized=False):
        '''
        Computes the distance between a point and the block boundary.
        This is a slow method, so it has been recoded in a few places
        throughout the whole library.

        Args:
            * lon               : Longitude of the point.
            * lat               : Latitude of the point.

        Kwargs:
            * discretized       : Uses the discretized block boundary.

        Returns:
            * distmin           : Shortest distance between the point and the boundary (km).
        '''

        # Block boundary coordinates
        if discretized:
            block_lon = self.loni
            block_lat = self.lati
        else:
            block_lon = self.lon
            block_lat = self.lat

        # Compute the distance
        geod = pyproj.Geod(ellps='WGS84')

        dist = geod.inv(lons1=np.full(block_lon.shape, lon),
                        lats1=np.full(block_lat.shape, lat),
                        lons2=block_lon,
                        lats2=block_lat)[2]
        
        distmin = np.nanmin(dist)
        
        # All done
        return distmin * 1e-3
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Compute the distance along the block boundary
    def distanceAlongBoundary(self, discretized=False):
        '''
        Computes the distance along the block boundary.
        
        Args:
            * discretized       : Use the discretized block boundary (default False)
        '''
        
        # Block boundary coordinates
        if discretized:
            block_lon = self.loni
            block_lat = self.lati
        else:
            block_lon = self.lon
            block_lat = self.lat
        
        # Compute the distance
        dist_step = np.zeros_like(block_lon)
        geod = pyproj.Geod(ellps="WGS84")
        
        for i in range(1, len(block_lon)):
            
            lon1, lat1 = block_lon[i-1], block_lat[i-1]
            lon2, lat2 = block_lon[i], block_lat[i]
            
            az12, az21, dist_m = geod.inv(lon1, lat1, lon2, lat2)
            
            dist_step[i] = dist_m * 1e-3 # km
        
        # Cumulative distance
        dist_along = np.cumsum(dist_step)
        
        # Save
        if discretized:
            self.dist_alongi = dist_along
        else:
            self.dist_along = dist_along
        
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Write the block boundary to a file
    def writeBoundary2File(self, filename, ref='lonlat', discretized=False):
        '''
        Writes the block boundary to a file. Format is ascii with two columns with
        either lon/lat (in degrees) or x/y (utm in km).

        Args:
            * filename      : Name of the file

        Kwargs:
            * ref           : can be lonlat or utm.
            * discretized   : Use the discretized block boundary (default False)

        Returns:
            * None
        '''

        # Get values
        if ref in ('utm') and not discretized:
            x = self.xf * 1000.
            y = self.yf * 1000.
        elif ref in ('utm') and discretized:
            x = self.xi * 1000.
            y = self.yi * 1000.
        elif ref in ('lonlat') and not discretized:
            x = self.lon
            y = self.lat
        elif ref in ('lonlat') and discretized:
            x = self.loni
            y = self.lati

        # Open file
        fout = open(filename, 'w')

        # Write
        for i in range(x.shape[0]):
            fout.write('{}\t{}\n'.format(x[i], y[i]))

        # Close file
        fout.close()

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Save data to file
    def saveData(self, dtype='d', outputDir='.'):
        '''
        Saves the Data in binary files.

        Kwargs:
            * dtype       : Format of the binary data saved. 'd' for double. 'f' for np.float32
            * outputDir   : Directory to save binary data

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing data to file for block {}'.format(self.name))

        # Loop over the data names in self.d
        for data in self.d.keys():

            # Get data
            D = self.d[data]

            # Write data file
            filename = '{}_{}.data'.format(self.name, data)
            D.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Save the Green's functions
    def saveGFs(self, dtype='d', outputDir='.', suffix_rot='rot', suffix_intradef='intradef'):
        '''
        Saves the Green's functions in different files.

        Kwargs:
            * dtype           : Format of the binary data saved
                                'd' for double
                                'f' for np.float32
            * outputDir       : Directory to save binary data.
            * suffix_rot      : Suffix for the rotation Green's functions
            * suffix_intradef : Suffix for the internal deformation Green's functions

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing Greens functions to file for block {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # Create one file for the rotation
            if 'rotation' in G.keys():
                g = G['rotation'].flatten()
                n = self.name.replace(' ', '_')
                d = data.replace(' ', '_')
                filename = '{}_{}_{}.gf'.format(n, d, suffix_rot)
                g = g.astype(dtype)
                g.tofile(os.path.join(outputDir, filename))
            
            # Create one file for the internal deformation
            if 'intradef' in G.keys():
                g = G['intradef'].flatten()
                n = self.name.replace(' ', '_')
                d = data.replace(' ', '_')
                filename = '{}_{}_{}.gf'.format(n, d, suffix_intradef)
                g = g.astype(dtype)
                g.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Check if a point is inside the block
    def PointInBlock(self, x, y, discretized=False, coord='ll'):
        '''
        Returns True if the point is inside the block, False otherwise.

        Args:
            * x                 : Longitude or x-coordinate of the point (utm).
            * y                 : Latitude or y-coordinate of the point (utm).

        Kwargs:
            * discretized       : Uses the discretized block boundary.
            * coord             : if 'll' or 'lonlat', input in degree. If 'xy' or 'utm', input in km

        Returns:
            * True/False        : True if the point is inside the block, False otherwise.
        '''
        
        
        
        # Create the polygon
        if coord in ('ll', 'lonlat'):            
            
            # Get coordinates
            if discretized:
                lons, lats = np.array(self.loni), np.array(self.lati)
            else:
                lons, lats = np.array(self.lon), np.array(self.lat)
            polygon_block = Polygon(np.column_stack((lons, lats)))
            
            # Check for antimeridian crossing
            if np.any(lons > 180):
                # shift to -180 to 180 range
                lons = np.where(lons > 180, lons - 360, lons)
            if x > 180:
                x_ = x - 360
            else:
                x_ = x
            y_ = y
            
        elif coord in ('xy', 'utm'):
            
            # Get coordinates
            if discretized:
                polygon_block = Polygon(np.column_stack((self.xi, self.yi)))
            else:
                polygon_block = Polygon(np.column_stack((self.xf, self.yf)))
            
            x_, y_ = x, y
        
        # Create the point
        point = Point(x_, y_)
        
        # Check if point in block
        from shapely import within
        return within(point, polygon_block)
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the block rotation
    def rotation2pole(self):
        '''
        Convert the rotation vector to an Euler pole.
        The rotation vector is oriented from the center of the Earth to the Euler pole.
        The rotation is positive for counterclockwise rotation.

        Returns:
            * None
        '''

        # Convert rotation vector to Euler pole
        self.pole_lon, self.pole_lat, self.pole_omega = xyz2llh(self.omega_x, self.omega_y, self.omega_z, earth_radius=0.)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the block rotation
    def rotation2vector(self):
        '''
        Convert the Euler pole to a rotation vector.
        The rotation vector is oriented from the center of the Earth to the Euler pole.
        The rotation is positive for counterclockwise rotation.

        Returns:
            * None
        '''

        # Convert the Euler pole to rotation vector
        self.omega_x, self.omega_y, self.omega_z = llh2xyz(self.pole_lon, self.pole_lat, self.pole_omega, earth_radius=0.)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the block rotation
    def set_rotation(self, omega_x=None, omega_y=None, omega_z=None,
                     pole_lat=None, pole_lon=None, rot_rate=None,
                     unit='mas/yr'):
        '''
        Set the block rotation parameters. Two possibilities:
        - Provide the rotation rates around x, y, z axis, omega_x, omega_y and omega_z (mas/yr or deg/Myr).
        - Provide the Euler pole coordinates, pol_lon/pole_lat, and the rotation rate rot_rate (mas/yr or deg/Myr).
        The rotation vector is oriented from the center of the Earth to the Euler pole.
        The rotation is positive for counterclockwise rotation.

        Args:
            * omega_x       : float, rotation rate around x axis (mas/yr or deg/My, specified by unit)
            * omega_y       : float, rotation rate around y axis (mas/yr or deg/My, specified by unit)
            * omega_z       : float, rotation rate around z axis (mas/yr or deg/My, specified by unit)
            * pole_lon      : float, longitude of the rotation pole (degrees)
            * pole_lat      : float, latitude of the rotation pole (degrees)
            * rot_rate      : float, rotation rate (mas/yr or deg/My, specified by unit)
            * unit          :

        Returns:
            * None
        '''
        
        # Convert rotation to mas/yr
        if unit == 'deg/Myr':
            omega_x = omega_x / MASYR2DEGMYR if omega_x is not None else None
            omega_y = omega_y / MASYR2DEGMYR if omega_y is not None else None
            omega_z = omega_z / MASYR2DEGMYR if omega_z is not None else None
            rot_rate = rot_rate / MASYR2DEGMYR if rot_rate is not None else None

        # Convert to rad/yr
        omega_x = omega_x * MAS2RAD if omega_x is not None else None
        omega_y = omega_y * MAS2RAD if omega_y is not None else None
        omega_z = omega_z * MAS2RAD if omega_z is not None else None
        rot_rate = rot_rate * MAS2RAD if rot_rate is not None else None
        
        # Set rotation parameters
        if omega_x is not None and omega_y is not None and omega_z is not None:
            # Set rotation vector
            self.omega_x = omega_x
            self.omega_y = omega_y
            self.omega_z = omega_z
            
            # Convert rotation vector to Euler pole
            self.rotation2pole()
        
        elif pole_lon is not None and pole_lat is not None and rot_rate is not None:
            # Set Euler pole
            self.pole_lon = pole_lon
            self.pole_lat = pole_lat
            self.pole_omega = rot_rate

            # TODO Convert Euler pole to rotation vector
            self.rotation2vector()
        
        else:
            raise ValueError("Please provide either omega_x, omega_y and omega_z or pole_lon, pole_lat and rot_rate")

        # Convert pole lon/lat to UTM
        pole_x, pole_y = self.ll2xy(self.pole_lon, self.pole_lat)
        self.pole_x, self.pole_y = pole_x, pole_y

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the block horizontal strain rate tensor
    def setStrainTensor(self, eps_lonlon=None, eps_latlat=None, eps_lonlat=None):
        '''
        Set the block spherical horizontal strain rate tensor.

        Args:
            * eps_lonlon    : float, horizontal strain rate tensor, lon/lon component (nstrain/time unit)
            * eps_latlat    : float, horizontal strain rate tensor, lat/lat component (nstrain/time unit)
            * eps_lonlat    : float, horizontal strain rate tensor, lon/lat component (nstrain/time unit)

        Returns:
            * None
        '''

        # Set horizontal strain rate tensor
        self.eps_lonlon = eps_lonlon # nanostrain/time unit
        self.eps_latlat = eps_latlat # nanostrain/time unit
        self.eps_lonlat = eps_lonlat # nanostrain/time unit

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Compute the second invariant of the strain rate tensor 
    def calcSecInv(self):
        '''
        Computes the second invariant of the strain rate tensor.

        Returns:
            * None
        '''

        # Compute the second invariant
        self.eps_secinv = np.sqrt(self.eps_lonlon**2 + self.eps_latlat**2 + 2 * self.eps_lonlat**2)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Show the block boundary
    def plot(self, show=True, figsize=(10, 10),
             box=None, shadedtopo=None, drawCoastlines=True, expand=0.2, savefig=False):
        '''
        Plot the available elements of the block.

        Kwargs:
            * figure        : Number of the figure.
            * equiv         : useless. For consitency between fault objects
            * show          : Show me
            * drawCoastline : Self-explanatory argument...
            * expand        : Expand the map by {expand} degree around the edges
                              of the blockboundary.
            * savefig       : Save figures as eps.

        Returns:
            * None
        '''

        # Get extent of the plot
        lonmin = np.min(self.lon)-expand
        lonmax = np.max(self.lon)+expand
        latmin = np.min(self.lat)-expand
        latmax = np.max(self.lat)+expand

        # Override
        if box is not None:
            assert len(box)==4, 'box must be 4 floats: box = {}'.format(tuple(box))
            lonmin, lonmax, latmin, latmax = box

        # Create a figure
        fig = geoplot(figure=None,
                      lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax,
                      figsize=(None, figsize), Map=True, Fault=False)

        # Shaded topo
        if shadedtopo is not None:
            fig.shadedTopography(**shadedtopo)
        
        # Draw the coastlines
        if drawCoastlines:
            fig.drawCoastlines(parallels=None, meridians=None, drawOnFault=True)

        # Block boundary trace
        color = self.color if hasattr(self, 'color') else 'k'
        linewidth = self.linewidth if hasattr(self, 'linewidth') else 1.
        linestyle = self.linestyle if hasattr(self, 'linestyle') else 'solid'

        fig.blockboundary(self, color=color, linewidth=linewidth, linestyle=linestyle)

        # Savefigs?
        if savefig:
            prefix = self.name.replace(' ','_')
            fig.savefig(prefix, ftype='eps')

        # show
        if show:
            fig.show(showFig=['map'])

        # Save the figure
        self.fig = fig

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the Green's functions
    def buildGFs(self, data, method='rotation', vertical=True, verbose=True, override_inblock=False):
        '''
        Builds the Green's function matrix.

        The Green's function matrix is stored in a dictionary.
        Each entry of the dictionary is named after the corresponding dataset.

        Args:
            * data          : data object (gps, insar, surfaceslip)
        
        Kwargs:
            * method        : method to compute the Green's functions
                                (combination of 'rotation', 'intradef', 'emptyrotation' and 'emptyintradef')
                                ex: 'rotation+emptyintradef'
            * vertical      : If True, will produce Green's functions for the vertical displacements in a gps object.
            * verbose       : If True, will print out information

        Returns:
            * None
        '''

        # Data type check
        if data.dtype not in ('gps', 'insar', 'surfaceslip'):
            raise NotImplementedError("Only GNSS, InSAR and surfaceslip data are supported.")
        
        # Data type check
        if data.dtype == 'insar':
            if not vertical:
                if verbose:
                    print('---------------------------------')
                    print('---------------------------------')
                    print(' WARNING WARNING WARNING WARNING ')
                    print('  You specified vertical=False   ')
                    print(' As this is quite dangerous, we  ')
                    print(' switched it directly to True... ')
                    print(' SAR data are very sensitive to  ')
                    print('     vertical displacements.     ')
                    print(' WARNING WARNING WARNING WARNING ')
                    print('---------------------------------')
                    print('---------------------------------')
                vertical = True

        # Compute the Green's functions
        
        Grot = None
        if 'rotation' in method and not 'emptyrotation' in method:
            Grot = self.rotationGFs(data, verbose=verbose, override_inblock=override_inblock)
        elif 'emptyrotation' in method:
            Grot = self.emptyrotationGFs(data, verbose=verbose)
        
        Gint = None
        if 'intradef' in method and not 'emptyintradef' in method:
            Gint = self.intradefGFs(data, verbose=verbose, override_inblock=override_inblock)
        elif 'emptyintradef' in method:
            Gint = self.emptyintradefGFs(data, verbose=verbose)
        
        if Grot is None and Gint is None:
            raise NotImplementedError("Method {} not supported".format(method))

        # Build the dictionary
        G = self._buildGFsdict(data, Grot=Grot, Gint=Gint, vertical=vertical)

        # Separate the Green's functions for each type of data set
        data.setGFsInSource(self, G, vertical=vertical)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build zero GFs for block rotation
    def emptyrotationGFs(self, data, verbose=True):
        '''
        Build zero rotation GFs.

        Args:
            * data          : Data object (gps, insar, surfaceslip)

        Kwargs:
            * verbose       : If True, will print out information.

        Returns:
            * G             : Dictionnary of GFs
        '''
        
        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building empty block rotation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))

        # Create the GF matrices
        if data.dtype in ('insar', 'gps', 'surfaceslip'):
            
            nd = len(data.lon)
            Grot = np.zeros((3, nd, 3))
        
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))

        # All done
        return Grot
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the rotation GFs
    def rotationGFs(self, data, verbose=True, override_inblock=False):
        '''
        Build the Green's functions for a block rotation.

        Args:
            * data          : Data object (gps, insar, surfaceslip)

        Kwargs:
            * verbose       : If True, will print out information.
            * override_inblock : If True, will compute GFs for all points regardless of whether they are inside the block or not.

        Returns:
            * G             : Dictionnary of GFs
        '''
        
        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building block rotation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))

        # Create the GF matrices
        if data.dtype in ('insar', 'gps', 'surfaceslip'):
            
            # Initialize the Green's functions
            Grot = np.zeros((3, len(data.lon), 3))
            
            # Loop through data points
            for k in range(len(data.lon)):
                
                # Check if point in block
                if not self.PointInBlock(data.lon[k], data.lat[k], discretized=False, coord='ll') and not override_inblock:
                    continue
                
                # Compute the Green's functions
                x_, y_, z_ = llh2xyz(lon=data.lon[k] if data.lon[k] <= 180 else data.lon[k]-360,
                                     lat=data.lat[k],
                                     height=0.)
                lon_ = np.deg2rad(data.lon[k] if data.lon[k] <= 180 else data.lon[k]-360)
                lat_ = np.deg2rad(data.lat[k])
                
                Pv = np.array([[-np.sin(lon_), np.cos(lon_), 0.],
                              [-np.sin(lat_) * np.cos(lon_), -np.sin(lat_) * np.sin(lon_), np.cos(lat_)],
                              [np.cos(lat_) * np.cos(lon_), np.cos(lat_) * np.sin(lon_), np.sin(lat_)]])
                
                Gb = np.array([[0., z_, -y_],
                              [-z_, 0., x_],
                              [y_, -x_, 0.]])

                G_ = Pv @ Gb
                G_ *= MAS2RAD
                
                # Assign the Green's functions
                Grot[0, k, :] = G_[0, :]
                Grot[1, k, :] = G_[1, :]
                Grot[2, k, :] = G_[2, :]
        
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))
    
        # All done
        return Grot
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Get the velocity due to block rotation at given points
    def rot2disp(self, lon, lat, z):
        '''
        Compute the ENU displacement due to block rotation 
        in the east/north/up direction at a given point lon/lat/z.
        
        Args:
            * lon           : Longitude of the point.s (in degrees)
            * lat           : Latitude of the point.s (in degrees)
            * z             : Height of the point.s (in m)
        
        Returns:
            * vel           : Velocity in the east/north/up direction (in km/yr)
        '''
        
        # Get the rotation vector
        Ω = np.array([self.omega_x, self.omega_y, self.omega_z])
        
        vel = np.zeros((len(lon), 3))
        
        for k in range(len(lon)):
            
            # Convert to cartesian
            x_, y_, z_ = llh2xyz(lon=lon[k] if lon[k] <= 180 else lon[k]-360,
                                 lat=lat[k],
                                 height=z[k])
            
            # Convert to radians
            lon_ = np.deg2rad(lon[k] if lon[k] <= 180 else lon[k]-360)
            lat_ = np.deg2rad(lat[k])
            
            # Compute velocities
            A = np.array([[-np.sin(lon_), np.cos(lon_), 0.],
                          [-np.sin(lat_) * np.cos(lon_), -np.sin(lat_) * np.sin(lon_), np.cos(lat_)],
                          [np.cos(lat_) * np.cos(lon_), np.cos(lat_) * np.sin(lon_), np.sin(lat_)]])
            
            B = np.array([[0., z_, -y_],
                          [-z_, 0., x_],
                          [y_, -x_, 0.]])
        
            G = A @ B
            vel[k, :] = G @ Ω
        
        # All done
        return vel
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build zero GFs for intrablock deformation
    def emptyintradefGFs(self, data, verbose=True):
        ''' 
        Build zero intrablock deformation GFs.

        Args:
            * data          : Data object (gps, insar, surfaceslip)

        Kwargs:
            * verbose       : If True, will print out information.

        Returns:
            * G             : Dictionnary of GFs
        '''
        
        # Print
        if verbose:           
            print('---------------------------------')
            print('---------------------------------')
            print("Building empty internal block deformation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))

        # Create the GF matrices
        if data.dtype in ('insar', 'gps', 'surfaceslip'):
            
            nd = len(data.lon)
            Gint = np.zeros((3, nd, 3))
        
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))
    
        # All done
        return Gint
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the internal deformation GFs
    def intradefGFs(self, data, verbose=True, override_inblock=False):
        '''
        Build the Green's functions for a block internal deformation following the approach of
        B.J. Meade and J.P. Loveless (2009).

        Args:
            * data          : Data object (gps, insar, surfaceslip)

        Kwargs:
            * verbose       : If True, will print out information.
            * override_inblock : If True, will compute GFs for all points regardless of whether they are inside the block or not.

        Returns:
            * G             : GFs
        '''
        
        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building internal block deformation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))

        # Create the GF matrices
        if data.dtype in ('insar', 'gps', 'surfaceslip'):
            
            # Initialize the Green's functions
            Gint = np.zeros((3, len(data.lon), 3))
            
            # Loop through data points
            for k in range(len(data.lon)):
                
                # Check if point in block
                if not self.PointInBlock(data.lon[k], data.lat[k], discretized=False, coord='ll') and not override_inblock:
                    continue
                
                # Compute the Green's functions
                lon_ = np.deg2rad(data.lon[k] if data.lon[k] <= 180 else data.lon[k]-360)
                lat_ = np.deg2rad(data.lat[k])
                
                lonref_ = np.deg2rad(self.lonref if self.lonref <= 180 else self.lonref-360)
                latref_ = np.deg2rad(self.latref)
                
                G_ = ERADIUS * np.array([[(lon_ - lonref_) * np.cos(latref_), latref_ - lat_, 0.],
                                         [0., (lon_ - lonref_) * np.cos(latref_), latref_ - lat_],
                                         [0., 0., 0.]])
                
                G_ *= 1e-9 # Consistency with nanostrain
                
                # Assign the Green's Functions
                Gint[0, k, :] = G_[0, :]
                Gint[1, k, :] = G_[1, :]
                Gint[2, k, :] = G_[2, :]
        
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))
    
        # All done
        return Gint
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the Green's functions dictionary
    def _buildGFsdict(self, data, Grot=None, Gint=None, vertical=True):
        '''
        Some ordering of the Gfs to make the computation routines simpler.

        Args:
            * data          : instance of data
            * Grot          : Block rotation Green's functions
            * Gint          : Block internal deformation Green's functions

        Kwargs:
            * vertical      : If true, assumes verticals are used for the GPS case

        Returns:
            * G             : Dictionary of GFs
        '''

        # Check if vertical
        Ncomp = 3
        if not vertical:
            Ncomp = 2

        # Block rotation
        if Grot is not None:

            # Consider vertical or not
            Grot = Grot[:Ncomp, :, :]
            
            # Size
            Npoints = Grot.shape[1]
            Nparam = Grot.shape[2]
            Ndata = Ncomp * Npoints
            
            # Check data format

            # InSAR or SurfaceSlip
            if hasattr(data, 'los') and data.los is not None:
                # Project the Green's functions onto the LOS direction
                Grot_los = []
                for k in range(Npoints):
                    Grot_los.append((data.los[k, :] @ Grot[:, k, :]).tolist())
                Grot = np.array(Grot_los).reshape((Npoints, Nparam))
            
            # GNSS or SurfaceSlip
            else:
                Grot = Grot.reshape((Ndata, Nparam))
        
        # Block internal deformation
        if Gint is not None:
            
            # Consider vertical or not
            Gint = Gint[:Ncomp, :, :]
            
            # Size
            Npoints = Gint.shape[1]
            Nparam = Gint.shape[2]
            Ndata = Ncomp * Npoints
            
            # Check data format
            
            # InSAR or SurfaceSlip
            if hasattr(data, 'los') and data.los is not None:
                # Project the Green's functions onto the LOS direction
                Gint_los = []
                for k in range(Npoints):
                    Gint_los.append((data.los[k, :] @ Gint[:, k, :]).tolist())
                Gint = np.array(Gint_los).reshape((Npoints, Nparam))
            
            # GNSS or SurfaceSlip
            else:
                Gint = Gint.reshape((Ndata, Nparam))
        
        # Create the dictionary
        G = {'rotation': Grot, 'intradef': Gint}

        # All done
        return G
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the Green's functions
    def setGFs(self, data,
               rotation=[None, None, None],
               intradef=[None, None, None],
               vertical=False, synthetic=False):
        '''
        Stores the input Green's functions matrices into the block source structure.

        These GFs are organized in a dictionary structure in self.G
        Entries of self.G are the data set names (data.name).
            Entries of self.G[data.name] are 'rotation'.

        If you provide GPS GFs, those are organised with E, N and U in lines

        If you provide InSAR GFs, these need to be projected onto the
        LOS direction already.

        Args:
            * data          : data structure
            * rotation      : Green's functions for block rotation displacements.
            * intradef      : Green's functions for internal block deformation displacements.
            
        Returns:
            * None
        '''

        # Get the number of data per point
        if data.dtype in ('insar', 'surfaceslip'):
            data.obs_per_station = 1
            
        elif data.dtype in ('gps'):
            data.obs_per_station = 0
            # Check components
            if not np.isnan(data.vel_enu[:, 0]).any():
                data.obs_per_station += 1
            if not np.isnan(data.vel_enu[:, 1]).any():
                data.obs_per_station += 1
            if vertical:
                if np.isnan(data.vel_enu[:, 2]).any():
                    raise ValueError('Vertical can only be true if all stations have vertical components')
                data.obs_per_station += 1

        # Create the storage for that dataset
        if data.name not in self.G.keys():
            self.G[data.name] = {}

        # Initializes the data vector
        if not synthetic:
            if data.dtype in ('insar', 'surfaceslip'):
                self.d[data.name] = data.vel
                vertical = True # Always true for InSAR
            elif data.dtype == 'gps':
                if vertical:
                    self.d[data.name] = data.vel_enu.T.flatten()
                else:
                    self.d[data.name] = data.vel_enu[:, :2].T.flatten()
                self.d[data.name] = self.d[data.name][np.isfinite(self.d[data.name])]

        if len(rotation) == 3: # gnss case
            
            # Block rotation
            E_rot = rotation[0]
            N_rot = rotation[1]
            U_rot = rotation[2]
            rot = []
            nd = 0
            
            if (E_rot is not None) and (N_rot is not None):
                d, m = E_rot.shape
                rot.append(E_rot)
                rot.append(N_rot)
                nd += 2
            if (U_rot is not None):
                d, m = U_rot.shape
                rot.append(U_rot)
                nd += 1
            if nd > 0:
                rot = np.array(rot)
                rot = rot.reshape((nd*d, m))
                self.G[data.name]['rotation'] = rot
            
            # Internal block deformation
            E_intra = intradef[0]
            N_intra = intradef[1]
            U_intra = intradef[2]
            intra = []
            nd = 0
            
            if (E_intra is not None) and (N_intra is not None):
                d, m = E_intra.shape
                intra.append(E_intra)
                intra.append(N_intra)
                nd += 2
            if (U_intra is not None):
                d, m = U_intra.shape
                intra.append(U_intra)
                nd += 1
            if nd > 0:
                intra = np.array(intra)
                intra = intra.reshape((nd*d, m))
                self.G[data.name]['intradef'] = intra
            
        elif len(rotation) == 1: # insar or surfaceslip case
            
            # Block rotation
            rot = rotation[0]
            if rot is not None:
                self.G[data.name]['rotation'] = rot
            
            # Internal block deformation
            intra = intradef[0]
            if intra is not None:
                self.G[data.name]['intradef'] = intra

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the Green's functions from file
    def setGFsFromFile(self, data, rotation=None, intradef=None, custom=None,
                       vertical=False, dtype='d', inDir='.'):
        '''
        Sets the Green's functions reading binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already). Basically, it will work better if
        you have computed the GFs using csi...

        Args:
            * data          : Data object

        kwargs:
            * rotation      : File containing the Green's functions for block rotation displacements.
            * intradef      : File containing the Green's functions for internal block deformation displacements.
            * vertical      : Deal with the UP component (gps: default is false, insar: it will be true anyway).
            * dtype         : Type of binary data. 'd' for double/float64. 'f' for np.float32

        Returns:
            * None
        '''
        
        # Check filenames
        if rotation is None:
            if os.path.isfile(os.path.join(inDir,'{}_{}_rot.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                rotation = os.path.join(inDir, '{}_{}_rot.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))
        
        if intradef is None:
            if os.path.isfile(os.path.join(inDir,'{}_{}_intradef.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                intradef = os.path.join(inDir, '{}_{}_intradef.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))

        
        # Load the block rotation Green's functions
        Grot = None
        if rotation is not None:
            
            # Talk to me
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print("Set up Green's functions for block {}".format(self.name))
                print("and data {} from files: ".format(data.name))
                print("     rotation: {}".format(rotation))    
            
            Grot = np.fromfile(rotation, dtype=dtype)
            ndl = int(Grot.shape[0])
            Grot = Grot.reshape((int(ndl/3), 3))
            
        # Load the internal deformation Green's functions
        Gint = None
        if intradef is not None:
            
            # Talk to me
            if self.verbose:
                print("     intradef: {}".format(intradef))
            
            Gint = np.fromfile(intradef, dtype=dtype)
            ndl = int(Gint.shape[0])
            Gint = Gint.reshape((int(ndl/3), 3))

        # Create the dictionary
        G = {'rotation': Grot, 'intradef': Gint}

        # The dataset sets the Green's functions itself
        data.setGFsInSource(self, G, vertical=vertical)

        # If custom
        if custom is not None:
            self.setCustomGFs(data, custom)

        # all done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set custom Green's functions
    def setCustomGFs(self, data, G):
        '''
        Sets a custom Green's Functions matrix in the G dictionary.

        Args:
            * data          : Data concerned by the Green's function
            * G             : Green's function matrix

        Returns:
            * None
        '''

        # Check
        if not hasattr(self, 'G'):
            self.G = {}

        # Check
        if not data.name in self.G.keys():
            self.G[data.name] = {}

        # Set
        self.G[data.name]['custom'] = G

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Assemble the Green's functions
    def assembleGFs(self, datas, polys=None, verbose=True, blockcomponent='rotation+intradef',
                    custom=False, computeNormFact=True):
        '''
        Assemble the Green's functions corresponding to the data in datas.
        This method allows to specify which transformation is going
        to be estimated in the data sets, through the polys argument.

        Assembled Green's function matrix is stored in self.Gassembled

        Args:
            * datas             : list of data sets. If only one data set is
                                  used, can be a data instance only.

        Kwargs:
            * polys             : None -> nothing additional is estimated

                 For InSAR, Optical, GPS:
                       1 -> estimate a constant offset
                       3 -> estimate z = ax + by + c
                       4 -> estimate z = axy + bx + cy + d

                 For GPS only:
                       'full'                -> Estimates a rotation,
                                                translation and scaling
                                                (Helmert transform).
                       'strain'              -> Estimates the full strain
                                                tensor (Rotation + Translation
                                                + Internal strain)
                       'strainnorotation'    -> Estimates the strain tensor and a
                                                translation
                       'strainonly'          -> Estimates the strain tensor
                       'strainnotranslation' -> Estimates the strain tensor and a
                                                rotation
                       'translation'         -> Estimates the translation
                       'translationrotation  -> Estimates the translation and a
                                                rotation

            * custom            : If True, gets the additional Green's function
                                  from the dictionary self.G[data.name]['custom']

            * blockcomponent   : str, which block components to consider
                any combination of 'rotation' and 'intradef'

            * computeNormFact   : bool
                if True, compute new OrbNormalizingFactor
                if False, uses parameters in self.OrbNormalizingFactor

            * verbose           : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''
        
        # Check
        if type(datas) is not list:
            datas = [datas]

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print("Assembling G for block {}".format(self.name))
            
        # Store block components to consider
        self.blockcomponent = blockcomponent

        # Create a dictionary to keep track of the orbital froms
        self.poly = {}
        
        # Set poly right
        if polys.__class__ is not list:
            for data in datas:
                if (polys.__class__ is not str) and (polys is not None):
                    self.poly[data.name] = polys*data.obs_per_station
                else:
                    self.poly[data.name] = polys
        elif polys.__class__ is list:
            for data, poly in zip(datas, polys):
                if (poly.__class__ is not str) and (poly is not None) and (poly.__class__ is not list):
                    self.poly[data.name] = poly*data.obs_per_station
                else:
                    self.poly[data.name] = poly

        # Create the transformation holder
        if not hasattr(self, 'helmert'):
            self.helmert = {}
        if not hasattr(self, 'strain'):
            self.strain = {}
        if not hasattr(self, 'transformation'):
            self.transformation = {}
        
        # Get the number of block parameters
        
        Nps = 0
        if 'rotation' in self.G[datas[0].name].keys() and 'rotation' in blockcomponent:
            Nps += 3
        if 'intradef' in self.G[datas[0].name].keys() and 'intradef' in blockcomponent:
            Nps += 3
        
        # Get the number of transformation parameters
        Npo = 0
        for data in datas :
            transformation = self.poly[data.name]
            if type(transformation) in (str, list):
                tmpNpo = data.getNumberOfTransformParameters(self.poly[data.name])
                Npo += tmpNpo
                if type(transformation) is str:
                    if transformation in ('full'):
                        self.helmert[data.name] = tmpNpo
                    elif transformation in ('strain', 'strainonly',
                                            'strainnorotation', 'strainnotranslation',
                                            'translation', 'translationrotation'):
                        self.strain[data.name] = tmpNpo
                else:
                    self.transformation[data.name] = tmpNpo
            elif transformation is not None:
                Npo += transformation
        Np = Nps + Npo
        
        # Save extra Parameters
        self.TransformationParameters = Npo
        
        # Custom ?
        if custom:
            Npc = 0
            for data in datas:
                if 'custom' in self.G[data.name].keys():
                    Npc += self.G[data.name]['custom'].shape[1]
            Np += Npc
            self.NumberCustom = Npc
        else:
            Npc = 0
        
        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Allocate G and d
        G = np.zeros((Nd, Np))

        # Create the list of data names, to keep track of it
        self.datanames = []
        
        # Loop over the datasets
        el = 0
        custstart = Nps # custom indices
        polstart = Nps + Npc # poly indices
        
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            if verbose:
                print("Dealing with {} of type {}".format(data.name, data.dtype))

            # Rotation Green's functions

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Npslocal = 0
            
            # Check if we have rotation GFs
            if 'rotation' in self.G[data.name].keys() and 'rotation' in self.blockcomponent:
                Glocal = self.G[data.name]['rotation']
                Npslocal = 3
            
                # Put Glocal into the big G
                G[el:el+Ndlocal, :Npslocal] = Glocal
            
            # Check if we have internal block deformation GFs
            if 'intradef' in self.G[data.name].keys() and 'intradef' in self.blockcomponent:
                Glocal = self.G[data.name]['intradef']
                
                # Put Glocal into the big G
                G[el:el+Ndlocal, Npslocal:Npslocal+3] = Glocal
            
            # Custom
            if custom:
                # Check if data has custom GFs
                if 'custom' in self.G[data.name].keys():
                    nc = self.G[data.name]['custom'].shape[1] # Nb of custom param
                    custend = custstart + nc
                    G[el:el+Ndlocal, custstart:custend] = self.G[data.name]['custom']
                    custstart += nc

            # Polynomes and strain
            if self.poly[data.name] is not None:

                # Build the polynomial function
                if data.dtype in ('gps', 'multigps'):
                    orb = data.getTransformEstimator(self.poly[data.name])
                elif data.dtype in ('insar', 'opticorr'):
                    orb = data.getPolyEstimator(self.poly[data.name],computeNormFact=computeNormFact)
                elif data.dtype == 'tsunami':
                    orb = data.getRampEstimator(self.poly[data.name])

                # Number of columns
                nc = orb.shape[1]

                # Put it into G for as much observable per station we have
                polend = polstart + nc
                G[el:el+Ndlocal, polstart:polend] = orb
                polstart += nc

            # Update el to check where we are
            el = el + Ndlocal

        # Store G in self
        self.Gassembled = G
        
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Assemble the data vector
    def assembled(self, datas, verbose=True):
        '''
        Assembles a data vector for inversion using the list datas
        Assembled vector is stored in self.dassembled

        Args:
            * datas         : list of data objects

        Returns:
            * None
        '''

        # Check
        if type(datas) is not list:
            datas = [datas]

        if verbose:
            # print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling d for block {}".format(self.name))

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Create a data vector
        d = np.zeros((Nd,))

        # Loop over the datasets
        el = 0
        for data in datas:

                # print
                if verbose:
                    print("Dealing with data {} of type {}".format(data.name, data.dtype))

                # Get the local d
                dlocal = self.d[data.name]
                Ndlocal = dlocal.shape[0]

                # Store it in d
                d[el:el+Ndlocal] = dlocal

                # update el
                el += Ndlocal

        # Store d in self
        self.dassembled = d
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Assemble the data covariance matrix
    def assembleCd(self, datas, add_prediction=None, verbose=True):
        '''
        Assembles the data covariance matrices that have been built for each
        data structure.

        Args:
            * datas         : List of data instances or one data instance

        Kwargs:
            * add_prediction: Precentage of displacement to add to the Cd
                              diagonal to simulate a Cp (dirty version of
                              a prediction error covariance, see Duputel et
                              al 2013, GJI).
            * verbose       : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check if the Green's function are ready
        assert self.Gassembled is not None, \
                "You should assemble the Green's function matrix first"

        if verbose:
            # print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling Cd for block {}".format(self.name))
        
        # Check
        if type(datas) is not list:
            datas = [datas]

        # Get the total number of data
        Nd = self.Gassembled.shape[0]
        Cd = np.zeros((Nd, Nd))

        # Loop over the data sets
        st = 0
        for data in datas:
            # Fill in Cd
            
            if verbose:
                print("Dealing with data {} of type {}".format(data.name, data.dtype))
                
            se = st + self.d[data.name].shape[0]
            Cd[st:se, st:se] = data.Cd
            # Add some Cp if asked
            if add_prediction is not None:
                Cd[st:se, st:se] += np.diag((self.d[data.name]*add_prediction/100.)**2)
            st += self.d[data.name].shape[0]

        # Store Cd in self
        self.Cd = Cd

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the model covariance matrix
    def buildCmGaussian(self, sigma_rot=None, sigma_intradef=None, verbose=True):
        '''
        Builds a diagonal Cm using user-defined values with sigma_rot and/or sigma_intradef
        values on the diagonal. Model covariance is stored in self.Cm.

        Kwargs:
            * sigma_rot     : standard deviation for the rotation parameters
            * sigma_intradef: standard deviation for the internal block deformation parameters
            * verbose       : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Talk to me
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print (f"Assembling the Cm matrix for block {self.name}")
            print (f"sigma rotation = {sigma_rot}")
            print (f"sigma intradef = {sigma_intradef}")

        # Check the number of parameters
        Nps = 0
        if 'rotation' in self.G[self.datanames[0]].keys() and 'rotation' in self.blockcomponent:
            Nps += 3
        if 'intradef' in self.G[self.datanames[0]].keys() and 'intradef' in self.blockcomponent:
            Nps += 3
        
        # Creates the Cm matrix
        Cm = np.eye(Nps)
        n = 0
        if 'rotation' in self.G[self.datanames[0]].keys() and 'rotation' in self.blockcomponent and sigma_rot is not None:
            Cm[:3, :3] *= sigma_rot
            n += 3
        if 'intradef' in self.G[self.datanames[0]].keys() and 'intradef' in self.blockcomponent and sigma_intradef is not None:
            Cm[n:n+3, n:n+3] *= sigma_intradef

        # Store Cm into self
        self.Cm = Cm

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Save Cm to file
    def writeCm2File(self, dtype='d', outputDir='.'):
        '''
        Write the model a priori covariance matrix to a binary file.

        Args:
            * filename      : Name of the file.
        
        Kwargs:
            * dtype         : Data type to use. Default is double ('d'). Can be 'f' for float32.
            * outputDir     : Directory to write the file in.

        Returns:
            * None
        '''

        # Check that Cm exists
        assert hasattr(self, 'Cm'), "No Cm matrix to write"

        # Write to file
        filename = f"{self.name.replace(' ', '_')}.cm"
        self.Cm.astype(dtype).tofile(os.path.join(outputDir, filename))
        print('Writing Cm matrix to file {}'.format(filename))

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Read Cm from file
    def setCmFromFile(self, filename=None, dtype='d', inDir='.'):
        '''
        Read the model a priori covariance matrix from a binary file.

        Args:
            * filename      : Name of the file.
            
        Kwargs:
            * dtype         : Data type to use. Default is double ('d'). Can be 'f' for float32.
            * inDir         : Directory to read the file from.
        
        Returns:
            * None
        '''
        
        # Check name conventions
        if filename is None:
            if os.path.isfile(os.path.join(inDir, f'{self.name.replace(" ","_")}.cm')):
                filename = os.path.join(inDir, f'{self.name.replace(" ","_")}.cm')

        # Read the files and reshape the Cm
        Cm = np.fromfile(filename, dtype=dtype)
        n = int(np.sqrt(Cm.size))
        Cm = Cm.reshape((n, n))
        
        # Store Cm
        self.Cm = Cm
        print('Reading Cm matrix from file {}'.format(filename))

        # All done
        return
    # ----------------------------------------------------------------------
    