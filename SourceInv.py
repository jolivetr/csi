''' 
A base class for faults and datasets

Written by Z. Duputel, November 2013.
'''

# Import external stuff
import copy
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt

#class SourceInv
class SourceInv(object):

    '''
    Class implementing the geographical transform. This class will
    be parent to almost all the classes in csi. 

    You can specify either an official utm zone number or provide
    longitude and latitude for a custom zone.

    Args:
        * name      : Instance Name 
        * utmzone   : UTM zone  (optional, default=None)
        * lon0      : Longitude defining the center of the custom utm zone
        * lat0      : Latitude defining the center of the custom utm zone
        * ellps     : ellipsoid (optional, default='WGS84')

    '''
    
    # ----------------------------------------------------------------------
    # Initialize class #
    def __init__(self,name,utmzone=None,ellps='WGS84',lon0=None, lat0=None):

        # Initialization
        self.name = name
        
        # Set the utm zone
        self.utmzone = utmzone
        self.ellps   = ellps
        self.lon0 = lon0
        self.lat0 = lat0
        self.set_utmzone(utmzone=utmzone, 
                         ellps = ellps, 
                         lon0 = lon0,
                         lat0 = lat0)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Lon lat Transform
    def ll2xy(self, lon, lat):
        '''
        Do the lat/lon to UTM transform. 
        Input is in degrees. UTM coordinates are returned in km.

        Args:
            * lon       : Longitude (deg)
            * lat       : Latitude (deg)

        Returns:
            * x         : UTM coordinate x (km)
            * y         : UTM coordinate y (km)
        '''

        # Transpose 
        x, y = self.putm(lon, lat)

        # Put it in Km
        x = x/1000.
        y = y/1000.

        # All done
        return x, y
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # UTM transform
    def xy2ll(self, x, y):
        '''
        Do the UTm to lat/lon transform.
        Input is in km. Output is in degrees.

        Args:
            * x         : UTM longitude (km).
            * y         : UTM latitude (km)

        Returns: 
            * lon       : Longitude (degrees)
            * lat       : Latitude (degree)
        '''

        # Transpose and return
        return self.putm(x*1000., y*1000., inverse=True)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Set the UTM zone
    def set_utmzone(self, utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        '''
        Sets the UTM zone in the class.
        Zone can be specified via its international number or 
        one can specify the center of a custom UTM zone via lon0, lat0.

        Kwargs:
            * ellps         : Reference Ellipsoid

            :Method 1:
                * utmzone       : International UTM zone number

            :Method 2: 
                * lon0          : Longitude of the center of the custom UTM zone (deg)
                * lat0          : Latitude of the center of the custom UTM zone (deg)
        '''

        # Cases
        if utmzone is not None:
            self.putm = pp.Proj(proj='utm', zone=self.utmzone, ellps=ellps)
        else:
            assert lon0 is not None, 'Please specify a 0 longitude'
            assert lat0 is not None, 'Please specify a 0 latitude'
            string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(lat0, lon0, ellps)
            self.putm = pp.Proj(string)

        # Set utmzone
        self.utmzone = utmzone
        self.lon0 = lon0
        self.lat0 = lat0
        
        # Set Geod
        self.geod = pp.Geod(ellps=ellps)

        # All done
        return        
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Checks that longitude is between 0 and 360 (kind of useless)
    def _checkLongitude(self):
        '''
        Iterates over the longitude array and checks if longitude is between 
        0 and 360
        '''

        # Check 
        if len(self.lon[self.lon<0.])>0:
            self.lon[self.lon<0.] += 360.

        # All done
        return
    # ----------------------------------------------------------------------

    

#EOF
