# Something that could go in Cartopy
"""
GEBCO Bathy + Topo
See https://www.gebco.net/
"""

import io
import os
import warnings

import numpy as np

import cartopy.crs as ccrs
from cartopy.io import RasterSource, LocatedImage

try:
    import netCDF4
except:
    print('Cannot import netCDF4, Gebco plotting impossible')

class GebcoSource(RasterSource):
    """
    A source of Gebco data, which implements Cartopy's :ref:`RasterSource
    interface <raster-source-interface>`.

    # Gebco file must be stored in ~/.local/share/cartopy/GEBCO

    """

    def __init__(self, dtype='sub_ice_topo', year=2022):
        """
        Parameters
        ----------
        dtype: 
            subice, icesurf 
        """

        #: The CRS of the underlying GEBCO data.
        self.crs = ccrs.PlateCarree()

        # Get file name
        fname = 'GEBCO_{}_{}.nc'.format(year, dtype)
        self.fname = os.path.join(os.path.expanduser('~/.local/share/cartopy/GEBCO'), fname)

        # All done
        return

    def validate_projection(self, projection):
        return projection == self.crs

    def fetch_raster(self, projection, extent, target_resolution):
        """
        Fetch SRTM elevation for the given projection and approximate extent.

        """
        # Check
        if not self.validate_projection(projection):
            raise ValueError(f'Unsupported projection for the '
                             f'Gebco source.')

	# Get img
        lon_min, lon_max, lat_min, lat_max = extent
        cropped, extent = self.crop(lon_min, lon_max, lat_min, lat_max)
	
        # All done
        return [LocatedImage(np.flipud(cropped), extent)]

    def crop(self, lon_min, lon_max, lat_min, lat_max):
        """
        Return an image and its extent
        """

        # Load data
        dataset = netCDF4.Dataset(self.fname)

        # Extract variables
        lon = dataset.variables['lon']
        lat = dataset.variables['lat']
        elevation = dataset.variables['elevation']
        
        # Crop
        u = np.flatnonzero(np.logical_and(lon[:]>=lon_min, lon[:]<=lon_max)) 
        v = np.flatnonzero(np.logical_and(lat[:]>=lat_min, lat[:]<=lat_max))      
        
        # Get
        cropped = elevation[v[0]:v[-1]+1, u[0]:u[-1]+1]
        
        # True extent
        deltalon = lon[1] - lon[0]
        deltalat = lat[1] - lat[0]
        extent = [lon[u[0]] - deltalon/2., 
                  lon[u[-1]] + deltalon/2., 
                  lat[v[0]] - deltalat/2.,
                  lat[v[-1]] + deltalat/2.]

        return cropped, extent

#EOF
