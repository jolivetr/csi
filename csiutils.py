import numpy as np
import scipy.interpolate as sciint
try:
    from netCDF4 import Dataset as netcdf
except:
    from scipy.io.netcdf import netcdf_file as netcdf

#----------------------------------------------------------------
#----------------------------------------------------------------
# A Dictionary with the months

months = {'JAN': 1,
          'FEB': 2,
          'MAR': 3, 
          'APR': 4, 
          'MAY': 5,
          'JUN': 6,
          'JUL': 7,
          'AUG': 8,
          'SEP': 9, 
          'OCT': 10,
          'NOV': 11,
          'DEC': 12}

#----------------------------------------------------------------
#----------------------------------------------------------------
# A routine to write netcdf files

def write2netCDF(filename, lon, lat, z, increments=None, nSamples=None, 
        title='CSI product', name='z', scale=1.0, offset=0.0, mask=None,
        xyunits=['Lon', 'Lat'], units='None', interpolation=True, verbose=True, 
        noValues=np.nan):
    '''
    Creates a netCDF file  with the arrays in Z. 
    Z can be list of array or an array, the size of lon.
                
    .. Args:
        
        * filename -> Output file name
        * lon      -> 1D Array of lon values
        * lat      -> 1D Array of lat values
        * z        -> 2D slice to be saved
        * mask     -> if not None, must be a 2d-array of a polynome to mask 
                      what is outside of it. This option is really long, so I don't 
                      use it...
   
    .. Kwargs:
               
        * title    -> Title for the grd file
        * name     -> Name of the field in the grd file
        * scale    -> Scale value in the grd file
        * offset   -> Offset value in the grd file
                
    .. Returns:
          
        * None
    '''

    if interpolation:

        # Check
        if nSamples is not None:
            if type(nSamples) is int:
                nSamples = [nSamples, nSamples]
            dlon = (lon.max()-lon.min())/nSamples[0]
            dlat = (lat.max()-lat.min())/nSamples[1]
        if increments is not None:
            dlon, dlat = increments

        # Resample on a regular grid
        olon, olat = np.meshgrid(np.arange(lon.min(), lon.max(), dlon),
                                 np.arange(lat.min(), lat.max(), dlat))
    else:
        # Get lon lat
        olon = lon
        olat = lat
        if increments is not None:
            dlon, dlat = increments
        else:
            dlon = olon[0,1]-olon[0,0]
            dlat = olat[1,0]-olat[0,0]

    # Create a file
    fid = netcdf(filename,'w')

    # Create a dimension variable
    fid.createDimension('side',2)
    if verbose:
        print('Create dimension xysize with size {}'.format(np.prod(olon.shape)))
    fid.createDimension('xysize', np.prod(olon.shape))

    # Range variables
    fid.createVariable('x_range','d',('side',))
    fid.variables['x_range'].units = xyunits[0]

    fid.createVariable('y_range','d',('side',))
    fid.variables['y_range'].units = xyunits[1]
    
    # Spacing
    fid.createVariable('spacing','d',('side',))
    fid.createVariable('dimension','i4',('side',))

    # Informations
    if title is not None:
        fid.title = title
    fid.source = 'CSI.utils.write2netCDF'

    # Filing rnage and spacing
    if verbose:
        print('x_range from {} to {} with spacing {}'.format(olon[0,0], olon[0,-1], dlon))
    fid.variables['x_range'][0] = olon[0,0]
    fid.variables['x_range'][1] = olon[0,-1]
    fid.variables['spacing'][0] = dlon

    if verbose:
        print('y_range from {} to {} with spacing {}'.format(olat[0,0], olat[-1,0], dlat))
    fid.variables['y_range'][0] = olat[0,0]
    fid.variables['y_range'][1] = olat[-1,0]
    fid.variables['spacing'][1] = dlat
    
    if interpolation:
        # Interpolate
        interpZ = sciint.LinearNDInterpolator(np.vstack((lon, lat)).T, z, fill_value=noValues)
        oZ = interpZ(olon, olat)
    else:
        # Get values
        oZ = z

    # Masking?
    if mask is not None:
        
        # Import matplotlib.path
        import matplotlib.path as path

        # Create the path
        poly = path.Path([[lo, la] for lo, la in zip(mask[:,0], mask[:,1])], 
                closed=False)

        # Create the list of points
        xy = np.vstack((olon.flatten(), olat.flatten())).T

        # Findthose outside
        bol = poly.contains_points(xy)

        # Mask those out
        oZ = oZ.flatten()
        oZ[bol] = np.nan
        oZ = oZ.reshape(olon.shape)

    # Range
    zmin = np.nanmin(oZ)
    zmax = np.nanmax(oZ)
    fid.createVariable('{}_range'.format(name),'d',('side',))
    fid.variables['{}_range'.format(name)].units = units
    fid.variables['{}_range'.format(name)][0] = zmin
    fid.variables['{}_range'.format(name)][1] = zmax

    # Create Variable
    fid.createVariable(name,'d',('xysize',))
    fid.variables[name].long_name = name
    fid.variables[name].scale_factor = scale
    fid.variables[name].add_offset = offset
    fid.variables[name].node_offset=0

    # Fill it
    fid.variables[name][:] = np.flipud(oZ).flatten()

    # Set dimension
    fid.variables['dimension'][:] = oZ.shape[::-1]

    # Synchronize and close
    fid.sync()
    fid.close()

    # All done
    return

#----------------------------------------------------------------
#----------------------------------------------------------------
# A routine to extract a profile

def coord2prof(csiobject, xc, yc, length, azimuth, width, minNum=5):
    '''
    Routine returning the profile

    Args:
        * csiobject         : An instance of a csi class that has
                              x and y attributes
        * xc                : X pos of center
        * yc                : Y pos of center
        * length            : length of the profile.
        * azimuth           : azimuth of the profile.
        * width             : width of the profile.

    Returns:
        dis                 : Distance from the center
        norm                : distance perpendicular to profile
        ind                 : indexes of the points
        boxll               : lon lat coordinates of the profile box used
        xe1, ye1            : coordinates (UTM) of the profile endpoint
        xe2, ye2            : coordinates (UTM) of the profile endpoint
    '''

    # Azimuth into radians
    alpha = azimuth*np.pi/180.

    # Copmute the across points of the profile
    xa1 = xc - (width/2.)*np.cos(alpha)
    ya1 = yc + (width/2.)*np.sin(alpha)
    xa2 = xc + (width/2.)*np.cos(alpha)
    ya2 = yc - (width/2.)*np.sin(alpha)

    # Compute the endpoints of the profile
    xe1 = xc + (length/2.)*np.sin(alpha)
    ye1 = yc + (length/2.)*np.cos(alpha)
    xe2 = xc - (length/2.)*np.sin(alpha)
    ye2 = yc - (length/2.)*np.cos(alpha)

    # Convert the endpoints
    elon1, elat1 = csiobject.xy2ll(xe1, ye1)
    elon2, elat2 = csiobject.xy2ll(xe2, ye2)

    # Design a box in the UTM coordinate system.
    x1 = xe1 - (width/2.)*np.cos(alpha)
    y1 = ye1 + (width/2.)*np.sin(alpha)
    x2 = xe1 + (width/2.)*np.cos(alpha)
    y2 = ye1 - (width/2.)*np.sin(alpha)
    x3 = xe2 + (width/2.)*np.cos(alpha)
    y3 = ye2 - (width/2.)*np.sin(alpha)
    x4 = xe2 - (width/2.)*np.cos(alpha)
    y4 = ye2 + (width/2.)*np.sin(alpha)

    # Convert the box into lon/lat for further things
    lon1, lat1 = csiobject.xy2ll(x1, y1)
    lon2, lat2 = csiobject.xy2ll(x2, y2)
    lon3, lat3 = csiobject.xy2ll(x3, y3)
    lon4, lat4 = csiobject.xy2ll(x4, y4)

    # make the box
    box = []
    box.append([x1, y1])
    box.append([x2, y2])
    box.append([x3, y3])
    box.append([x4, y4])

    # make latlon box
    boxll = []
    boxll.append([lon1, lat1])
    boxll.append([lon2, lat2])
    boxll.append([lon3, lat3])
    boxll.append([lon4, lat4])

    # Get the points in this box.
    # 1. import shapely and nxutils
    import matplotlib.path as path
    import shapely.geometry as geom

    # 2. Create an array with the positions
    XY = np.vstack((csiobject.x, csiobject.y)).T

    # 3. Create a box
    rect = path.Path(box, closed=False)

    # 4. Find those who are inside
    Bol = rect.contains_points(XY)

    # 4. Get these values
    xg = csiobject.x[Bol]
    yg = csiobject.y[Bol]
    lon = csiobject.lon[Bol]
    lat = csiobject.lat[Bol]

    # Check if lengths are ok
    assert len(xg)>minNum, \
            'Not enough points to make a worthy profile: {}'.format(len(xg))

    # 5. Get the sign of the scalar product between the line and the point
    vec = np.array([xe1-xc, ye1-yc])
    xy = np.vstack((xg-xc, yg-yc)).T
    sign = np.sign(np.dot(xy, vec))

    # 6. Compute the distance (along, across profile) and get the velocity
    # Create the list that will hold these values
    Dacros = []; Dalong = []
    # Build lines of the profile
    Lalong = geom.LineString([[xe1, ye1], [xe2, ye2]])
    Lacros = geom.LineString([[xa1, ya1], [xa2, ya2]])
    # Build a multipoint
    PP = geom.MultiPoint(np.vstack((xg,yg)).T.tolist())
    # Loop on the points
    for p in range(len(PP.geoms)):
        Dalong.append(Lacros.distance(PP.geoms[p])*sign[p])
        Dacros.append(Lalong.distance(PP.geoms[p]))

    Dalong = np.array(Dalong)
    Dacros = np.array(Dacros)

    # All done
    return Dalong, Dacros, Bol, boxll, box, xe1, ye1, xe2, ye2, lon, lat

#----------------------------------------------------------------
#----------------------------------------------------------------
# get intersection between profile and a fault trace

def intersectProfileFault(xe1, ye1, xe2, ye2, xc, yc, fault):
    '''
    Gets the distance between the fault/profile intersection and the profile center.
    Args:
        * xe1, ye1  : X and Y coordinates of one endpoint of the profile
        * xe2, ye2  : X and Y coordinates of the other endpoint of the profile
        * xc, yc    : X and Y coordinates of the centre of the profile
        * fault     : CSI fault object that has a trace.
    '''

    # Import shapely
    import shapely.geometry as geom

    # Grab the fault trace
    xf = fault.xf
    yf = fault.yf

    # Build a linestring with the profile center
    Lp = geom.LineString([[xe1, ye1],[xe2, ye2]])

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

    # Get the sign
    vec1 = [xe1-xc, ye1-yc]
    vec2 = [p[0]-xc, p[1]-yc]
    sign = np.sign(np.dot(vec1, vec2))

    # Compute the distance to the center
    d = np.sqrt( (xc-p[0])**2 + (yc-p[1])**2)*sign

    # All done
    return d

# List splitter
def _split_seq(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq

# Check if points are colocated
def colocated(point1,point2,eps=0.):
    '''
    Check if point 1 and 2 are colocated
    Args:
        * point1: x,y,z coordinates of point 1
        * point2: x,y,z coordinates of point 2
        * eps   : tolerance value 
    '''
    if np.linalg.norm(point1-point2)<=eps:
        return True
    return False
