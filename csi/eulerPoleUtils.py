
import numpy as np

# Earth radius in m
ERADIUS = 6378137.0
# Conversion milli arc second to radian) 
MAS2RAD = np.pi / 3600000 / 180
# Conversion milli arc second per year to degree per million year
MASYR2DEGMYR = 1e6 / 3600000

def gps2euler(lon, lat, elev, ve, vn, se=None, sn=None):
    """
    Solve for the Euler pole given station positions and velocitites.

    Arguments:
    lat                     array of station latitudes (degrees)
    lon                     array of station longitudes (degrees)
    elev                    array of station elevations (meters)
    ve                      array of station east velocities/displacements (meters)
    vn                      array of station north velocities/displacements (meters)
    se                      east velocitiy uncertainties (meters)
    sn                      north velocity uncertainties (meters)

    Output:
    elat                    latitude of euler pole
    elon                    longitude of euler pole
    omega                   pole rotation angle
    """
    # Convert lat-lon to ECEF xyz
    x, y, z = llh2xyz(lon, lat, elev)
    Pxyz = np.row_stack((x, y, z)).transpose()
    ndat = lat.size

    # Weighting matrix
    if se is not None:
        sig_mag = np.sqrt(se**2 + sn**2)
        Wmat = np.tile(1.0 / sig_mag, (3,1)).transpose()
    else:
        Wmat = np.ones((ndat,3), dtype=float)

    # Normalize input points
    Pxyz_unit, Pxyz_norm = unit(Pxyz)

    # Convert velocities to ECEF reference frame
    Vune = np.vstack((np.zeros(ve.shape), -vn, ve))
    Vxyz = topo2geo(Vune, Pxyz.T)

    # Rotated point using small-angle approximation
    Qxyz = Pxyz + Vxyz

    # Normalize rotated point
    Qxyz_unit, Qxyz_norm = unit(Qxyz)

    # Compute the rotation matrix
    X = np.dot((Wmat * Qxyz_unit).T, Wmat * Pxyz_unit)
    u, s, v = np.linalg.svd(X)
    darr = np.array([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(v))])
    R = np.dot(u, np.dot(np.diag(darr), v))

    # Rotation matrix -> euler vector
    elat, elon, omega = rotmat2euler(R)
    
    return np.rad2deg(elat), np.rad2deg(elon), omega


def euler2gps(evec, Pxyz):
    """
    Compute linear velocity due to angular rotation about an Euler pole.
    """
    ex, ey, ez = evec
    assert Pxyz.shape[1] == 3, 'Pxyz must be Nx3 dimensions'
    px = Pxyz[:,0]
    py = Pxyz[:,1]
    pz = Pxyz[:,2]
   
    # V = (w E) x (r P), where E is the Euler pole unit vector
    Vxyz = np.zeros(Pxyz.shape)
    Vxyz[:,0] = ey * pz - py * ez
    Vxyz[:,1] = ez * px - pz * ex
    Vxyz[:,2] = ex * py - px * ey
    Venu = geo2topo(Vxyz.T, Pxyz.T)

    return Venu


def rotmat2euler(R):
    """
    Computes the Euler pole vector from a rotation matrix R.
    """
    R11,R12,R13,R21,R22,R23,R31,R32,R33 = R.ravel()

    # Factor used in computing latitue and rotation angle
    fact = np.sqrt((R32 - R23)**2 + (R13 - R31)**2 + (R21 - R12)**2)

    # Get lat and lon of pole
    elat = np.arcsin((R21 - R12) / fact)
    atop = R13 - R31
    abot = R32 - R23
    if atop >= 0.0:
        elon = np.arctan(atop / abot)
    else:
        elon = -np.pi + np.arctan(atop / abot)
    
    # Ensure rotation angle is [0,180]
    ang = fact / (R11 + R22 + R33 - 1.0)
    omega = np.arctan(ang)
    if omega < 0.0:
        omega += np.pi

    return elat, elon, omega



def llh2xyz(lon, lat, height, earth_radius=ERADIUS):
    """
    Convert spherical coordinates (lat/lon/height) to ECEF cartesian coordinates (x/y/z)
    assuming a spherical Earth.
    
    Args:
        * lon           : float/np.ndarray, longitude (degrees)
        * lat           : float/np.ndarray, latitude (degrees)
        * height        : float/np.ndarray, height above the ellipsoid (meters)
    
    Kwargs:
        * earth_radius  : float, Earth radius (meters), default is 6378137.0 meters

    Returns:
        * x             : float/np.ndarray, cartesian x-coordinate (meters)
        * y             : float/np.ndarray, cartesian y-coordinate (meters)
        * z             : float/np.ndarray, cartesian z-coordinate (meters)
    """
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    
    # x-coordinate
    x = (earth_radius + height) * np.cos(lat) * np.cos(lon)
    
    # y-coordinate
    y = (earth_radius + height) * np.cos(lat) * np.sin(lon)
    
    # z-coordinate
    z = (earth_radius + height) * np.sin(lat)

    return x, y, z


def xyz2llh(x, y, z, earth_radius=ERADIUS):
    """
    Convert ECEF cartesian coordinates (x/y/z) to spherical coordinates (lat/lon/height)
    assuming a spherical Earth.
    
    Args:
        * x            : float/np.ndarray, x-coordinate (meters)
        * y            : float/np.ndarray, y-coordinate (meters)
        * z            : float/np.ndarray, z-coordinate (meters)
    
    Kwargs:
        * earth_radius  : float, Earth radius (meters), default is 6378137.0 meters
    
    Returns:
        * lon          : float/np.ndarray, longitude (degrees)
        * lat          : float/np.ndarray, latitude (degrees)
        * height       : float/np.ndarray, height above the ellipsoid (meters)
    """
    # Longitude
    lon = np.arctan2(y, x)
    lon = np.rad2deg(lon)

    # Latitude
    lat = np.arctan(z / np.sqrt(x**2 + y**2))
    lat = np.rad2deg(lat)

    # Height
    height = np.sqrt(x**2 + y**2 + z**2) - earth_radius

    return lon, lat, height


def geo2topo(dR, Rr, indat=()):
    """
    Transforms an ECEF relative position vector 'dR' to a topocentric vector ENU 'rt'
     - dR.shape MUST be (3,N) for matvecseq to work
    """
    # First compute the geodetic latitude/longitude of station
    if len(indat) == 3:
        lon, lat, h = indat
    else:
        lon, lat, h = xyz2llh(Rr[0, :], Rr[1, :], Rr[2, :])
    if isinstance(lat, np.ndarray):
        n = lat.size
        T = np.zeros((3,3,n), dtype=float)
    else:
        T = np.zeros((3,3,1), dtype=float)
    
    # Convert to radians
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    # Compute rotation matrix from geo -> topo
    T[0,0,...] = -np.sin(lon);                T[0,1,...] =  np.cos(lon);                T[0,2,...] = 0.0
    T[1,0,...] = -np.sin(lat)*np.cos(lon);    T[1,1,...] = -np.sin(lat)*np.sin(lon);    T[1,2,...] = np.cos(lat)
    T[2,0,...] =  np.cos(lat)*np.cos(lon);    T[2,1,...] =  np.cos(lat)*np.sin(lon);    T[2,2,...] = np.sin(lat)

    # Apply rotation
    return matvecseq(T, dR.T)


def topo2geo(rt, Rr, indat=()):
    """
    Transforms a topocentric ENU vector 'rt' to ECEF coordinate vector 'dR'
     - rt.shape MUST be (3,N) for matvecseq to work
    """
    # First compute the geodetic latitude/longitude of station
    if len(indat) == 3:
        lon, lat, h = indat
    else:
        lon, lat, h = xyz2llh(Rr[0, :], Rr[1, :], Rr[2, :])
    if isinstance(lat, np.ndarray):
        n = lat.size
        T = np.zeros((3,3,n), dtype=float)
    else:
        T = np.zeros((3,3,1), dtype=float)
    
    # Convert to radians
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    # Compute rotation matrix from topo -> geo
    T[0,0,...] = np.cos(lat)*np.cos(lon); T[0,1,...] = np.sin(lat)*np.cos(lon); T[0,2,...] = -np.sin(lon)
    T[1,0,...] = np.cos(lat)*np.sin(lon); T[1,1,...] = np.sin(lat)*np.sin(lon); T[1,2,...] =  np.cos(lon)
    T[2,0,...] = np.sin(lat);             T[2,1,...] = -np.cos(lat);            T[2,2,...] = 0.0

    # Apply rotation
    return matvecseq(T, rt.T)


def unit(A):
    """
    Reads in an input Nx3 matrix of XYZ points and outputs the unit vectors and norms.
    """
    m, n = A.shape
    assert n == 3, 'Input must be Nx3 dimension'
    Anorm = np.sqrt(A[:,0]**2 + A[:,1]**2 + A[:,2]**2)
    Aunit = A / np.tile(Anorm, (n,1)).T
    return Aunit, Anorm


def matvecseq(mat, vec):
    """
    Multiplies matrix stacked along the depth dimension by a row vector stacked along the
    vertical dimension

    mat.shape must be (3,3,N)
    vec.shape must be (N,3)
    out.shape will be (N,3)
    """

    # Reshape vector to be (3,1,N)
    N = mat[0,0,...].size
    b = np.reshape(np.transpose(vec), (3,1,N))

    # Perform multiplication
    out = (mat[:,:,None]*b).sum(axis=1)

    # Reshape to original (N,3)
    out = np.reshape(np.transpose(out), (N,3))

    return out


def matmatseq(mat1, mat2):
    """
    Multiplies matrix stacked along the depth dimension by another matrix stacked
    along the depth dimension

    mat1.shape must be (3,3,N)
    mat2.shape must be (3,3,N)
    out.shape will be (3,3,N)
    """

    out = (mat1[:,:,None]*mat2).sum(axis=1)
    return out   

#EOF
