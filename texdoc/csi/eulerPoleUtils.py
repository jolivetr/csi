
import numpy as np

# Earth radius in m
ERADIUS = 6378137.0

def gps2euler(lat, lon, elev, ve, vn, se=None, sn=None):
    """
    Solve for the Euler pole given station positions and velocitites.

    Arguments:
    lat                     array of station latitudes (radians)
    lon                     array of station longitudes (radians)
    elev                    array of station elevations
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
    Pxyz = llh2xyz(lat, lon, elev).transpose()
    ndat = lat.size

    # Weighting matrix
    if se is not None:
        sig_mag = np.sqrt(se**2 + sn**2)
        Wmat = np.tile(1.0 / sig_mag, (3,1)).transpose()
    else:
        Wmat = np.ones((ndat,3), dtype=float)

    # Normalize input points
    Pxyz_unit,Pxyz_norm = unit(Pxyz)

    # Convert velocities to ECEF reference frame
    Vune = np.vstack((np.zeros(ve.shape), -vn, ve))
    Vxyz = topo2geo(Vune, Pxyz.T)

    # Rotated point using small-angle approximation
    Qxyz = Pxyz + Vxyz

    # Normalize rotated point
    Qxyz_unit,Qxyz_norm = unit(Qxyz)

    # Compute the rotation matrix
    X = np.dot((Wmat * Qxyz_unit).T, Wmat * Pxyz_unit)
    u,s,v = np.linalg.svd(X)
    darr = np.array([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(v))])
    R = np.dot(u, np.dot(np.diag(darr), v))

    # Rotation matrix -> euler vector
    elat,elon,omega = rotmat2euler(R)
    
    return elat, elon, omega


def euler2gps(evec, Pxyz):
    """
    Compute linear velocity due to angular rotation about an Euler pole.
    """
    ex,ey,ez = evec
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

    return elat,elon,omega


def llh2xyz(lat, lon, h):
    """
    Convert lat-lon-h to XYZ assuming a spherical Earth with radius equal to
    the local ellipsoid radius.
    """
    if isinstance(lat, np.ndarray):
        n = len(lat)
        x = np.zeros((3,n), dtype=float)
    else:
        x = np.zeros((3,), dtype=float)
    
    # Compute position vector
    x[0,...] = (ERADIUS + h) * np.cos(lat) * np.cos(lon) 
    x[1,...] = (ERADIUS + h) * np.cos(lat) * np.sin(lon)
    x[2,...] = (ERADIUS + h) * np.sin(lat)

    return x.squeeze()


def xyz2llh(X):
    """
    Convert XYZ coordinates to latitude, longitude, height, assuming a spherical Earth.
    """
    x = X[0,...]
    y = X[1,...]
    z = X[2,...]

    # Longitude
    lon = np.arctan2(y, x)

    # Latitude
    p = np.sqrt(x*x + y*y)
    lat = np.arctan(z / p)

    # Height
    h = np.sqrt(x*x + y*y + z*z) - ERADIUS

    return lat, lon, h


def geo2topo(dR, Rr, indat=()):
    """
    Transforms an ECEF relative position vector 'dR' to a topocentric vector ENU 'rt'
     - dR.shape MUST be (3,N) for matvecseq to work
    """
    # First compute the geodetic latitude/longitude of station
    if len(indat) == 3:
        lat, lon, h = indat
    else:
        lat, lon, h = xyz2llh(Rr)
    if isinstance(lat, np.ndarray):
        n = lat.size
        T = np.zeros((3,3,n), dtype=float)
    else:
        T = np.zeros((3,3,1), dtype=float)

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
        lat, lon, h = indat
    else:
        lat, lon, h = xyz2llh(Rr)
    if isinstance(lat, np.ndarray):
        n = lat.size
        T = np.zeros((3,3,n), dtype=float)
    else:
        T = np.zeros((3,3,1), dtype=float)

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
    m,n = A.shape
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
