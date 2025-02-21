'''
A group of routines that runs a point compound dislocation model (pCDM) (composed of three mutually orthogonal point tensile dislocations in a half-space.).

Written by T. Shreve, July 2019.
Adapted from Francois Beauducel and Antoine Villi's vectorized adaption of Nikkhoo's pCDM model.
# Reference ***************************************************************
# Nikkhoo, M., Walter, T.R., Lundgren, P.R., Prats-Iraola, P., 2017. Compound dislocation models (CDMs) for volcano deformation analyses.
# Okada, Y., 1985. Surface Deformation Due to Shear and Tensile Faults in a Half-Space.
'''

# External
import numpy as np
import sys
import warnings


#--------------------------------------------------
# Check inputs
def ArraySizes(*args):
    '''
    Only requirement is that each arguments has the same size and can be converted to a numpy array
    Returns : Numpy arrays
    '''

    # Create a list of sizes
    Sizes = []
    Arrays = []

    # Check class
    for arg in args:
        if arg.__class__ in (list, tuple):
            arg = np.array(arg)
        elif arg.__class__ in (float, np.float64, int):
            arg = np.array([arg])
        Arrays.append(arg)
        Sizes.append(arg.shape)

    # Assert sizes
    assert (len(np.unique(Sizes))==1), 'The {} provided arrays are not the same size'.format(len(args))

    # All done
    return Arrays

#--------------------------------------------------
# Displacements only
#def displacement(xs, ys, zs, xc, yc, zc, omegaX, omegaY, omegaZ, DVtot, A, B, nu=0.25):
def displacement(xs, ys, zs, xc, yc, zc, omegaX, omegaY, omegaZ, DVx, DVy, DVz, nu=0.25):

    '''
    Returns the displacements at the stations located on (xs, ys, zs) for pCDM pressure source
        with center on (xc, yc, zc). All arguments can be float, list or array.

    Notes :
        - This equations are only correct if the radius of curvature of the upper surface is less than or equal to its depth.???
        - Potencies must have the same sign


    Args:
            * (xs, ys, zs)      : data point locations (xs and ys must be the same size arrays)
            * (xc, yc, zc)      : center of pressure source
            * omegaX, omegaY, omegaZ     : clockwise rotations around X, Y, and Z (in degrees). omegaX = plunge, omegaY = pi/2 - dip, omegaZ = strike
            * DVtot             : DVx + DVy + DVz of the point tensile dislocations (PTD's) that before applying the rotations are normal to the X, Y and Z axes, respectively. The potency has the unit of volume (the unit of displacements and CDM semi-axes to the power of 3).
            * A                 : Horizontal divided by total volume variation ratio, A = DVz/(DVx+DVy+DVz) --- ??? or potency ???
            * B                 : Vertical volume variation ratio, B : DVy/(DVx+DVy) --- ??? or potency ???
            * nu                : poisson's ratio
    Examples:
        A = 1, any B : horizontal sill
        A = 0, B = 0.5 : vertical pipe
 		A = 0, B = 0 or 1 : vertical dyke
 		A = 1/3, B = 0.5 : isotrop source
 		A > 0, B > 0 : dyke + sill

    Returns:
            * u       : Displacement array
    '''


    mu = 30e9

    X0 = np.zeros(np.size(np.asarray(xs).flatten()))
    Y0 = np.zeros(np.size(np.asarray(xs).flatten()))
    Z0 = np.zeros(np.size(np.asarray(xs).flatten()))
    # Nu does matter here, and it is by default 0.25

    #Center coordinate system around (xc,0)
    xxn = xs - xc; yyn = ys - yc

    # Convert degrees to radians
    omegaXr = np.deg2rad(np.asarray(omegaX).flatten())
    omegaYr = np.deg2rad(np.asarray(omegaY).flatten())
    omegaZr = np.deg2rad(np.asarray(omegaZ).flatten())

    # Recomputes DV per component from DVtot, using A and B -- potency ??
    #DVz = DVtot*A
    #DVy = (DVtot-DVz)*B
    #DVx = (DVtot-DVz)*(1-B)

    # R1, R2, and R3 are the coefficients from the 3-D matrix of rotation

    # Contribution of the first point tensile dislocation (PTD)
    R1 = np.cos(omegaYr)*np.cos(omegaZr)
    R2 = -1*np.cos(omegaYr)*np.sin(omegaZr)
    R3 = np.sin(omegaYr)
    nstrike = np.sqrt(R1**2 + R2**2)
    strike = np.arctan2(-1*R2/nstrike,R1/nstrike)*180./np.pi
    strike[np.isnan(strike)] = 0
    [ux1,uy1,uz1] = runPTD_disp(xxn,yyn,zc,strike-90.,np.arccos(R3),DVx,nu)

    # Contribution of the second point tensile dislocation (PTD)
    R1 = np.cos(omegaZr)*np.sin(omegaXr)*np.sin(omegaYr) + np.cos(omegaXr)*np.sin(omegaZr)
    R2 = -1*np.sin(omegaXr)*np.sin(omegaYr)*np.sin(omegaZr) + np.cos(omegaXr)*np.cos(omegaZr)
    R3 = -1*np.sin(omegaXr)*np.cos(omegaYr)
    nstrike = np.sqrt(R1**2 + R2**2)
    strike = np.arctan2(-1*R2/nstrike,R1/nstrike)*180./np.pi
    strike[np.isnan(strike)] = 0
    [ux2,uy2,uz2] = runPTD_disp(xxn,yyn,zc,strike-90.,np.arccos(R3),DVy,nu)

    # Contribution of the third point tensile dislocation (PTD)
    R1 = -1*np.cos(omegaXr)*np.sin(omegaYr)*np.cos(omegaZr) + np.sin(omegaXr)*np.sin(omegaZr)
    R2 = np.cos(omegaXr)*np.sin(omegaYr)*np.sin(omegaZr) + np.sin(omegaXr)*np.cos(omegaZr)
    R3 = np.cos(omegaXr)*np.cos(omegaYr)
    nstrike = np.sqrt(R1**2 + R2**2)
    strike = np.arctan2(-1*R2/nstrike,R1/nstrike)*180./np.pi
    strike[np.isnan(strike)] = 0
    [ux3,uy3,uz3] = runPTD_disp(xxn,yyn,zc,strike-90.,np.arccos(R3),DVz,nu)

    Ux1 = np.reshape(ux1, np.size(xxn))
    Ux2 = np.reshape(ux2, np.size(xxn))
    Ux3 = np.reshape(ux3, np.size(xxn))
    Uy1 = np.reshape(uy1, np.size(xxn))
    Uy2 = np.reshape(uy2, np.size(xxn))
    Uy3 = np.reshape(uy3, np.size(xxn))
    Uz1 = np.reshape(uz1, np.size(xxn))
    Uz2 = np.reshape(uz2, np.size(xxn))
    Uz3 = np.reshape(uz3, np.size(xxn))

    Ux = np.reshape(ux1 + ux2 + ux3, np.size(xxn))
    Uy = np.reshape(uy1 + uy2 + uy3, np.size(xxn))
    Uz = np.reshape(uz1 + uz2 + uz3, np.size(xxn))

    # All Done
    return Ux1, Ux2, Ux3, Uy1, Uy2, Uy3, Uz1, Uz2, Uz3, Ux, Uy, Uz

#--------------------------------------------------
# Displacements only
def runPTD_disp(xxn, yyn, d, beta, dip, DV, nu):
    '''
    Calculates surface displacements associated with a tensile point dislocation (PTD) in an elastic half-space (Okada, 1985).

    Args:
            * (xxn, yyn)        : data point locations shifted by center of source
            * d                 : depth of pressure source
            * beta              : azimuth in degrees (azimuth=0 is aligned North)
            * dip               : plunge angle in degrees (dip=90 is vertical source)
            * DV                : delta volume
            * nu                : poisson's ratio

    Returns:
            * Ux, Uy, Uz        : horizontal and vertical displacements

    '''

    betar = np.deg2rad(beta)

    x = xxn*np.cos(betar) - yyn*np.sin(betar)
    y = xxn*np.sin(betar) + yyn*np.cos(betar)
    r = np.sqrt(x**2+y**2+d**2)
    q25 = 3*(y*np.sin(dip) - d*np.cos(dip))**2/(r**5)

    e = x*(q25 - np.sin(dip)**2*(1-2*nu)*(1/(r**3) - 1/(r*(r+d)**2) + y**2*(3*r+d)/(r**3*(r+d)**3)))
    n = y*(q25 - np.sin(dip)**2*(1-2*nu)*(1/(r*(r+d)**2) - x**2*(3*r+d)/(r**3*(r+d)**3)))
    v = (d*q25 - np.sin(dip)**2*(1-2*nu)*(1/(r*(r+d)) - x**2*(2*r+d)/(r**3*(r+d)**2)))

    Ux = (e*np.cos(betar) + n*np.sin(betar))*DV/(2*np.pi)
    Uy = (-1*e*np.sin(betar) + n*np.cos(betar))*DV/(2*np.pi)
    Uz = v*DV/(2*np.pi)

    return Ux, Uy, Uz
