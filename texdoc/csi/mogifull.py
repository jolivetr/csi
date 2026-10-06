'''
A group of routines that runs mogi.

Written by T. Shreve, June 2019.

References:
Mogi, K., 1958, Relations between the eruptions of various volcanoes and the deformations of the ground surfaces around them
Beauducel, Francois, 2011, Mogi: point source in elastic half-space, https://fr.mathworks.com/matlabcentral/fileexchange/25943-mogi-point-source-in-elastic-half-space

'''

# External
import numpy as np
import sys


#--------------------------------------------------
# Displacements only
def displacement(xs, ys, zs, xc, yc, zc, a, DP, nu=0.25):
    '''
    Returns the displacements at the stations located on (xs, ys, zs) for spheroid pressure source
        with center on (xc, yc, zc). All arguments can be float, list or array.

    Note :
        This equations are only correct if the radius of curvature of the upper surface is less than or equal to its depth.


    Args:
            * (xs, ys, zs)      : data point locations
            * (xc, yc, zc)      : center of pressure source
            * a                 : semi-major axis
            * DP                : dimensionless pressure
            * nu                : poisson's ratio

    Returns:
            * u       : Displacement array
    '''

    mu = 30e9

    #Define parameters correctly
    lambd = 2.*mu*nu/(1.-2.*nu)        #first Lame's elastic modulus
    P = DP*mu                       #Excess pressure

    # Run mogi
    Ux, Uy, Uz = runMogi_disp(xs, ys, zs, xc, yc, zc, a, P, mu, nu, lambd)

    # All Done
    return Ux, Uy, Uz

#--------------------------------------------------
# Displacements only
def runMogi_disp(xs, ys, zs, xc, yc, zc, a, P, mu, nu, lambd):
    '''
    Mogi formulation for 3D displacements at the surface (yangdisp.m).

    Args:
            * (xs, ys, zs)      : data point locations
            * (xc, yc, zc)      : center of pressure source
            * a                 : semi-major axis
            * P                 : excess pressure
            * mu                : shear modulus
            * nu                : poisson's ratio
            * lambd             : lame's constant

    Returns:
            * Ux, Uy, Uz        : horizontal and vertical displacements

    '''

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

    #Shift center
    xxn = xs - xc; yyn = ys - yc

    #Convert to polar coordinates
    [rho, phi] = cart2pol(xxn,yyn)

    #Distance from source
    R = np.sqrt((zc)**2+(rho)**2)

    #Displacements
    Ur = a**3*P*(1.0-nu)*rho/(mu*R**3)
    Uz = a**3*P*(1.0-nu)*zc/(mu*R**3)

    #Convert to cartesian coordinates
    [Ux, Uy] = pol2cart(Ur,phi)

    return Ux, Uy, Uz

#--------------------------------------------------
# Strain only - Needs to be implemented
def runMogi_strain(xs, ys, zs, xc, yc, zc, A, dip, strike, DP, mu, nu):
    '''
    Formulation adapted from dMODELS.
    Maurizio Battaglia, et al, dMODELS: A MATLAB software package for modeling crustal deformation near active faults and volcanic centers, JVGR, Volume 254, 2013.
    '''
    print("To be implemented")
    return u, d, s
#
# #--------------------------------------------------
# # Stress only - Needs to be implemented
def runMogi_stress(xs, ys, zs, xc, yc, zc, width, length, strike, dip, ss, ds, ts, mu=30e9, nu=0.25, full=False):
    print("To be implemented")
    return
# #EOF
