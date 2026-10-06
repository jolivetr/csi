'''
A group of routines that runs yang.

Written by T. Shreve, May 2019.
Adapted from USGS's dMODELS MATLAB scripts.
% Reference ***************************************************************
%
% Note ********************************************************************
% compute the displacement due to a pressurized ellipsoid
% using the finite prolate spheroid model by from Yang et al (JGR,1988)
% and corrections to the model by Newmann et al (JVGR, 2006).
% The equations by Yang et al (1988) and Newmann et al (2006) are valid for a
% vertical prolate spheroid only. There is and additional typo at pg 4251 in
% Yang et al (1988), not reported in Newmann et al. (2006), that gives an error
% when the spheroid is tilted (plunge different from 90?):
%           C0 = y0*cos(theta) + z0*sin(theta)
% The correct equation is
%           C0 = z0/sin(theta)
% This error has been corrected in this script.
% *************************************************************************
'''

# External
import numpy as np
import sys


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
def displacement(xs, ys, zs, xc, yc, zc, a, A, dip, strike, DP, nu=0.25):
    '''
    Returns the displacements at the stations located on (xs, ys, zs) for an prolate spheroid pressure source
        with center on (xc, yc, zc). All arguments can be float, list or array.

    Note :
        This equations are only correct if the radius of curvature of the upper surface is less than or equal to its depth.


    Args:
            * (xs, ys, zs)      : data point locations
            * (xc, yc, zc)      : center of pressure source
            * a                 : semi-major axis
            * A                 : geometric aspect ratio (b/a)
            * dip               : plunge angle (dip=90 is vertical source)
            * strike            : azimuth (azimuth=0 is aligned North)
            * DP                : dimensionless pressure
            * nu                : poisson's ratio

    Returns:
            * Ux, Uy, Uz        : horizontal and vertical displacements
    '''


    mu = 30e9

    # Nu does matter here, and it is by default 0.25

    #deal with singularities
    if dip >= 89.99:
        dip = 89.99
    if dip == 0.0:
        dip = 0.0001
    if A >= 0.99:
        A = 0.99

    # Convert degrees to radians for dip and strike
    dip = dip*np.pi/180.
    strike = strike*np.pi/180.

    #Define parameters correctly
    b = A*a                         #semi-minor axis
    lambd = 2.*mu*nu/(1.-2.*nu)        #first Lame's elastic modulus
    P = DP*mu                       #Excess pressure

    ### Double check this
    if float(zc) < float(A*a)**2/float(a):
        raise Exception('radius of curvature has to be less than the depth...')

    # Run yang
    Ux, Uy, Uz = runYang_disp(xs, ys, zs, xc, yc, zc, a, b, dip, strike, P, mu, nu, lambd)

    # All Done
    return Ux, Uy, Uz

#--------------------------------------------------
# Displacements only
def runYang_disp(xs, ys, zs, xc, yc, zc, a, b, dip, strike, P, mu, nu, lambd):
    '''
    Yang formulation adapted from dMODELS for 3D displacements at the surface (yangdisp.m).
    Maurizio Battaglia, et al, dMODELS: A MATLAB software package for modeling crustal deformation near active faults and volcanic centers, JVGR, Volume 254, 2013.

    Args:
            * (xs, ys, zs)      : data point locations
            * (xc, yc, zc)      : center of pressure source
            * a                 : semi-major axis
            * b                 : semi-minor axis
            * dip               : plunge angle (dip=90 is vertical source)
            * strike            : azimuth (azimuth=0 is aligned North)
            * P                 : excess pressure
            * mu                : shear modulus
            * nu                : poisson's ratio
            * lambd             : lame's constant

    Returns:
            * Ux, Uy, Uz        : horizontal and vertical displacements

    '''

    a1, b1, c, Pdila, Pstar = runYang_param(a, b, P, mu, nu, lambd)

    #Center coordinate system around (xc,0) and rotate (see Fig. 3 in Yang et al, 1988)
    xxn = xs - xc; yyn = ys - yc
    xxp = np.cos(strike)*xxn - np.sin(strike)*yyn
    yyp = np.sin(strike)*xxn + np.cos(strike)*yyn

    #Compute displacements for c and -c (focus of prolate spheroid)
    [U1p,U2p,U3p] = runYang_int(xxp,yyp,zs,zc,dip,a1,b1,a,b,c,mu,nu,Pdila)
    [U1m,U2m,U3m] = runYang_int(xxp,yyp,zs,zc,dip,a1,b1,a,b,-1*c,mu,nu,Pdila)
    Upx = -1.*U1p - U1m
    Upy = -1.*U2p - U2m
    Upz = U3p + U3m

    #Rotate back to original coordinate system
    Ux = np.cos(strike)*Upx + np.sin(strike)*Upy
    Uy = -1.*np.sin(strike)*Upx + np.cos(strike)*Upy
    Uz = Upz

    return Ux, Uy, Uz

#--------------------------------------------------
# Strain only -- Needs to be implemented
def runYang_strain(xs, ys, zs, xc, yc, zc, A, dip, strike, DP, mu, nu):
    '''
    Yang formulation adapted from dMODELS.
    Maurizio Battaglia, et al, dMODELS: A MATLAB software package for modeling crustal deformation near active faults and volcanic centers, JVGR, Volume 254, 2013.
    '''
    print("To be implemented")
    return

#--------------------------------------------------
# Compute parameters for the prolate spheroid model
def runYang_param(a, b, P, mu, nu, lambd):
    '''
    Computes correct parameters for displacement calculation
    Yang formulation adapted from dMODELS (yangpar.m).
    Maurizio Battaglia, et al, dMODELS: A MATLAB software package for modeling crustal deformation near active faults and volcanic centers, JVGR, Volume 254, 2013.
    Args:
            * a                 : semi-major axis
            * b                 : semi-minor axis
            * P                : dimensionless pressure
            * mu                : shear modulus
            * nu                : poisson's ratio
            * lambd             : lame's constant
    Returns:
            * a1, b1, c, Pdila, Pstar   : Parameters used to calculate surface displacements
    '''
    c = np.sqrt((a)**2-(b)**2)

    a2 = (a)**2; a3 = (a)**3
    b2 = (b)**2
    c2 = (c)**2; c3 = (c)**3; c4 = (c)**4; c5 = (c)**5
    ac = (a-c)/(a+c)
    coef1 = 2.*np.pi*a*b2; den1 = 8.*np.pi*(1-nu)

    Q = 3./den1; R = (1.-2.*nu)/den1
    Ia = -coef1*((2./(a*c2)) + (np.log(ac)/c3)); Iaa = -coef1*((2./(3.*a3*c2))+(2./(a*c4))+(np.log(ac)/c5))

    a11 = 2.*R*(Ia-4.*np.pi)
    a12 = -2.*R*(Ia+4.*np.pi)
    a21 = Q*a2*Iaa + R*Ia - 1.
    a22 = -Q*a2*Iaa - Ia*(2.*R-Q)

    den2 = 3.*lambd + 2.*mu; den3 = a11*a22 - a12*a21
    num2 = 3.*a22 - a12; num3 = a11 - 3.*a21

    Pdila = P*(2.*mu/den2)*(num2-num3)/den3
    Pstar = (P/den2)*((num2*lambd) + (2.*(lambd+mu)*num3))/den3

    a1 = -2.*b2*Pdila
    b1 = 3.*(b2/c2)*Pdila + 2.*(1.-2.*nu)*Pstar



    return a1, b1, c, Pdila, Pstar


#--------------------------------------------------
# Compute displacements for the spheroid model
def runYang_int(xs,ys,zs,z0,dip,a1,b1,a,b,csi,mu,nu,Pdila):
    '''
    Computes displacement
    Yang formulation adapted from dMODELS (yangint.m).
    Maurizio Battaglia, et al, dMODELS: A MATLAB software package for modeling crustal deformation near active faults and volcanic centers, JVGR, Volume 254, 2013.

    Args:
            * (xs, ys, zs) : data point locations
            * z0           : depth of center of source
            * dip          : plunge angle (dip=90 is vertical source)
            * (a1, b1, Pdila)     : parameters calculated in runYang_param
            * a            : semi-major axis
            * b            : semi-minor axis
            * csi          : distance to focal point of spheroid
            * mu           : shear modulus
            * nu           : poisson's ratio

    Returns:
            * (U1, U2, U3) : displacements at data points

    '''

    sint = np.sin(dip); cost = np.cos(dip)

    #parameters and coordinates
    csi2 = csi*cost; csi3 = csi*sint
    x1 = xs; x2 = ys; x3 = zs - z0; xbar3 = zs + z0
    y1 = x1; y2 = x2 - csi2; y3 = x3 - csi3; ybar3 = xbar3 + csi3
    r2 = x2*sint - x3*cost; q2 = x2*sint + xbar3*cost
    r3 = x2*cost + x3*sint; q3 = -x2*cost + xbar3*sint
    rbar3 = r3 - csi; qbar3 = q3 + csi;
    R1 = np.sqrt((y1)**2+(y2)**2+(y3)**2); R2 = np.sqrt((y1)**2+(y2)**2+(ybar3)**2)

    #Correct C0
    C0 = z0/sint

    #10^-15 increment added to avoid issues at origin
    beta = (q2*cost + (1. + sint)*(R2 + qbar3))/(cost*y1 + 0.0000000000000001)

    drbar3 = R1 + rbar3; dqbar3 = R2 + qbar3; dybar3 = R2 + ybar3
    lrbar3 = np.log(R1 + rbar3); lqbar3 = np.log(R2 + qbar3); lybar3 = np.log(R2 + ybar3)
    atanb = np.arctan(beta)

    Astar1 = a1/(R1*drbar3) + b1*(lrbar3 + ((r3 + csi)/drbar3))
    Astarbar1 = -a1/(R2*dqbar3) - b1*(lqbar3 + ((q3-csi)/dqbar3))

    A1 = csi/R1 + lrbar3; Abar1 = csi/R2 - lqbar3
    A2 = R1 - r3*lrbar3; Abar2 = R2 - q3*lqbar3
    A3 = csi*rbar3/R1 + R1; Abar3 = csi*qbar3/R2 - R2

    Bstar = ((a1/R1)+(2.*b1*A2)) + (3.-4.*nu)*((a1/R2)+(2.*b1*Abar2))
    B = csi*(csi + C0)/R2 - Abar2 - C0*lqbar3

    Fstar1 = 0.
    Fstar2 = 0.
    F1 = 0.
    F2 = 0.

    f1 = csi*y1/dybar3 + (3./(cost)**2)*(y1*sint*lybar3 - y1*lqbar3 + 2.*q2*atanb) + 2.*y1*lqbar3 - 4.*xbar3*atanb/cost
    f2 = csi*y2/dybar3 + (3./(cost)**2)*(q2*sint*lqbar3 - q2*lybar3 + 2.*y1*sint*atanb + cost*(R2 - ybar3)) - 2.*cost*Abar2 + (2./cost)*(xbar3*lybar3 - q3*lqbar3)
    f3 = (1./cost)*(q2*lqbar3 - q2*sint*lybar3 + 2.*y1*atanb) + 2.*sint*Abar2 + q3*lybar3 - csi

    cstar = (a*(b)**2/(csi)**3)/(16.*mu*(1.-nu)); cdila = 2.*cstar*Pdila


    #Primitive (indefinite integral) of equation 1 from Yang, et al 1988, but z-contribution removed to improve internal deformation fit with FEM
    Ustar1 = cstar*(Astar1*y1 + (3.-4.*nu)*Astarbar1*y1 + Fstar1*y1)
    Ustar2 = cstar*(sint*(Astar1*r2 + (3.-4.*nu)*Astarbar1*q2 + Fstar1*q2) + cost*(Bstar - Fstar2))
    Ustar3 = cstar*(-cost*(Astar1*r2 + (3.-4.*nu)*Astarbar1*q2 - Fstar1*q2) + sint*(Bstar + Fstar2))

    Udila1 = cdila*((A1*y1 + (3.-4.*nu)*Abar1*y1 + F1*y1) - 4.*(1.-nu)*(1.-2.*nu)*f1)
    Udila2 = cdila*(sint*(A1*r2 + (3.-4.*nu)*Abar1*q2 + F1*q2) - 4.*(1.-nu)*(1.-2.*nu)*f2 + 4.*(1.-nu)*cost*(A2+Abar2) + cost*(A3-(3.-4.*nu)*Abar3 - F2))
    Udila3 = cdila*(cost*(-A1*r2 + (3.-4.*nu)*Abar1*q2 + F1*q2) + 4.*(1.-nu)*(1.-2.*nu)*f3 + 4.*(1.-nu)*sint*(A2+Abar2) + sint*(A3+(3.-4.*nu)*Abar3 + F2 - 2.*(3.-4.*nu)*B))

    U1 = Ustar1 + Udila1
    U2 = Ustar2 + Udila2
    U3 = Ustar3 + Udila3


    return U1, U2, U3

# #--------------------------------------------------
# # Stress only -- needs to be implemented
def runYang_stress(xs, ys, zs, xc, yc, zc, width, length, strike, dip, ss, ds, ts, mu=30e9, nu=0.25, full=False):

    print("To be implemented")
    return


def plotYang(xc, yc, zc, a, A, dip, strike, DP, nu=0.25, deform='z'):
    """
    A routine to plot and check the outputs of the yang equations.

    Args:
        * (xc, yc, zc)      : center of pressure source
        * a                 : semi-major axis
        * A                 : geometric aspect ratio (b/a)
        * dip               : plunge angle (dip=90 is vertical source)
        * strike            : azimuth (azimuth=0 is aligned North)
        * DP                : dimensionless pressure
        * nu                : poisson's ratio


    Returns:
        None
    """

    from matplotlib import pyplot as plt
    import yangfull
    
    # create 30000 m x 30000 m grid
    x=np.linspace(-15000,15000,301)
    y=x
    X,Y = np.meshgrid(x,y)
    nx=301
    ny=301

    # displacements from yang's equations
    Ux,Uy,Uz = yangfull.displacement(X.flatten(),Y.flatten(),Y.flatten()*0,xc, yc, zc, a, A, dip, strike, DP, nu)

    # which displacement do you want to plot
    if deform == "z":
        Ufin = Uz.reshape([nx,ny])
    elif deform == "y":
        Ufin = Uy.reshape([nx,ny])
    elif deform == "x":
        Ufin = Ux.reshape([nx,ny])

    # plot
    plt.scatter(X,Y,c=Ufin)
    plt.colorbar()
    plt.show()



    return
#
# #EOF
