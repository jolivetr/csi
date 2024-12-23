'''
A group of routines that runs a compound dislocation model (CDM) (composed of three mutually orthogonal tensile dislocations in a half-space.).

Written by T. Shreve, July 2019.
Adapted from Francois Beauducel and Antoine Villi's vectorized adaption of Nikkhoo's CDM model.

# Reference ***************************************************************
# Nikkhoo, M., Walter, T.R., Lundgren, P.R., Prats-Iraola, P., 2017. Compound dislocation models (CDMs) for volcano deformation analyses.
# Okada, Y., 1985. Surface Deformation Due to Shear and Tensile Faults in a Half-Space.
'''

# External
import numpy as np
import sys
import warnings
import math



#--------------------------------------------------
# Displacements only
def displacement(xs, ys, zs, xc, yc, zc, omegaX, omegaY, omegaZ, ax, ay, az, opening, nu=0.25):
    '''
    Returns the displacements at the stations located on (xs, ys, zs) for CDM pressure source
        with center on (xc, yc, zc). All arguments can be float, list or array.

    Args:
            * (xs, ys, zs)      : Data point locations (xs and ys must be the same size arrays)
            * (xc, yc, zc)      : Center of pressure source
            * omegaX, omegaY, omegaZ     : Clockwise rotations around X, Y, and Z (in degrees). omegaX = plunge, omegaY = pi/2 - dip, omegaZ = strike
            * ax, ay, az        : Semi-axes of the CDM along the x, y and z axes respectively, before applying rotations. Has same unit as X and Y.
            * opening           : Tensile component of Burgers vectors of Rectangular Dislocations (RD's) that form the CDM. Unit of opening same as unit of ax, ay, and az.
            * nu                : Poisson's ratio

    Returns:
            * u       : Displacement array
            * DV      : Potency of CDM, with same unit as volume (unit of displacements, opening, and CDM semi-axes to the power of 3)
    '''


    X0 = np.zeros(np.size(np.asarray(xs).flatten()))
    Y0 = np.zeros(np.size(np.asarray(xs).flatten()))
    Z0 = np.zeros(np.size(np.asarray(xs).flatten()))

    #Center coordinate system around (xc,0)
    xxn = xs - xc; yyn = ys - yc

    #Convert semi-axes to axes and flatten (if multiple sources)
    ax = 2. * np.asarray(ax).flatten()
    ay = 2. * np.asarray(ay).flatten()
    az = 2. * np.asarray(az).flatten()

    # Convert degrees to radians and flatten (if multiple sources)
    omegaXr = np.deg2rad(np.asarray(omegaX).flatten())
    omegaYr = np.deg2rad(np.asarray(omegaY).flatten())
    omegaZr = np.deg2rad(np.asarray(omegaZ).flatten())

    # Define opening and flatten (if multiple sources)
    op = np.asarray(opening).flatten()

    # Convert poisson ratio to 1-(2*nu)
    nu = 1. - (2.*nu)

    # Normalize size to allow any mixing of scalar/matrix
    omegaXr,omegaYr,omegaZr,ax,ay,az = [np.tile(i,(np.size(X0))) for i in [omegaXr,omegaYr,omegaZr,ax,ay,az]]

    #Coefficients from 3-D matrix of rotation
    r11 = np.cos(omegaYr)*np.cos(omegaZr)
    r12 = -1.*np.cos(omegaYr)*np.sin(omegaZr)
    r13 = np.sin(omegaYr)
    r21 = np.cos(omegaZr)*np.sin(omegaXr)*np.sin(omegaYr) + np.cos(omegaXr)*np.sin(omegaZr)
    r22 = -1.*np.sin(omegaXr)*np.sin(omegaYr)*np.sin(omegaZr) + np.cos(omegaXr)*np.cos(omegaZr)
    r23 = -1.*np.sin(omegaXr)*np.cos(omegaYr)
    r31 = -1.*np.cos(omegaXr)*np.sin(omegaYr)*np.cos(omegaZr) + np.sin(omegaXr)*np.sin(omegaZr)
    r32 = np.cos(omegaXr)*np.sin(omegaYr)*np.sin(omegaZr) + np.sin(omegaXr)*np.cos(omegaZr)
    r33 = np.cos(omegaXr)*np.cos(omegaYr)


    #Coordinates for RD summits
    p1 = np.asarray([ay*r21/2. + az*r31/2., ay*r22/2. + az*r32/2., ay*r23/2. + az*r33/2. - zc])
    p2 = p1 - np.tile(ay,(3,1))*np.asarray([r21,r22,r23])
    p3 = p2 - np.tile(az,(3,1))*np.asarray([r31,r32,r33])
    p4 = p1 - np.tile(az,(3,1))*np.asarray([r31,r32,r33])

    q1 = np.asarray([-1*ax*r11/2. + az*r31/2., -1*ax*r12/2. + az*r32/2., -1*ax*r13/2. + az*r33/2. - zc])
    q2 = q1 + np.tile(ax,(3,1))*np.asarray([r11,r12,r13])
    q3 = q2 - np.tile(az,(3,1))*np.asarray([r31,r32,r33])
    q4 = q1 - np.tile(az,(3,1))*np.asarray([r31,r32,r33])

    r1 = np.asarray([ax*r11/2. + ay*r21/2., ax*r12/2. + ay*r22/2., ax*r13/2. + ay*r23/2. - zc])
    r2 = r1 - np.tile(ax,(3,1))*np.asarray([r11,r12,r13])
    r3 = r2 - np.tile(ay,(3,1))*np.asarray([r21,r22,r23])
    r4 = r1 - np.tile(ay,(3,1))*np.asarray([r21,r22,r23])


    ux1,uy1,uz1 = RDdispSurf(xxn,yyn,p1,p2,p3,p4,op,nu)
    ux2,uy2,uz2 = RDdispSurf(xxn,yyn,q1,q2,q3,q4,op,nu)
    ux3,uy3,uz3 = RDdispSurf(xxn,yyn,r1,r2,r3,r4,op,nu)

    Ux = ux1 + ux2 + ux3
    Uy = uy1 + uy2 + uy3
    Uz = uz1 + uz2 + uz3

    # Special cases (one dimension is zero). If written properly, runtime could be sped up here?
    k = np.all([ax==0,ay!=0,az!=0],axis=0)
    Ux[k] = ux1[k]
    Uy[k] = uy1[k]
    Uz[k] = uz1[k]

    k = np.all([ax!=0,ay==0,az!=0],axis=0)
    Ux[k] = ux2[k]
    Uy[k] = uy2[k]
    Uz[k] = uz2[k]

    k = np.all([ax!=0,ay!=0,az==0],axis=0)
    Ux[k] = ux3[k]
    Uy[k] = uy3[k]
    Uz[k] = uz3[k]

    # Special cases (two or three dimensions are zero)
    k = np.all([ax==0,ay==0,az==0],axis=0) | np.all([ax==0,ay==0],axis=0) | np.all([ax==0,az==0],axis=0) | np.all([ay==0,az==0],axis=0)
    Ux[k] = 0
    Uy[k] = 0
    Uz[k] = 0

    #Half-space solution: the CDM must be under the free surface!
    kairTest = []
    for i in [p1[2,:],p2[2,:],p3[2,:],p4[2,:],q1[2,:],q2[2,:],q3[2,:],q4[2,:],r1[2,:],r2[2,:],r3[2,:],r4[2,:]]:
        kbool = i > 0.2
        kairTest.append(kbool)
    kair = np.any(kairTest, axis=0)

    Ux[kair] = float('NaN')
    Uy[kair] = float('NaN')
    Uz[kair] = float('NaN')

    Ux = np.reshape(Ux, np.size(xxn))
    Uy = np.reshape(Uy, np.size(xxn))
    Uz = np.reshape(Uz, np.size(xxn))

    # Calculate the CDM total potency (AX, AY and AZ were converted to full axes)
    Dv = np.reshape((ax*ay + ax*az + ay*az)*op, np.size(xxn))

    # All Done
    return Ux, Uy, Uz, Dv

#--------------------------------------------------
# Surface displacements associated with retangular dislocation in an elastic half-space
def RDdispSurf(xxn, yyn, p1, p2, p3, p4, op, nu):
    '''
    Calculates surface displacements associated with a rectangular dislocation in an elastic half-space (Okada, 1985).
    Args:
            * (xxn, yyn)        : data point locations
            * (p1,p2,p3,p4)     : coordinates for RD summits
            * op                : tensile component of Burgers vector (opening) of the RD's that form the CDM
            * nu                : poisson's ratio

    Returns:
            * Ux, Uy, Uz        : horizontal and vertical displacements

    '''

    Vnorm = np.transpose(np.cross(np.transpose(p2-p1),np.transpose(p4-p1)))

    Vnorm = Vnorm/np.tile(np.sqrt(np.sum(Vnorm**2.,axis=0)),(3,1))

    bX = op*Vnorm[0,:]
    bY = op*Vnorm[1,:]
    bZ = op*Vnorm[2,:]

    u1, v1, w1 = AngSetupFSC(xxn, yyn, bX, bY, bZ, p1, p2, nu)
    u2, v2, w2 = AngSetupFSC(xxn, yyn, bX, bY, bZ, p2, p3, nu)
    u3, v3, w3 = AngSetupFSC(xxn, yyn, bX, bY, bZ, p3, p4, nu)
    u4, v4, w4 = AngSetupFSC(xxn, yyn, bX, bY, bZ, p4, p1, nu)

    Ux = u1 + u2 + u3 + u4
    Uy = v1 + v2 + v3 + v4
    Uz = w1 + w2 + w3 + w4


    return Ux, Uy, Uz


#--------------------------------------------------
# Calculates displacements associated with an angular dislocation pair on each side of an RD in a half-space
def AngSetupFSC(xxn, yyn, bX, bY, bZ, pa, pb, nu):

    pb = np.transpose(pb)
    pa = np.transpose(pa)
    SideVec = pb - pa
    beta = np.arccos(-1*SideVec[:,2]/np.sqrt(np.sum(SideVec**2.,axis=1)))
    Vnorm = np.sqrt(np.sum(SideVec[:,0:2]**2.,axis=1))



    a1 = SideVec[:,0]/Vnorm
    a2 = SideVec[:,1]/Vnorm

    # Transform coordinates from EFCS (earth-fixed coordinate system) to thet first ADCS (angular dislocation coordinate system)
    y1A = a1*(xxn - pa[:,0]) + a2*(yyn - pa[:,1])
    y2A = a2*(xxn - pa[:,0]) - a1*(yyn - pa[:,1])

    # Transform coordinates from EFCS to the second ADCS
    y1B = y1A - (a1*SideVec[:,0] + a2*SideVec[:,1])
    y2B = y2A - (a2*SideVec[:,0] - a1*SideVec[:,1])


    # Transform slip vector components from EFCS to ADCS
    b1 = a1*bX + a2*bY
    b2 = a2*bX - a1*bY
    b3 = -1*bZ


    v1A, v2A, v3A = AngDisDispSurf(y1A,y2A,beta,b1,b2,b3,nu,-1*pa[:,2])
    v1B, v2B, v3B = AngDisDispSurf(y1B,y2B,beta,b1,b2,b3,nu,-1*pb[:,2])

    # Artefact-free for the calculation points near the free surface
    I = (beta*y1A)>=0

    if np.any(I):

        v1A[I],v2A[I],v3A[I] = AngDisDispSurf(y1A[I],y2A[I],beta[I]-np.pi,b1[I],b2[I],b3[I],nu,-1*pa[I,2])
        v1B[I],v2B[I],v3B[I] = AngDisDispSurf(y1B[I],y2B[I],beta[I]-np.pi,b1[I],b2[I],b3[I],nu,-1*pb[I,2])

    # Calculate total displacements in ADCS
    v1 = v1B - v1A
    v2 = v2B - v2A
    v3 = v3B - v3A

    # Transform total displacements from ADCS to EFCS
    Ux = a1*v1 + a2*v2
    Uy = a2*v1 - a1*v2
    Uz = -1*v3

    k = np.logical_or(np.abs(beta)<np.spacing(1),np.abs(np.pi-beta)<np.spacing(1))

    Ux[k] = 0
    Uy[k] = 0
    Uz[k] = 0



    return Ux, Uy, Uz

#--------------------------------------------------
# Calculates displacements associated with an angular dislocation in a half-space

def AngDisDispSurf(y1,y2,beta,b1,b2,b3,nu,a):
    sinB = np.sin(beta)
    cosB = np.cos(beta)
    cotB = 1./np.tan(beta)
    z1 = y1*cosB + a*sinB
    z3 = y1*sinB - a*cosB
    r = np.sqrt(y1**2. + y2**2. + a**2.)

    Fi = 2.*np.arctan2(y2,(r+a)*(1./(np.tan(beta/2.))) - y1)

    v1b1 = b1*((1. - nu*cotB**2.)*Fi + y2/(r+a)*(nu*(cotB + y1/(2.*(r+a))) - y1/r) - y2*(r*sinB - y1)*cosB/(r*(r-z3)))
    v2b1 = b1*(nu*((0.5+cotB**2.)*np.log(r+a)-cotB/sinB*np.log(r-z3)) - 1./(r+a)*(nu*(y1*cotB - (a/2.) - y2**2./(2.*(r+a))) + y2**2./r) + y2**2.*cosB/(r*(r-z3)))
    v3b1 = b1*(nu*Fi*cotB + y2/(r+a)*(1-nu + (a/r)) - y2*cosB/(r-z3)*(cosB+(a/r)))

    v1b2 = b2*(-1.*nu*((0.5-cotB**2.)*np.log(r+a) + cotB**2.*cosB*np.log(r-z3)) - (1./(r+a))*(nu*(y1*cotB + (0.5*a) + y1**2./(2.*(r+a))) - y1**2./r) + z1*(r*sinB-y1)/(r*(r-z3)))
    v2b2 = b2*((1.+nu*cotB**2.)*Fi - y2/(r + a)*(nu*(cotB + y1/(2.*(r+a))) - y1/r) - y2*z1/(r*(r-z3)))
    v3b2 = b2*(-1.*nu*cotB*(np.log(r+a) - cosB*np.log(r-z3)) - y1/(r+a)*(1.-nu + (a/r)) + z1/(r-z3)*(cosB + (a/r)))

    v1b3 = b3*(y2*(r*sinB - y1)*sinB/(r*(r-z3)))
    v2b3 = b3*(-1.*y2**2.*sinB/(r*(r-z3)))
    v3b3 = b3*(Fi + y2*(r*cosB + a)*sinB/(r*(r-z3)))

    v1 = (v1b1 + v1b2 + v1b3)/(2.*np.pi)
    v2 = (v2b1 + v2b2 + v2b3)/(2.*np.pi)
    v3 = (v3b1 + v3b2 + v3b3)/(2.*np.pi)

    for v in [v1,v2,v3]:
        v[np.where(abs(v)>1e10)]=float('NaN')


    return v1, v2, v3
