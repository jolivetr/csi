#-*- coding: utf-8 -*-

import copy
import numpy as np
from matplotlib.path import Path

# Translated from Brendan Meade's MATLAB code, citation:
#
# Brendan J. Meade, Algorithms for the calculation of exact 
# displacements, strains, and stresses for Triangular Dislocation 
# Elements in a uniform elastic half space, Computers & 
# Geosciences (2007), doi:10.1016/j.cageo.2006.12.003.


def displacement(sx, sy, sz, vertices, ss, ds, ts, nu=0.25):
    """
    Computes the displacement vector at an observation point due to slip on one
    triangular patch at depth.

    Args:
        * sx        : x coordinates of ground points
        * sy        : y coordinates of ground points
        * sz        : z coordinates of ground points
        * vertices  : list of 3-component vertices of the triangle
        * ss        : amount of strike-slip
        * ds        : amount of dip-slip
        * ts        : amount of tensile/opening-slip
        * nu        : Poisson's ratio
    
    Returns:
        * ux        : x component of displacement
        * uy        : y component of displacement
        * uz        : z component of displacement
    """

    # Modify input vertices by appending first point to end
    verts = copy.deepcopy(vertices)
    verts.append(verts[0])
    verts = [np.array(p) for p in verts]
    p1, p2, p3 = verts[:3]

    # Compute normal vector
    dx1 = p2 - p1
    dx2 = p3 - p1 
    normVec = np.cross(dx1, dx2)
    normVec /= np.linalg.norm(normVec)
    # Enforce clockwise circulation
    if normVec[2] < 0: 
        normVec *= -1.0
        # Update vertices by swapping 2nd and 3rd vertices
        verts = [p1, p3, p2, p1]
    
    # Compute slip vector in global coordinates
    angle = np.arctan2(normVec[1], normVec[0])
    strikeVec = np.array([-np.sin(angle), np.cos(angle), 0.0])
    dipVec = np.cross(normVec, strikeVec)
    # Change sign of slip components to be consistent with Okada (BVR)
    slipComp = np.array([-ss, -ds, -ts])
    slipVec = np.dot(np.column_stack((strikeVec, dipVec, normVec)), slipComp)

    # Allocate solution vectors
    ux = np.zeros_like(sx)
    uy = np.zeros_like(sx)
    uz = np.zeros_like(sx)
    
    # Loop over legs of triangle
    halfPi = 0.5 * np.pi
    for iTri in range(3):
    
        # Get the vertices of the leg
        p1, p2 = verts[iTri], verts[iTri+1]
        x1, y1, z1 = p1
        x2, y2, z2 = p2
    
        # Calculate strike and dip of current leg
        dx, dy = x2 - x1, y2 - y1
        strike = np.arctan2(dy, dx)
        segMapLength = np.sqrt(dx**2 + dy**2)
        rx, ry = RotateXyVec(dx, dy, -strike)
        dip = np.arctan2(z2 - z1, rx)
    
        if dip >= 0.0:
            beta = halfPi - dip
            if beta > halfPi:
                beta = halfPi - beta
        else:
            beta = -1.0 * (halfPi + dip)
            if beta < -halfPi:
                beta = halfPi - abs(beta)

        ssVec = np.array([np.cos(strike), np.sin(strike), 0.0])
        tsVec = np.array([-np.sin(strike), np.cos(strike), 0.0])
        dsVec = np.cross(ssVec, tsVec)
        lss = np.dot(slipVec, ssVec)
        lts = np.dot(slipVec, tsVec)
        lds = np.dot(slipVec, dsVec)
    
        if (abs(beta) > 1.0e-6) and (abs(beta - np.pi) > 1.0e-6):
            # First angular dislocation
            sx1, sy1 = RotateXyVec(sx - x1, sy - y1, -strike)
            ux1, uy1, uz1 = adv(sx1, sy1, sz - z1, z1, beta, nu, lss, lts, lds)

            # Second angular dislocation
            sx2, sy2 = RotateXyVec(sx - x2, sy - y2, -strike)
            ux2, uy2, uz2 = adv(sx2, sy2, sz - z2, z2, beta, nu, lss, lts, lds)
    
            # Rotate vectors to correct for strike
            uxn, uyn = RotateXyVec(ux1 - ux2, uy1 - uy2, strike)
            uzn = uz1 - uz2
         
            # Add the displacements from current leg
            ux += uxn
            uy += uyn
            uz += uzn

    # Identify indices for stations under current triangle
    p1, p2, p3 = verts[:3]
    path = Path([p1[:2], p2[:2], p3[:2], p1[:2]])
    inPolyIdx = path.contains_points(np.column_stack((sx, sy))).nonzero()[0]
    underIdx = []
    for iIdx in inPolyIdx:
        d = LinePlaneIntersect(sx[iIdx], sy[iIdx], sz[iIdx], p1, p2, p3)
        if d[2] - sz[iIdx] < 0.0:
            underIdx.append(iIdx)
    
    # Apply static offset to the points that lie underneath the current triangle
    ux[underIdx] -= slipVec[0]
    uy[underIdx] -= slipVec[1]
    uz[underIdx] -= slipVec[2]

    # Make sure to change sign on the vertical displacements!
    return ux, uy, -uz


def LinePlaneIntersect(sx, sy, sz, p1, p2, p3):
    """
    Calculate the intersection of a line and a plane using a parametric
    representation of the plane.  This is hardcoded for a vertical line.

    Args:
        * sx        : x coordinates of ground points
        * sy        : y coordinates of ground points
        * sz        : z coordinates of ground points
        * p1        : xyz tuple or list of first triangle vertex
        * p2        : xyz tuple or list of second triangle vertex
        * p3        : xyz tuple or list of third triangle vertex
    """
    # Extract the separate x,y,z values
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3

    numerator = np.array([[1.0, 1.0, 1.0, 1.0],
                          [x1, x2, x3, sx],
                          [y1, y2, y3, sy],
                          [z1, z2, z3, sz]])
    numerator = np.linalg.det(numerator)
    denominator = np.array([[1.0, 1.0, 1.0, 0.0], 
                            [x1, x2, x3, 0.0],
                            [y1, y2, y3, 0.0], 
                            [z1, z2, z3, -sz]])
    denominator = np.linalg.det(denominator)
    if denominator == 0:
        denominator = 1.0e-15
    # Parameteric curve parameter
    t = numerator / denominator
    d = np.array([sx, sy, sz]) - np.array([0.0, 0.0, -sz]) * t

    return d


def RotateXyVec(x, y, alpha):
    """
    Rotate components by an angle alpha.
    """
    sina, cosa = np.sin(alpha), np.cos(alpha)
    xp = x * cosa - y * sina + np.spacing(1)
    yp = x * sina + y * cosa + np.spacing(1)
    return xp, yp


def adv(y1, y2, y3, a, beta, nu, B1, B2, B3):
    """    
    These are the displacements in a uniform elastic half space due to slip
    on an angular dislocation (Comninou and Dunders, 1975).  Some of the
    equations for the B2 and B3 cases have been corrected following Thomas
    1993.  The equations are coded in way such that they roughly correspond
    to each line in original text.  Exceptions have been made where it made 
    more sense because of grouping symbols.
    """

    sinbeta = np.sin(beta)
    cosbeta = np.cos(beta)
    cotbeta = 1.0 / np.tan(beta)
    z1 = y1 * cosbeta - y3 * sinbeta
    z3 = y1 * sinbeta + y3 * cosbeta
    R2 = y1 * y1 + y2 * y2 + y3 * y3
    R = np.sqrt(R2)
    y3bar = y3 + 2.0 * a
    z1bar =  y1 * cosbeta + y3bar * sinbeta
    z3bar = -y1 * sinbeta + y3bar * cosbeta
    R2bar = y1 * y1 + y2 * y2 + y3bar * y3bar
    Rbar = np.sqrt(R2bar)
    F = (-np.arctan2(y2, y1) + np.arctan2(y2, z1) 
        + np.arctan2(y2*R*sinbeta, y1*z1 + y2*y2*cosbeta))
    Fbar = (-np.arctan2(y2, y1) + np.arctan2(y2, z1bar) 
           + np.arctan2(y2*Rbar*sinbeta, y1*z1bar + y2*y2*cosbeta))
    Rbar3 = Rbar**3
    cotbeta2 = cotbeta**2
    Rby3bar = Rbar + y3bar
    Rbz3bar = Rbar + z3bar

    # Cache Poisson's ratio terms
    nu1 = 1.0 - nu
    nu2 = 1.0 - 2.0 * nu
    
    # Case I: Burgers vector (B1,0,0)
    v1InfB1 = (2.0 * nu1 * (F + Fbar) - y1 * y2 * (1.0 / (R * (R - y3))
             + 1.0 / (Rbar * (Rby3bar))) - y2 * cosbeta * ((R * sinbeta - y1) 
             / (R * (R-z3)) + (Rbar * sinbeta - y1) / (Rbar * (Rbz3bar))))
    v2InfB1 = (nu2 * (np.log(R - y3) + np.log(Rby3bar) 
              - cosbeta * (np.log(R - z3) + np.log(Rbz3bar))) 
              - y2 * y2 * (1.0 / (R * (R - y3)) + 1.0 / (Rbar * (Rby3bar)) 
              - cosbeta * (1.0 / (R * (R - z3)) + 1.0 / (Rbar * (Rbz3bar)))))
    v3InfB1 = (y2 * (1.0 / R - 1.0 / Rbar - cosbeta * ((R * cosbeta - y3) / (R * (R - z3)) 
            - (Rbar * cosbeta + y3bar) / (Rbar * (Rbz3bar)))))
    denom = 1.0 / (8.0 * np.pi * nu1)
    v1InfB1 *= denom
    v2InfB1 *= denom
    v3InfB1 *= denom
    
    v1CB1 = (-2.0 * nu1 * nu2 * Fbar * (cotbeta2) 
           + nu2 * y2 / (Rby3bar) * ((1.0 - 2.0 * nu - a / Rbar) * cotbeta 
           - y1 / (Rby3bar) * (nu + a / Rbar)) + nu2 * y2 * cosbeta * cotbeta 
           / (Rbz3bar) * (cosbeta + a / Rbar) + a * y2 * (y3bar - a) * cotbeta 
           / Rbar3 + y2 * (y3bar - a) / (Rbar * (Rby3bar)) * (-(1.0 - 2.0*nu)
           * cotbeta + y1 / (Rby3bar) * (2.0 * nu + a / Rbar) + a * y1 / (Rbar * Rbar)) 
           + y2 * (y3bar - a) / (Rbar * (Rbz3bar)) * (cosbeta / (Rbz3bar) 
           * ((Rbar * cosbeta + y3bar) * (nu2 * cosbeta - a / Rbar) * cotbeta 
           + 2.0 * nu1 * (Rbar * sinbeta - y1) * cosbeta) - a * y3bar *cosbeta *cotbeta 
           / (Rbar * Rbar)))

    v2CB1 = (nu2 * ((2.0 * nu1 * (cotbeta2) - nu) 
           * np.log(Rby3bar) - (2.0 * nu1 * (cotbeta2) + 1.0 - 2.0 * nu) 
           * cosbeta * np.log(Rbz3bar)) - nu2 / (Rby3bar) * (y1 * cotbeta
           * (1.0 - 2.0 * nu - a / Rbar) + nu * y3bar - a + (y2 * y2) / (Rby3bar) 
           * (nu + a / Rbar)) - nu2 * z1bar * cotbeta / (Rbz3bar) * (cosbeta + a / Rbar) 
           - a * y1 * (y3bar - a) * cotbeta / Rbar3 + (y3bar - a) / (Rby3bar)
           * (-2.0 * nu + 1.0 / Rbar * (nu2 * y1 * cotbeta - a) + (y2 * y2) 
           / (Rbar * (Rby3bar)) * (2.0 * nu + a / Rbar) + a * (y2 * y2) 
           / Rbar3) + (y3bar - a) / (Rbz3bar) * ((cosbeta * cosbeta) - 1.0 
           / Rbar * (nu2 * z1bar * cotbeta + a * cosbeta) + a * y3bar * z1bar 
           * cotbeta / Rbar3 - 1.0 / (Rbar * (Rbz3bar)) * ((y2 * y2) 
           * (cosbeta * cosbeta) - a * z1bar * cotbeta / Rbar * (Rbar * cosbeta + y3bar))))
    
    v3CB1 = (2.0 * nu1 * ((nu2 * Fbar * cotbeta) + (y2 / (Rby3bar) 
          * (2.0 * nu + a / Rbar)) - (y2 * cosbeta / (Rbz3bar) * (cosbeta + a / Rbar))) 
          + y2 * (y3bar - a) / Rbar * (2 * nu / (Rby3bar) + a / (Rbar * Rbar)) + y2 
          * (y3bar - a) * cosbeta / (Rbar * (Rbz3bar)) * (1.0 -2.0 * nu - (Rbar * cosbeta
          + y3bar) / (Rbz3bar) * (cosbeta + a / Rbar) - a * y3bar / (Rbar * Rbar)))
   
    denom = 1.0 / (4.0 * np.pi * nu1) 
    v1CB1 *= denom 
    v2CB1 *= denom 
    v3CB1 *= denom 
    
    v1B1 = v1InfB1 + v1CB1
    v2B1 = v2InfB1 + v2CB1
    v3B1 = v3InfB1 + v3CB1

    # Case II: Burgers vector (0,B2,0)
    v1InfB2 = (-nu2 * (np.log(R - y3) + np.log(Rby3bar) - cosbeta 
            * (np.log(R - z3) + np.log(Rbz3bar))) + y1 * y1 * (1.0 / (R * (R - y3)) + 1.0 
            / (Rbar * (Rby3bar))) + z1 * (R * sinbeta - y1) / (R * (R - z3)) + z1bar 
            * (Rbar * sinbeta - y1) / (Rbar * (Rbz3bar)))
    v2InfB2 = (2.0 * nu1 * (F + Fbar) + y1 * y2 * (1.0 / (R * (R - y3)) + 1.0 / (Rbar 
            * (Rby3bar))) - y2 * (z1 / (R * (R - z3)) + z1bar / (Rbar * (Rbz3bar))))
    v3InfB2 = (-nu2 * sinbeta * (np.log(R - z3) - np.log(Rbz3bar)) - y1 
            * (1.0 / R - 1.0 / Rbar) + z1 * (R * cosbeta - y3) / (R * (R - z3)) - z1bar 
            * (Rbar * cosbeta + y3bar) / (Rbar * (Rbz3bar)))
    denom = 1.0 / (8.0 * np.pi * nu1)
    v1InfB2 *= denom
    v2InfB2 *= denom
    v3InfB2 *= denom
    
    v1CB2 = (nu2 * ((2.0 * nu1 * (cotbeta2) + nu) 
          * np.log(Rby3bar) - (2.0 * nu1 * (cotbeta2) + 1.0) * cosbeta
          * np.log(Rbz3bar)) + nu2 / (Rby3bar) * (-nu2 
          * y1 * cotbeta + nu * y3bar - a + a * y1 * cotbeta / Rbar + (y1 * y1) / (Rbar 
          + y3bar) * (nu + a / Rbar)) - nu2 * cotbeta / (Rbz3bar) * (z1bar * cosbeta 
          - a * (Rbar * sinbeta - y1) / (Rbar * cosbeta)) - a * y1 * (y3bar - a) * cotbeta 
          / Rbar3 + (y3bar - a) / (Rby3bar) * (2.0 * nu + 1.0 / Rbar 
          * (nu2 * y1 * cotbeta + a) - (y1 * y1) / (Rbar * (Rby3bar)) 
          * (2.0 * nu + a / Rbar) - a * (y1 * y1) / Rbar3) + (y3bar - a) * cotbeta
          / (Rbz3bar) * (-cosbeta * sinbeta + a * y1 * y3bar / (Rbar3 * cosbeta) 
          + (Rbar * sinbeta - y1) / Rbar * (2.0 * nu1 * cosbeta - (Rbar * cosbeta + y3bar) 
          / (Rbar+z3bar) * (1.0 + a / (Rbar * cosbeta)))))

    v2CB2 = (2.0 * nu1 * nu2 * Fbar * cotbeta2 + (1.0 - 2 * nu) 
          * y2 / (Rby3bar) * (-(1.0 - 2.0 * nu - a / Rbar) * cotbeta + y1 / (Rby3bar)
          * (nu + a / Rbar)) - nu2 * y2 * cotbeta / (Rbz3bar) * (1.0 + a 
          / (Rbar * cosbeta)) - a * y2 * (y3bar - a) * cotbeta / Rbar3 + y2 
          * (y3bar - a) / (Rbar * (Rby3bar)) * (nu2 * cotbeta - 2.0 * nu * y1 
          / (Rby3bar) - a * y1 / Rbar * (1.0 / Rbar + 1.0 / (Rby3bar))) + y2 
          * (y3bar - a) * cotbeta / (Rbar * (Rbz3bar)) * (-2.0 * nu1 * cosbeta 
          + (Rbar * cosbeta + y3bar) / (Rbz3bar) * (1.0 + a / (Rbar * cosbeta)) + a 
          * y3bar / ((Rbar * Rbar) * cosbeta)))

    v3CB2 = (-2.0 * nu1 * nu2 * cotbeta * (np.log(Rby3bar) - cosbeta
          * np.log(Rbz3bar)) - 2.0 * nu1 * y1 / (Rby3bar) * (2.0 * nu + a / Rbar) 
          + 2.0 * nu1 * z1bar / (Rbz3bar) * (cosbeta + a / Rbar) + (y3bar - a) / Rbar
          * (nu2 * cotbeta - 2.0 * nu * y1 / (Rby3bar) - a * y1 / (Rbar * Rbar)) 
          - (y3bar - a) / (Rbz3bar) * (cosbeta * sinbeta + (Rbar * cosbeta + y3bar) 
          * cotbeta / Rbar * (2.0 * nu1 * cosbeta - (Rbar * cosbeta + y3bar) / (Rbz3bar)) 
          + a / Rbar * (sinbeta - y3bar * z1bar / (Rbar * Rbar) - z1bar * (Rbar * cosbeta 
          + y3bar) / (Rbar * (Rbz3bar)))))

    denom = 1.0 / (4.0 * np.pi * nu1)
    v1CB2 *= denom
    v2CB2 *= denom
    v3CB2 *= denom
    v1B2 = v1InfB2 + v1CB2
    v2B2 = v2InfB2 + v2CB2
    v3B2 = v3InfB2 + v3CB2

    # Case III: Burgers vector (0,0,B3)
    v1InfB3 = (y2 * sinbeta * ((R * sinbeta - y1) / (R * (R - z3)) + (Rbar * sinbeta - y1) 
            / (Rbar * (Rbz3bar))))
    v2InfB3 = (nu2 * sinbeta * (np.log(R - z3) + np.log(Rbz3bar)) - (y2 * y2) 
            * sinbeta * (1.0 / (R * (R - z3)) + 1.0 / (Rbar * (Rbz3bar))))
    v3InfB3 = (2.0 * nu1 * (F - Fbar) + y2 * sinbeta * ((R * cosbeta - y3) / (R 
            * (R - z3)) - (Rbar * cosbeta + y3bar) / (Rbar * (Rbz3bar))))
    denom = 1.0 / (8.0 * np.pi * nu1)
    v1InfB3 *= denom
    v2InfB3 *= denom
    v3InfB3 *= denom
    
    v1CB3 = (nu2 * (y2 / (Rby3bar) * (1.0 + a / Rbar) - y2 * cosbeta / (Rbar
          + z3bar) * (cosbeta + a / Rbar)) - y2 * (y3bar - a) / Rbar * (a / (Rbar * Rbar) 
          + 1.0 / (Rby3bar)) + y2 * (y3bar - a) * cosbeta / (Rbar * (Rbz3bar)) 
          * ((Rbar * cosbeta + y3bar) / (Rbz3bar) * (cosbeta + a / Rbar) + a * y3bar 
          / (Rbar * Rbar)))

    v2CB3 = (nu2 * (-sinbeta * np.log(Rbz3bar) - y1 / (Rby3bar) * (1.0 
          + a / Rbar) + z1bar / (Rbz3bar) * (cosbeta + a / Rbar)) + y1 * (y3bar - a) 
          / Rbar * (a / (Rbar * Rbar) + 1.0 / (Rby3bar)) - (y3bar - a) / (Rbz3bar) 
          * (sinbeta * (cosbeta - a / Rbar) + z1bar / Rbar * (1 + a * y3bar / (Rbar * Rbar)) 
          - 1.0 / (Rbar * (Rbz3bar)) * ((y2 * y2) * cosbeta * sinbeta - a * z1bar / Rbar 
          * (Rbar * cosbeta + y3bar))))

    v3CB3 = (2.0 * nu1 * Fbar + 2.0 * nu1 * (y2 * sinbeta / (Rbz3bar) 
          * (cosbeta + a / Rbar)) + y2 * (y3bar - a) * sinbeta / (Rbar * (Rbz3bar)) 
          * (1.0 + (Rbar * cosbeta + y3bar) / (Rbz3bar) * (cosbeta + a / Rbar) + a 
          * y3bar / (Rbar * Rbar)))

    denom = 1.0 / (4.0 * np.pi * nu1)
    v1CB3 *= denom
    v2CB3 *= denom
    v3CB3 *= denom
    
    v1B3 = v1InfB3 + v1CB3
    v2B3 = v2InfB3 + v2CB3
    v3B3 = v3InfB3 + v3CB3

    # Sum the for each slip component
    v1 = B1 * v1B1 + B2 * v1B2 + B3 * v1B3
    v2 = B1 * v2B1 + B2 * v2B2 + B3 * v2B3
    v3 = B1 * v3B1 + B2 * v3B2 + B3 * v3B3

    return v1, v2, v3

#EOF
