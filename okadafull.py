'''
A group of routines that allows to interact with the okada4py routine.

Written by R. Jolivet, Feb 2014.
'''

# External
import numpy as np
import sys

# Personals
import okada4py as ok92

# NOTE: In this convention, ss(+) means left-lateral, and ss(-) means right-lateral

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
def displacement(xs, ys, zs, xc, yc, zc, width, length, strike, dip, ss, ds, ts, nu=0.25):
    '''
    Returns the displacements at the stations located on (xs, ys, zs) for patches
        with centers on (xc, yc, zc). All arguments can be float, list or array.
    '''

    # Here Mu can be anything. RJ tested it and the displacement is not-sensitive to Mu as it should be.
    # Although, it does not work with Mu = 0.0 GPa... So we take a random value of 30GPa
    mu = 30e9

    # Nu does matter here, and it is by default 0.25

    # Check 
    xs, ys, zs = ArraySizes(xs, ys, zs)
    xc, yc, zc, width, length, strike, dip, ss, ds, ts = ArraySizes(xc, yc, zc, width, length, strike, dip, ss, ds, ts)

    # Normally, StaticInv does angles in Radians
    dip = dip*180./np.pi
    strike = strike*180./np.pi

    # Run okada
    u, d, s, flag, flag2 = ok92.okada92(xs, ys, zs, xc, yc, zc, length, width, dip, strike, ss, ds, ts, mu, nu)

    # Check if things went well
    if not (flag==0).all():
        if not np.where(flag!=0)==[]:
            print(' Error: {}'.format(tuple(np.where(flag!=0))))
            print('Something went wrong in okada4py... You should check...')

    # Reshape the displacement
    u = u.reshape((len(xs), 3))

    # All Done
    return u
        
#--------------------------------------------------
# Strain only
def strain(xs, ys, zs, xc, yc, zc, width, length, strike, dip, ss, ds, ts, nu=0.25, full=False):
    '''
    Returns the strain at the stations located on (xs, ys, zs) for patches
        with centers on (xc, yc, zc). All arguments can be float, list or array.
    if Full is True, returns the full strain tensor, 
            is False, returns and array [nstations, 9] = [Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz]
    '''

    # Here Mu can be anything. RJ tested it and the trainis not-sensitive to Mu as it should be.
    # Although, it does not work with Mu = 0.0 GPa... So we take a random value of 30GPa
    mu = 30e9

    # Nu does matter here, and it is by default 0.25

    # Check 
    xs, ys, zs = ArraySizes(xs, ys, zs)
    xc, yc, zc, width, length, strike, dip, ss, ds, ts = ArraySizes(xc, yc, zc, width, length, strike, dip, ss, ds, ts)

    # Normally, StaticInv does angles in Radians
    dip = dip*180./np.pi
    strike = strike*180./np.pi

    # Run okada
    u, d, s, flag, flag2 = ok92.okada92(xs, ys, zs, xc, yc, zc, length, width, dip, strike, ss, ds, ts, mu, nu)

    # Check if things went well
    if not (flag==0).all():
        if not np.where(flag!=0)==[]:
            print(' Error: {}'.format(tuple(np.where(flag!=0))))
            print('Something went wrong in okada4py... You should check...')

    # Reshape the displacement
    d = d.reshape((len(xs), 9))

    if not full: 
        return d
    else:
        # Strain
        Strain = np.zeros((3,3,len(xs)))
        # Fill it
        Strain[0,0,:] = d[:,0]  # Uxx
        Strain[0,1,:] = d[:,1]  # Uxy
        Strain[0,2,:] = d[:,2]  # Uxz
        Strain[1,0,:] = d[:,3]  # Uyx
        Strain[1,1,:] = d[:,4]  # Uyy
        Strain[1,2,:] = d[:,5]  # Uyz
        Strain[2,0,:] = d[:,6]  # Uzx
        Strain[2,1,:] = d[:,7]  # Uzy
        Strain[2,2,:] = d[:,8]  # UUzz
        return Strain
         
#--------------------------------------------------
# Stress only
def stress(xs, ys, zs, xc, yc, zc, width, length, strike, dip, ss, ds, ts, mu=30e9, nu=0.25, full=False):
    '''
    Returns the stress at the stations located on (xs, ys, zs) for patches
        with centers on (xc, yc, zc). All arguments can be float, list or array.
    if Full is True, returns the full strain tensor, 
            is False, returns and array [nstations, 6] = [Sxx, Sxy, Sxz, Syy, Syz, Szz]
    '''

    # Mu and Nu do matter here, there is default values, but feel free to change...

    # Check 
    xs, ys, zs = ArraySizes(xs, ys, zs)
    xc, yc, zc, width, length, strike, dip, ss, ds, ts = ArraySizes(xc, yc, zc, width, length, strike, dip, ss, ds, ts)

    # Normally, StaticInv does angles in Radians
    dip = dip*180./np.pi
    strike = strike*180./np.pi

    # Run okada
    u, d, s, flag, flag2 = ok92.okada92(xs, ys, zs, xc, yc, zc, length, width, dip, strike, ss, ds, ts, mu, nu)

    # Check if things went well
    if not (flag==0.).all():
        if not np.where(flag!=0)==[]:
            print('Something went wrong in okada4py... You should check...')
            print(' Error: {}'.format(tuple(np.where(flag!=0.))))

    # Reshape the displacement
    s = s.reshape((len(xs), 6))

    if not full:
        return s, flag, flag2
    else:
        Stress = np.zeros((3, 3, len(xs)))
        Stress[0,0,:] = s[:,0]  # Sxx
        Stress[1,1,:] = s[:,3]  # Syy
        Stress[2,2,:] = s[:,5]  # Szz
        Stress[0,1,:] = s[:,1]  # Sxy
        Stress[1,0,:] = s[:,1]  # Sxy
        Stress[0,2,:] = s[:,2]  # Sxz
        Stress[2,0,:] = s[:,2]  # Sxz
        Stress[1,2,:] = s[:,4]  # Syz
        Stress[2,1,:] = s[:,4]  # Syz
        return Stress, flag, flag2
         
#EOF




