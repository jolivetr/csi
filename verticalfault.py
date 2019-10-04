'''
A class that deals with vertical faults.

Written by R. Jolivet, April 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import copy
import sys

# Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok
from .RectangularPatches import RectangularPatches

class verticalfault(RectangularPatches):

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        '''
        Args:
            * name          : Name of the fault.
        '''

        # Initialize base class
        super(verticalfault,self).__init__(name,utmzone=utmzone,ellps=ellps,lon0=lon0,lat0=lat0)

        # All done
        return

    def extrapolate(self, length_added=50, tol=2., fracstep=5., extrap='ud'):
        ''' 
        Extrapolates the surface trace. This is usefull when building deep patches for interseismic loading.
        Args:
            * length_added  : Length to add when extrapolating.
            * tol           : Tolerance to find the good length.
            * fracstep      : control each jump size.
            * extrap        : if u in extrap -> extrapolates at the end
                              if d in extrap -> extrapolates at the beginning
                              default is 'ud'
        '''

        # print 
        print ("Extrapolating the fault for {} km".format(length_added))

        # Check if the fault has been interpolated before
        if self.xi is None:
            print ("Run the discretize() routine first")
            return

        # Build the interpolation routine
        import scipy.interpolate as scint
        fi = scint.interp1d(self.xi, self.yi)

        # Build the extrapolation routine
        fx = self.extrap1d(fi)

        # make lists
        self.xi = self.xi.tolist()
        self.yi = self.yi.tolist()

        if 'd' in extrap:
        
            # First guess for first point
            xt = self.xi[0] - length_added/2.
            yt = fx(xt)
            d = np.sqrt( (xt-self.xi[0])**2 + (yt-self.yi[0])**2)

            # Loop to find the best distance
            while np.abs(d-length_added)>tol:
                # move up or down
                if (d-length_added)>0:
                    xt = xt + d/fracstep
                else:
                    xt = xt - d/fracstep
                # Get the corresponding yt
                yt = fx(xt)
                # New distance
                d = np.sqrt( (xt-self.xi[0])**2 + (yt-self.yi[0])**2) 
        
            # prepend the thing
            self.xi.reverse()
            self.xi.append(xt)
            self.xi.reverse()
            self.yi.reverse()
            self.yi.append(yt)
            self.yi.reverse()

        if 'u' in extrap:

            # First guess for the last point
            xt = self.xi[-1] + length_added/2.
            yt = fx(xt)
            d = np.sqrt( (xt-self.xi[-1])**2 + (yt-self.yi[-1])**2)

            # Loop to find the best distance
            while np.abs(d-length_added)>tol:
                # move up or down
                if (d-length_added)<0:
                    xt = xt + d/fracstep
                else:
                    xt = xt - d/fracstep
                # Get the corresponding yt
                yt = fx(xt)
                # New distance
                d = np.sqrt( (xt-self.xi[-1])**2 + (yt-self.yi[-1])**2)

            # Append the result
            self.xi.append(xt)
            self.yi.append(yt)

        # Make them array again
        self.xi = np.array(self.xi)
        self.yi = np.array(self.yi)

        # Build the corresponding lon lat arrays
        self.loni, self.lati = self.xy2ll(self.xi, self.yi)

        # All done
        return

    def extrap1d(self,interpolator):
        '''
        Linear extrapolation routine. Found on StackOverflow by sastanin.
        '''

        # import a bunch of stuff
        from scipy import arange, array, exp

        xs = interpolator.x
        ys = interpolator.y
        def pointwise(x):
            if x < xs[0]:
                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            elif x > xs[-1]:
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                return interpolator(x)
        def ufunclike(xs):
            return pointwise(xs) #array(map(pointwise, array(xs)))
        return ufunclike
    
    def setDepth(self, depth, top=0, num=5):
        '''
        Set the maximum depth of the fault patches.

        Args:
            * depth         : Depth of the fault patches.
            * num           : Number of fault patches at depth.
        '''

        # Set depth
        self.top = top
        self.depth = depth
        self.numz = num

        # All done
        return

    def build_patches(self):
        '''
        Builds rectangular patches from the discretized fault.
        A patch is a list of 4 corners.
        '''

        # If the maximum depth and the number of patches is not set
        if self.depth is None:
            print("Depth and number of patches are not set.")
            print("Please use setdepth to define maximum depth and number of patches")
            return

        print ("Build patches for fault {} between depths: {}, {}".format(self.name, self.top, self.depth))

        # Define the depth vector
        z = np.linspace(self.top, self.depth, num=self.numz+1)
        self.z_patches = z

        # If the discretization is not done
        if self.xi is None:
            self.discretize()

        # Define a patch list
        self.patch = []
        self.patchll = []
        self.slip = []

        # Iterate over the surface discretized fault points
        for i in range(len(self.xi)-1):
            # First corner
            x1 = self.xi[i]
            y1 = self.yi[i]
            lon1 = self.loni[i]
            lat1 = self.lati[i]
            # Second corner
            x2 = self.xi[i]
            y2 = self.yi[i]
            lon2 = self.loni[i]
            lat2 = self.lati[i]
            # Third corner
            x3 = self.xi[i+1]
            y3 = self.yi[i+1]
            lon3 = self.loni[i+1]
            lat3 = self.lati[i+1]
            # Fourth corner
            x4 = self.xi[i+1]
            y4 = self.yi[i+1]
            lon4 = self.loni[i+1]
            lat4 = self.lati[i+1]
            # iterate at depth
            for j in range(len(z)-1):
                p = np.zeros((4,3))
                pll = np.zeros((4,3))
                p[0,:] = [x1, y1, z[j]]
                pll[0,:] = [lon1, lat1, z[j]]
                p[3,:] = [x2, y2, z[j+1]]
                pll[3,:] = [lon2, lat2, z[j+1]]
                p[2,:] = [x3, y3, z[j+1]]
                pll[2,:] = [lon3, lat3, z[j+1]]
                p[1,:] = [x4, y4, z[j]]
                pll[1,:] = [lon4, lat4, z[j]]
                self.patch.append(p)
                self.patchll.append(pll)
                self.slip.append([0.0, 0.0, 0.0])

        # Translate slip to np.array
        self.slip = np.array(self.slip)

        # Compute the equivalent patches
        self.equivpatch = copy.deepcopy(self.patch)

        # All done
        return

    def BuildPatchesVarResolution(self, depths, Depthpoints, Resolpoints, interpolation='linear', minpatchsize=0.1, extrap=None):
        '''
        Patchizes the fault with a variable patch size at depth.
        The variable patch size is given by the respoints table.
        Depthpoints = [depth1, depth2, depth3, ...., depthN]
        Resolpoints = [Resol1, Resol2, Resol3, ...., ResolN]
        The final resolution is interpolated given the 'interpolation' method.
        Interpolation can be 'linear', 'cubic'.
        '''

        print('Build fault patches for fault {} between {} and {} km deep, with a variable resolution'.format(self.name, self.top, self.depth))

        # Define the depth vector
        z = np.array(depths)
        self.z_patches = z

        # Interpolate the resolution
        if interpolation is not 'nointerpolation':
            fint = sciint.interp1d(Depthpoints, Resolpoints, kind=interpolation)
            resol = fint(z)
        else:
            resol = Resolpoints

        # build lists for storing things
        self.patch = []
        self.patchll = []
        self.slip = []

        # iterate over the depths 
        for j in range(len(z)-1):

            # discretize the fault at the desired resolution
            print('Discretizing at depth {}'.format(z[j]))
            self.discretize(every=np.floor(resol[j]), tol=resol[j]/20., fracstep=resol[j]/1000.)
            if extrap is not None:
                self.extrapolate(length_added=extrap[0], extrap=extrap[1])

            # iterate over the discretized fault
            for i in range(len(self.xi)-1):
                # First corner
                x1 = self.xi[i]
                y1 = self.yi[i]
                lon1 = self.loni[i]
                lat1 = self.lati[i]
                # Second corner
                x2 = self.xi[i]
                y2 = self.yi[i]
                lon2 = self.loni[i]
                lat2 = self.lati[i]
                # Third corner
                x3 = self.xi[i+1]
                y3 = self.yi[i+1]
                lon3 = self.loni[i+1]
                lat3 = self.lati[i+1]
                # Fourth corner
                x4 = self.xi[i+1]
                y4 = self.yi[i+1]
                lon4 = self.loni[i+1]
                lat4 = self.lati[i+1]
                # build patches
                p = np.zeros((4,3))
                pll = np.zeros((4,3))
                # fill them
                p[0,:] = [x1, y1, z[j]]
                pll[0,:] = [lon1, lat1, z[j]]
                p[3,:] = [x2, y2, z[j+1]]
                pll[3,:] = [lon2, lat2, z[j+1]]
                p[2,:] = [x3, y3, z[j+1]]
                pll[2,:] = [lon3, lat3, z[j+1]]
                p[1,:] = [x4, y4, z[j]]
                pll[1,:] = [lon4, lat4, z[j]]
                psize = np.sqrt( (x3-x2)**2 + (y3-y2)**2 )
                if psize>minpatchsize:
                    self.patch.append(p)
                    self.patchll.append(pll)
                    self.slip.append([0.0, 0.0, 0.0])
                else:           # Increase the size of the previous patch
                    self.patch[-1][2,:] = [x3, y3, z[j+1]]
                    self.patch[-1][1,:] = [x4, y4, z[j]]
                    self.patchll[-1][2,:] = [lon3, lat3, z[j+1]]
                    self.patchll[-1][1,:] = [lon4, lat4, z[j]]


        # Translate slip into a np.array
        self.slip = np.array(self.slip)

        # Compute the equivalent patches
        self.computeEquivRectangle()

        # all done
        return

    def cutPatchesVertically(self, iP, cuttingDepth):
        '''
        Cut a patche into 2 patches at depth given by cuttingDepth.
        Args:
            * iP            : patch index or list of patch indexes.
            * cuttingDepth  : Depth where patch is going to be split in 2.
        '''

        # Check
        if type(iP) is not list:
            iP = [iP]

        # Iterate over patches
        for p in iP:
    
            # Get patch value
            patch = self.patch[p]

            # Get values
            x1, y1, z1 = patch[0]
            x2, y2 = patch[1][:2]
            z2 = patch[2][2]

            # Make 2 patches
            patchUp = [ [x1, y1, z1],
                        [x2, y2, z1],
                        [x2, y2, cuttingDepth],
                        [x1, y1, cuttingDepth] ]
            patchDown = [ [x1, y1, cuttingDepth],
                          [x2, y2, cuttingDepth],
                          [x2, y2, z2],
                          [x1, y1, z2] ]

            # Add patch
            self.addpatch(patchUp)
            self.addpatch(patchDown)

        # Delete the old patches
        self.deletepatches(iP)

        # All done
        return

    def rotationHoriz(self, center, angle):
        '''
        Rotates the geometry of the fault around center, of an angle.
        Args:
            * center    : [lon,lat]
            * angle     : degrees
        '''

        # Translate the center to x, y
        xc, yc = self.ll2xy(center[0], center[1])
        ref = np.array([xc, yc])

        # Create the rotation matrix
        angle = angle*np.pi/180.
        Rot = np.array( [ [np.cos(angle), -1.0*np.sin(angle)],
                          [np.sin(angle), np.cos(angle)] ] )

        # Loop on the patches
        for i in range(len(self.patch)):

            # Get patch
            p = self.patch[i]
            pll = self.patchll[i]

            for j in range(4):
                x, y = np.dot( Rot, p[j][:-1] - ref )
                p[j][0] = x + xc
                p[j][1] = y + yc
                lon, lat = self.xy2ll(p[j][0],p[j][1])
                pll[j][0] = lon
                pll[j][1] = lat

        # All done 
        return

    def translationHoriz(self, dx, dy):
        '''
        Translates the patches.
        Args:
            * dx    : Translation along x (km)
            * dy    : Translation along y (km)
        '''

        # Loop on the patches
        for i in range(len(self.patch)):

            # Get patch
            p = self.patch[i]
            pll = self.patchll[i]

            for j in range(4):
                p[j][0] += dx
                p[j][1] += dy
                lon, lat = self.xy2ll(p[j][0],p[j][1])
                pll[j][0] = lon
                pll[j][1] = lat

        # All done 
        return


    def differentiateGFs(self, datas):
        '''
        Uses the Delaunay triangulation to prepare a differential Green's function matrix, data vector
        and data covariance matrix.
        Args:   
            * datas         : List of dataset concerned
        '''

        # Create temporary Green's function, data and Cd dictionaries to hold the new ones
        Gdiff = {}
        ddiff = {}

        # Loop over the datasets
        for data in datas:

            # Check something
            if data.dtype not in ('gps', 'multigps'):
                print('This has not been implemented for other data set than gps and multigps')
                return

            # Get the GFs, the data and the data covariance
            G = self.G[data.name]
            d = self.d[data.name]
            Cd = data.Cd

            # Get some size informations
            nstation = data.station.shape[0]
            lengthd = d.shape[0]
            if (lengthd == 3*nstation):
               vertical = True
               ncomp = 3
            else:
               vertical = False
               ncomp = 2

            # Get the couples
            edges = data.triangle['Edges']

            # How many lines/columns ?
            Nd = edges.shape[0]
            k = G.keys()[0]
            Np = G[k].shape[1]

            # Create the spaces
            Gdiff[data.name] = {}
            for key in G.keys():
                Gdiff[data.name][key] = np.zeros((Nd*ncomp, Np))
            ddiff[data.name] = np.zeros((Nd*ncomp,))
            Cddiff = np.zeros((Nd*ncomp, Nd*ncomp))

            # Loop over the lines of Edges
            for i in range(Nd):

                # Get the couple
                m = edges[i][0]
                n = edges[i][1]

                # Deal with the GFs
                for key in G.keys():
                    # East component
                    Line1 = G[key][m,:]
                    Line2 = G[key][n,:]
                    Gdiff[data.name][key][i,:] = Line1 - Line2
                    # North Component
                    Line1 = G[key][m+nstation,:]
                    Line2 = G[key][n+nstation,:]
                    Gdiff[data.name][key][i+Nd,:] = Line1 - Line2
                    # Vertical
                    if vertical:
                        Line1 = G[key][m+2*nstation,:]
                        Line2 = G[key][n+2*nstation,:]
                        Gdiff[data.name][key][i+2*Nd,:] = Line1 - Line2

                # Deal with the data vector
                # East
                d1 = d[m]
                d2 = d[n]
                ddiff[data.name][i] = d1 - d2
                # North
                d1 = d[m+nstation]
                d2 = d[n+nstation]
                ddiff[data.name][i+Nd] = d1 - d2
                # Vertical
                if vertical:
                    d1 = d[m+2*nstation]
                    d2 = d[n+2*nstation]
                    ddiff[data.name][i+2*Nd] = d1 - d2

                # Deal with the Covariance (Only diagonal, for now)
                # East
                cd1 = Cd[m,m]
                cd2 = Cd[n,n]
                Cddiff[i,i] = cd1+cd2
                # North
                cd1 = Cd[m+nstation,m+nstation]
                cd2 = Cd[n+nstation,n+nstation]
                Cddiff[i+Nd,i+Nd] = cd1+cd2
                # Vertical
                if vertical:
                    cd1 = Cd[m+2*nstation,m+2*nstation]
                    cd2 = Cd[n+2*nstation,n+2*nstation]
                    Cddiff[i+2*Nd,i+2*Nd] = cd1+cd2

            # Once the data loop is done, store Cd
            data.Cd = Cddiff

        # Once it is all done, store G and d
        self.G = Gdiff
        self.d = ddiff

        # All done
        return

    def computetotalslip(self):
        '''
        Computes the total slip.
        '''

        # Computes the total slip
        self.totalslip = np.sqrt(self.slip[:,0]**2 + self.slip[:,1]**2 + self.slip[:,2]**2)
    
        # All done
        return

    def associatePatch2PDFs(self, directory='.', prefix='step_001_param'):
        '''
        Associates a patch with a pdf called directory/prefix_{#}.dat.
        import AltarExplore....
        '''

        # Import necessary
        import AltarExplore as alt
        
        # Parameters index are in self.index_parameter
        istrikeslip = self.index_parameter[:,0]
        idipslip = self.index_parameter[:,1]
        itensile = self.index_parameter[:,2]

        # Create a list of slip pdfs
        self.slippdfs = []
        for i in range(self.slip.shape[0]):
            sys.stdout.write('\r Patch {}/{}'.format(i,self.slip.shape[0]))
            sys.stdout.flush()
            # integers are needed
            iss = np.int(istrikeslip[i])
            ids = np.int(idipslip[i])
            its = np.int(itensile[i])
            # Create the file names
            pss = None
            pds = None
            pts = None
            if istrikeslip[i]< 10000:
                pss = '{}/{}_{:03d}.dat'.format(directory, prefix, iss)
            if idipslip[i]<10000:
                pds = '{}/{}_{:03d}.dat'.format(directory, prefix, ids)
            if itensile[i]<10000:
                pts = '{}/{}_{:03d}.dat'.format(directory, prefix, its)
            # Create the parameters
            Pss = None; Pds = None; Pts = None
            if pss is not None:
                Pss = alt.parameter('{:03d}'.format(iss), pss)
            if pds is not None:
                Pds = alt.parameter('{:03d}'.format(ids), pds)
            if pts is not None:
                Pts = alt.parameter('{:03d}'.format(its), pts)
            # Store these
            self.slippdfs.append([Pss, Pds, Pts])

        sys.stdout.write('\n')
        sys.stdout.flush()

        # all done
        return

#EOF
