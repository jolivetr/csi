''' 
A class that deals with 3D velocity models.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import scipy.linalg as scilin
import shapely.geometry as geom
import scipy.interpolate as interp

from .SourceInv import SourceInv

class velocitymodel(SourceInv):

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Args:
            * name      : Name of the dataset.
            * utmzone   : UTM zone. Default is 10 (Western US).
        '''

        # Base class inita
        super(velocitymodel, self).__init__(name, utmzone=utmzone, 
                                            lon0=lon0, lat0=lat0,
                                            ellps=ellps)

        # Set things
        self.name = name

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize Velocity Model {}".format(self.name))
        self.verbose = verbose

        # Initialize things
        self.Vp = None
        self.Vs = None
        self.Rho = None
        
        self.lon = None
        self.lat = None 
        self.depth = None
        
        self.VpVert = None
        self.VsVert = None
        self.RhoVert = None

        self.StdVpVert = None
        self.StdVsVert = None
        self.StdRhoVert = None
        
        self.SimDepth = None
        self.SimVpVert = None
        self.SimVsVert = None
        self.SimRhoVert = None

        # All done
        return

    def readVpfromascii(self, filename, clon=0, clat=1, cdepth=2, cvp=3, hdr=0):
        '''
        Reads Lon, Lat, depth and Vp from an ascii file.
        Args:
            * filename      : name of the ascii file
            * clon          : index of the column for longitude.
            * clat          : index of the column for latitude.
            * cdepth        : index of the column for depth.
            * cvp           : index of the column for Vp.
            * hdr           : Header length.
        '''

        # Open the file
        fin = open(filename, 'r')

        # Read what's in there
        A = fin.readlines()

        # Initialize things
        Lon = []
        Lat = []
        Vp = []
        Depth = []

        # Loop on the file
        for i in range(hdr,len(A)):
            
            B = A[i].split()
            Lon.append(np.float(B[clon]))
            Lat.append(np.float(B[clat]))
            Depth.append(np.float(B[cdepth]))
            Vp.append(np.float(B[cvp]))

        # Make arrays
        self.lon = np.array(Lon)
        self.lat = np.array(Lat)
        self.Vp = np.array(Vp)
        self.depth = np.array(Depth)

        # LonLat2UTM
        self.x, self.y = self.lonlat2xy(self.lon, self.lat)

        # All done
        return

    def lonlat2xy(self, lon, lat):
        '''
        Lon/Lat to UTM transformation.
        Args:
            * lon       : Longitude array
            * lat       : Latitude array
        Returns:
            * x         : Easting (km)
            * y         : Northing (km)
        '''

        x, y = self.putm(lon, lat)
        x /= 1000.
        y /= 1000.

        # All done
        return x,y

    def xy2lonlat(self, x, y):
        '''
        UTM to Lon/Lat transformation.
        Args:
            * x         : Easting (km)
            * y         : Northing (km)
        Returns:
            * lon       : Longitude.
            * lat       : Latitude.
        '''

        # All done
        return self.putm(x*1000., y*1000., inverse=True)

    def SelectBox(self, box):
        '''
        Keeps only the points inside the box:
        Args:
            * box       : Coordinates of the UL and BR corners in lon,lat.
        '''

        # Import shapely
        import shapely.geometry as geom
        
        # Transform box into xy
        xul, yul = self.lonlat2xy(box[0], box[1])
        xbr, ybr = self.lonlat2xy(box[2], box[3])

        # Make a polygon with the box
        poly = geom.Polygon([ [xul,yul], [xbr,yul], [xbr,ybr], [xul,ybr], [xul,yul] ])

        # Make a list of points with self.x and self.y
        pl = np.vstack((self.x, self.y)).T.tolist()

        # Check if each point is in the box or not
        i = [poly.contains(geom.Point(p)) for p in pl]

        # Keep only the good points
        self.x = self.x[np.where(i)]
        self.y = self.y[np.where(i)]
        self.lon = self.lon[np.where(i)]
        self.lat = self.lat[np.where(i)]
        self.depth = self.depth[np.where(i)]

        # Vp
        if self.Vp is not None:
            self.Vp = self.Vp[np.where(i)]

        # Vs
        if self.Vs is not None:
            self.Vs = self.Vs[np.where(i)]

        # Density
        if self.Rho is not None:
            self.Rho = self.Rho[np.where(i)]

        # All done
        return
    
    def regrid(self, box=None, Nlon=100, Nlat=100, Ndepth=10, method='linear'):
        '''
        Re-samples the data into a regular grid.
        Args:
            * box       : Coordinates of the UL and BR corners in the UTM reference.
            * Nlon      : Number of Longitude points.
            * Nlat      : Number of Latitude points.
            * Ndepth    : Number of depths.
            * method    : interpolation method ('linear', 'nearest' or 'cubic').
        '''

        # Get the box
        if box is None:
            box = [self.lon.min(), self.lat.max(), self.lon.max(), self.lat.min()]

        # Compute the x, y and depth vectors
        xul, yul = self.lonlat2xy(box[0], box[1])
        xbr, ybr = self.lonlat2xy(box[2], box[3])
        dmi = self.depth.min()
        dma = self.depth.max()
        XX = np.linspace(xul,xbr,Nlon)

        YY = np.linspace(yul,ybr,Nlat)
        ZZ = np.linspace(dmi,dma,Ndepth)
        XX, YY, ZZ = np.meshgrid(XX, YY, ZZ)
        XX = XX.flatten()
        YY = YY.flatten()
        ZZ = ZZ.flatten()

        # Griddata
        import scipy.interpolate as sciint
        oldpoints = np.array([self.x, self.y, self.depth]).T
        newpoints = np.array([XX, YY, ZZ]).T

        # Vp
        if self.Vp is not None:
            vp = sciint.griddata(oldpoints, self.Vp, newpoints, method=method)
            self.Vp = vp

        # Vs
        if self.Vs is not None:
            vs = sciint.griddata(oldpoints, self.Vs, newpoints, method=method)
            self.Vs = vs

        # Rho
        if self.Rho is not None:
            rho = sciint.griddata(oldpoints, self.Rho, newpoints, method=method)
            self.Rho = rho

        # Remove nans
        u = np.where(np.isfinite(self.Vp))
        self.Vp = self.Vp[u]
        if self.Vs is not None:
            self.Vs = self.Vs[u]
        if self.Rho is not None:
            self.Rho = self.Rho[u]

        # Coordinates
        self.x = XX[u]
        self.y = YY[u]
        self.depth = ZZ[u]

        # All done
        return

    def KeepDistanceFromFault(self, faults, distance):
        '''
        Keep on the point to a certain distance to the fault.
        Args:
            * faults    : Fault structure from verticalfault or list of structures.
            * distance  : Distance maximum.
        '''

        # Check class
        if faults.__class__ is not list:
            faults = [faults]

        # Build a list
        mll = []

        for f in faults:

            # Get the fault trace
            xf = f.xf
            yf = f.yf

            # Store the line
            mll.append(np.vstack((xf,yf)).T.tolist())

        # Build a multiline object
        Ml = geom.MultiLineString(mll)

        # Get the distances
        pl = np.vstack((self.x, self.y)).T.tolist()
        d = np.array([Ml.distance(geom.Point(p)) for p in pl])

        # Get the points that are close to the faults
        u = np.where(d<=distance)

        # Select those points
        self.x = self.x[u]
        self.y = self.y[u]
        self.depth = self.depth[u]
        if self.Vp is not None:
            self.Vp = self.Vp[u]
        if self.Vs is not None:
            self.Vs = self.Vs[u]
        if self.Rho is not None:
            self.Rho = self.Rho[u]

        # Put these in lon/lat
        self.lon, self.lat = self.xy2lonlat(self.x, self.y)

        # All done
        return

    def Vp2VsPoisson(self, poisson, data='vertical'):
        '''
        Computes the value of Vs from the values of Vp and a poisson's ratio.
        Args:
            * poisson       : Poisson's ratio.
            * data          : Which data to convert. Can be 'all' or 'vertical'.

            Vs = np.sqrt( (1-2nu)/(1-nu) * vp^2/2 )
        '''

        # Get data
        if data is 'all':
            d = self.Vp
            prof = self.depth
        elif data is 'vertical':
            d = self.VpVert
            std = self.StdVpVert
            prof = self.DVert
        elif data is 'model':
            d = self.SimVpVert
            prof = self.SimDepth

        # Set up the poisson vector
        if poisson.__class__ is float:
            poisson = np.ones((d.shape))*poisson
        elif poisson.__class__ is list:
            import copy
            save = copy.deepcopy(poisson)
            poisson = copy.deepcopy(prof)
            pmin = 0
            for p in save:
                u = np.where((prof<p[0]) & (prof>=pmin))
                poisson[u] = p[1]
                pmin = p[0]

        # Convert to Vs
        vs = np.sqrt( (1-2*poisson)/(1-poisson) * d**2 * 0.5 )
        if data is 'vertical':
            svs = np.sqrt( (1-2*poisson)/(1-poisson) * std**2 * 0.5 )

        # Stores Vs
        if data is 'all':
            self.Vs = vs
        elif data is 'vertical':
            self.VsVert = vs
            self.StdVsVert = svs
        elif data is 'model':
            self.SimVsVert = vs

        # All done
        return

    def setDensityProfile(self, density, std=None):
        '''
        Builds a density profile from std input.
        Args:   
            * density   : list of densities.
            * std       : list of standard deviations.
        '''

        # Create the vector
        self.RhoVert = density

        # Std
        if std is None:
            self.StdRhoVert = np.zeros((len(self.DVert,)))
        else:
            self.StdRhoVert = std

        # All done
        return

    def fitLayers(self, NLayers):
        '''
        Fit a Nlayer model on the 1D profile.
        Args:
            * NLayers       : Number of Layers.
        '''

        # Import
        import scipy.optimize as sciopt

        # Get the data
        if self.VpVert is not None:
            d1 = self.VpVert
        else:
            print ('No Vp profile, Abort...')
            return
        if self.VsVert is not None:
            d2 = self.VsVert
        else:
            print ('No Vs profile, Abort...')
            return

        # get the depth 
        prof = self.DVert

        # Build x0 and bounds
        x0 = np.zeros((3*NLayers + 2))
        bounds = []
        dstart = prof.max()/NLayers
        for i in range(NLayers):
            x0[3*i] = np.mean(d1)
            bounds.append((d1.min()-0.1, d1.max()+0.1))
            x0[3*i+1] = np.mean(d2)
            bounds.append((d2.min()-0.1, d2.max()+0.1))
            x0[3*i+2] = dstart
            bounds.append((prof.min()-0.1, prof.max()+0.1))
            dstart += prof.max()/NLayers
        x0[-2] = np.mean(d1)
        x0[-1] = np.mean(d2)
        bounds.append((d1.min()-0.1, d1.max()+0.1))
        bounds.append((d2.min()-0.1, d2.max()+0.1))

        # Minimize
        Results = sciopt.minimize(self.F, x0, args=(d1, d2, prof, NLayers), bounds=None, method='BFGS')

        # Get x
        x = Results['x']

        # Store the guys
        p = []
        d1 = []
        d2 = []
        for i in range(NLayers):
            p.append(x[3*i+2])
            d1.append(x[3*i])
            d2.append(x[3*i+1])

        # Store these
        self.SimDepth = np.array(p)
        self.SimVpVert = np.array(d1)
        self.SimVsVert = np.array(d2)

        # All done
        return

    def F(self, x, d1, d2, prof, NLayers):
        '''
        Forward model for layer estimations.
        '''

        dpred1 = np.zeros((prof.shape))
        dpred2 = np.zeros((prof.shape))
        pmin = 0
        for i in range(NLayers):
            v1 = x[3*i]
            v2 = x[3*i+1]
            p = x[3*i+2]
            u = np.where( (prof>=pmin) & (prof<p) )
            dpred1[u] = v1
            dpred2[u] = v2
            pmin = p

        v1 = x[-2]
        v2 = x[-1]
        u = np.where((prof>=pmin))
        dpred1[u] = v1
        dpred2[u] = v2

        return scilin.norm(d1 - dpred1 + d2 - dpred2)

    def setAverageVerticalModel(self, Vp, Vs, Rho, D, shear=None):
        '''
        Inputs an average velocity model in 1D.
        Args:
            * Vp        : Pwave velocity.
            * Vs        : Swave velocity.
            * Rho       : Density.
            * D         : Depth.
        '''

        self.SimDepth = np.array(D)
        self.SimVpVert = np.array(Vp)
        self.SimVsVert = np.array(Vs)
        self.SimRhoVert = np.array(Rho)
        if shear is not None:
            self.SimShearVert = np.array(shear)
        else:
            self.SimShearVert = None

        # All done
        return

    def VerticalAverage(self):
        '''
        Averages Vp, Vs and Rho along depth.
        '''

        # Get the depths
        depth = np.unique(self.depth)

        # Create storage
        Vp = []
        sVp = []
        Vs = []
        sVs = []
        Rho = []
        sRho = []

        # Average for each depth
        for d in depth:
            u = np.where(self.depth==d)
            if self.Vp is not None:
                v = self.Vp[u]
                v = v[np.isfinite(v)]
                Vp.append(np.mean(v))
                sVp.append(np.std(v))
            if self.Vs is not None:
                v = self.Vs[u]
                v = v[np.isfinite(v)]
                Vs.append(np.mean(v))
                sVs.append(np.std(v))
            if self.Rho is not None:
                r = self.Rho[u]
                r = r[np.isfinite(v)]
                Rho.append(np.mean(r))
                sRho.append(np.std(r))

        # Make arrays
        if self.Vp is not None:
            self.VpVert = np.array(Vp)
            self.StdVpVert = np.array(sVp)
        else:
            self.VpVert = None
        if self.Vs is not None:
            self.VsVert = np.array(Vs)
            self.StdVsVert = np.array(sVs)
        else:
            self.VsVert = None
        if self.Rho is not None:
            self.RhoVert = np.array(Rho)
            self.StdRhoVert = np.array(sRho)
        else:
            self.RhoVert = None
        self.DVert = depth

        # All done
        return

    def readVpVsRhoFromAsciiVertAve(self, infile, header=0, depthfact=1., allfact=1., readshear=False):
        '''
        Reads vertical profiles of Vp, Vs and Density from an ascii file.
        Format:
        DEPTH  DENSITY  DENSITYSTD  VS  VSSTD  VP  VPSTD (ShearMod ShearModStd)
        Args:
            * infile    : name of the input file
            * header    : Length of the header (default=0)
            * depthfact : Multiply depth 
        '''

        # Open the file
        fin = open(infile, 'r')

        # Read lines
        All = fin.readlines()
        All = All[header:]

        # Close file
        fin.close()

        # Create lists
        depths = []
        vp = []
        vpstd = []
        vs = []
        vsstd = []
        rho = []
        rhostd = []
        if readshear:
            shear = []
            shearstd = []

        # iterate and fill those in 
        for line in All:
            a = line.split()
            depths.append(np.float(a[0])*depthfact)
            rho.append(np.float(a[1]))
            rhostd.append(np.float(a[2]))
            vs.append(np.float(a[3]))
            vsstd.append(np.float(a[4]))
            vp.append(np.float(a[5]))
            vpstd.append(np.float(a[6]))
            if readshear:
                shear.append(np.float(a[7]))
                shearstd.append(np.float(a[8]))
    
        # Save those
        self.DVert = np.array(depths)
        self.VpVert = np.array(vp)*allfact
        self.StdVpVert = np.array(vpstd)*allfact
        self.VsVert = np.array(vs)*allfact
        self.StdVsVert = np.array(vsstd)*allfact
        self.RhoVert = np.array(rho)*allfact
        self.StdRhoVert = np.array(rhostd)*allfact
        self.ShearVert = np.array(shear)
        self.StdShearVert = np.array(shearstd)

        # All done
        return

    def WriteEDKSModelFile(self, filename):
        '''
        Writes an input file for computing Kernels with EDKS.
        Args:   
            * filename      : Name of the output file.
        '''

        # Get the depth profile
        d = self.SimDepth

        # Get the velocity profiles
        vp = self.SimVpVert
        vs = self.SimVsVert

        # Get the density profile
        r = self.SimRhoVert

        # number of layers
        Nlayers = d.shape[0]

        # open a file
        fout = open(filename, 'w')

        # Write the first line
        fout.write("{} 1000. \n".format(Nlayers))

        # Loop over the depths
        d0 = 0
        for i in range(Nlayers):
            if i < Nlayers - 1:
                string = " {:3.2f}  {:3.2f}  {:3.2f}  {:3.2f} \n".format(r[i], vp[i], vs[i], d[i]-d0)
            else:
                string = " {:3.2f}  {:3.2f}  {:3.2f}  0.00 \n".format(r[i], vp[i], vs[i])
            fout.write(string)
            # update d0
            d0 = d[i]

        # Close the file
        fout.close()

        # All Done
        return

    def readEDKSModelFile(self, filename):
        '''
        Reads the EDKS model file.
        Args:
            * filename      : Name of the input file.
        '''

        # Open the file
        fin = open(filename, 'r')

        # Readall
        Lines = fin.readlines()

        # Create the list
        vp = []
        vs = []
        rho = []
        depth = [0.]

        # Read
        for line in Lines[1:-1]:
            line = line.split()
            depth.append(depth[-1]+float(line[-1]))
            rho.append(float(line[0]))
            vp.append(float(line[1]))
            vs.append(float(line[2]))

        # Last one
        vp.append(float(Lines[-1].split()[1]))
        vs.append(float(Lines[-1].split()[2]))
        rho.append(float(Lines[-1].split()[0]))

        # Set things
        self.VpVert = vp
        self.VsVert = vs
        self.RhoVert = rho
        self.Dvert = depth

        # Close file
        fin.close()

        # All done
        return

    def getModelOnFault(self, fault):
        '''
        Returns the velocity model on each fault patch or tent.
        '''

        # Get the fault depths
        if fault.patchType is ('triangletent'):
            depths = np.array([tent[2] for tent in fault.tent])
        else:
            depths = np.array([center[2] for center in fault.getcenters()])

        # Create the lists 
        vp = []; vs = []; rho = []

        # models
        mDepths = np.vstack((self.Dvert[:-1], self.Dvert[1:])).T.flatten()
        mVp = np.vstack((self.VpVert[:-1], self.VpVert[1:])).T.flatten()
        iVp = interp.interp1d(mDepths, mVp)
        mVs = np.vstack((self.VsVert[:-1], self.VsVert[1:])).T.flatten()
        iVs = interp.interp1d(mDepths, mVs)
        mRho = np.vstack((self.RhoVert[:-1], self.RhoVert[1:])).T.flatten()
        iRho = interp.interp1d(mDepths, mRho)

        # Iterate over the depths
        for d in depths:
            vp.append(iVp(d))
            vs.append(iVs(d))
            rho.append(iRho(d))

        # All done
        return rho, vp, vs

    def plotVertical(self, figure=67, depth=50):
        '''
        Plots the average vertical values
        '''
        
        # Import
        import matplotlib.collections as col

        # Get the number of plots
        Np = 0
        title = []
        d = []
        s = []
        sim = []
        if self.VpVert is not None:
            Np += 1
            title.append('Vp (km/s)')
            d.append(self.VpVert)
            s.append(self.StdVpVert)
            sim.append(self.SimVpVert)
        if self.VsVert is not None:
            Np += 1
            title.append('Vs (km/s)')
            d.append(self.VsVert)
            s.append(self.StdVsVert)
            sim.append(self.SimVsVert)
        if self.RhoVert is not None:
            Np += 1
            title.append('Density (g/cm3)')
            d.append(self.RhoVert)
            s.append(self.StdRhoVert)
            sim.append(self.SimRhoVert)

        if self.ShearVert is not None:
            Np += 1
            title.append('Shear Modulus (Pa)')
            d.append(self.ShearVert)
            s.append(self.StdShearVert)
            sim.append(self.SimShearVert)

        # open figure
        fig = plt.figure(figure)
        plots = []
        for i in range(Np):
            plots.append(fig.add_subplot(1,Np,i+1))

        # Set the zaxis
        zticks = []
        zticklabels = []
        for z in np.linspace(0, depth, 5):
            zticks.append(-1.0*z)
            zticklabels.append(z)
        for i in range(Np):
            plots[i].set_ylim([-1.0*(depth + 5), 0])
            plots[i].set_yticks(zticks)
            plots[i].set_yticklabels(zticklabels)
        
        # Set the labels
        for i in range(Np):
            plots[i].set_xlabel(title[i])
            plots[i].xaxis.tick_top()
            plots[i].xaxis.set_label_position('top') 
            plots[i].set_ylabel('Depth (km)')

        # Plot the averaged values
        for i in range(Np):
            
            xpoly = []
            ypoly = []
            down = d[i]-s[i]
            up = d[i]+s[i]
            xpoly.append(down[0])
            ypoly.append(1.)
            for p in range(len(d[i])):
                xpoly.append(down[p])
                ypoly.append(-1.0*self.DVert[p])
            xpoly.append(down[-1])
            ypoly.append(-1.0*depth)
            xpoly.append(up[-1])
            ypoly.append(-1.0*depth)
            for p in range(len(d[i])-1,-1,-1):
                xpoly.append(up[p])
                ypoly.append(-1.0*self.DVert[p])
            xpoly.append(up[0])
            ypoly.append(1.)
            poly = [zip(xpoly, ypoly)]
            poly = col.PolyCollection(poly, facecolor='gray', edgecolor='black')
            plots[i].add_collection(poly)
            plots[i].plot(d[i], -1.0*self.DVert, '-k', linewidth=2)
            if sim[i] is not None:
                xu = []
                yu = []
                pp = 0
                for n in range(len(sim[i])):
                    xu.append(sim[i][n])
                    yu.append(pp)
                    xu.append(sim[i][n])
                    pp = -1.0*self.SimDepth[n]
                    yu.append(pp)
                plots[i].plot(xu, yu, '-r', linewidth=3)

        # Show it
        plt.show()

        # All done
        return

    def plot3D(self, fault=None, data='vp', figure=234, norm=None, markersize=5, depthmax=50):
        '''
        Plots the desired data set in 3D, using scatter 3D.
        Args:
            * fault     : Adds a fault trace at the surface (structure from vertical fault).
            * data      : Selects the data to plot ('vp', 'vs' or 'density').
            * figure    : Number of the figure.
            * norm      : Minimum and maximum for the color scale.
            * markersize: Size of the scattered dots.
            * depthmax  : Maximum depth.
        '''

        # opens a figure
        fig = plt.figure(figure)
        carte = fig.add_subplot(111,projection='3d')

        # Set the axes
        carte.set_xlabel('Easting (km)')
        carte.set_ylabel('Northing (km)')
        carte.set_zlabel('Depth (km)')

        # Get data
        if data is 'vp':
            d = self.Vp
        elif data is 'vs':
            d = self.Vs
        elif data is 'density':
            d = self.Rho

        # Remove the nans
        u = np.where(np.isfinite(d))

        # Color scale
        if norm is None:
            vmin = d[u].min()
            vmax = d[u].max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Plot the data
        sc = carte.scatter3D(self.x[u], self.y[u], -1*self.depth[u], s=markersize, c=d[u], vmin=vmin, vmax=vmax, linewidth=0.01)

        # Colorbar
        fig.colorbar(sc, shrink=0.6, orientation='h')

        # Plot the fault
        if fault is not None:
            if fault.__class__ is not list:
                fault = [fault]
            for f in fault:
                carte.plot(f.xf, f.yf, '-r')

        # Set the z-axis
        carte.set_zlim3d([-1.0*(depthmax), 0])
        zticks = []
        zticklabels = []
        for z in np.linspace(0,depthmax,5):
            zticks.append(-1.0*z)
            zticklabels.append(z)
        carte.set_zticks(zticks)
        carte.set_zticklabels(zticklabels)

        # Show
        plt.show()


#EOF
