'''
A class that deals with StrainField data.

Written by R. Jolivet, April 2013.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
try:
    import h5py
except:
    print('No hdf5 capabilities detected')

class strainfield(object):
    '''
    Class that handles a strain field. Has not been used in a long time... Might be incorrect and untested.

    Args:
        * name          : Name of the StrainField dataset.

    Kwargs:
        * utmzone       : UTM zone. Default is 10 (Western US).
        * lon0          : Longitude of the custom utmzone
        * lat0          : Latitude of the custom utmzone
        * ellps         : Ellipsoid
        * verbose       : Talk to me

    Returns:
        * None
    '''

    def __init__(self, name, utmzone=None, lon0=None, lat0=None, ellps='WGS84', verbose=True):

        # Base class init
        super(strainfield, self).__init__(name,
                                          utmzone=utmzone, 
                                          lon0=lon0, 
                                          lat0=lat0,
                                          ellps=ellps)

        # Initialize the data set 
        self.name = name
        self.dtype = 'strainfield'

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print (" Initialize StrainField data set {}".format(self.name))
        self.verbose=verbose

        # Initialize some things
        self.lon = None
        self.lat = None
        self.x = None
        self.y = None
        self.DeltaV = None
        self.vel_east = None
        self.vel_north = None
        self.mask = None

        # All done
        return

    def lonlat2xy(self, lon, lat):
        '''
        Uses the transformation in self to convert  lon/lat vector to x/y utm.

        Args:
            * lon           : Longitude array.
            * lat           : Latitude array.

        Returns:
            * None
        '''

        x, y = self.putm(lon,lat)
        x /= 1000.
        y /= 1000.

        return x, y

    def xy2lonlat(self, x, y):
        '''
        Uses the transformation in self to convert x.y vectors to lon/lat.

        Args:
            * x             : Xarray
            * y             : Yarray

        Returns:    
            * None
        '''

        lon, lat = self.putm(x*1000., y*1000., inverse=True)
        return lon, lat

    def read_from_h5(self, filename):
        '''
        Read the Continuous strain field from a hdf5 file.

        Args:
            * filename      : Name of the input file.

        Returns:
            * None
        '''

        # Open the file
        h5in = h5py.File(filename, 'r')

        # Get the sizes
        l = h5in['mask'].shape[0]
        w = h5in['mask'].shape[1]
        self.length = l
        self.width = w

        # Get the lon/lat and spacing
        lonBL = h5in['bottomLeftLon'].value
        latBL = h5in['bottomLeftLat'].value
        deltaLon = h5in['lonSpacing'].value
        deltaLat = h5in['latSpacing'].value

        # Get the longitude/latitude and build the arrays
        lon = np.linspace(lonBL, lonBL+w*deltaLon, w)
        lat = np.linspace(latBL, latBL+l*deltaLat, l)
        self.lon, self.lat = np.meshgrid(lon,lat)

        # Reshape lon lat and build x, y
        self.lon = self.lon.reshape((w*l,))
        self.lat = self.lat.reshape((w*l,))
        self.x, self.y = self.lonlat2xy(self.lon, self.lat)

        # Build the corners
        self.corners = [ [lonBL,latBL+l*deltaLat], [lonBL+w*deltaLon,latBL+l*deltaLat], 
                         [lonBL+w*deltaLon,latBL], [lonBL,latBL] ]
        self.deltaLon = deltaLon
        self.deltaLat = deltaLat

        # Get values
        self.DeltaV = h5in['velocityGradient']
        self.vel_east = h5in['veast']
        self.vel_north = h5in['vnorth']
        self.mask = h5in['mask']

        # Save the file
        self.hdf5 = h5in

        # All done
        return

    def closeHDF5(self):
        '''
        Closes the input hdf5 file.
        '''

        # Close it
        self.hdf5.close()

        # All done
        return

    def computeStrainRateTensor(self):
        '''
        Computes the strain rate tensor on each point of the grid and stores that in
        self.D. The strain rate tensor is the symmetric part of the velocity gradient. It writes self.D = 1/2 (L + L').
        '''
        
        # Print stuff
        print('Compute the Strain Rate Tensor')

        # initialize the strain tensor
        self.D = np.zeros((self.DeltaV.shape))

        # Loop over the pixels
        for j in range(self.D.shape[0]):

            # Get the velocity gradient
            L = self.DeltaV[j,:].reshape((3,3))

            # Compute the symmetric part
            d = 0.5 * (L+L.T)

            # store it in self.D
            self.D[j,:] = d.flatten()

        # All done
        return

    def computeRotationRateTensor(self):
        '''
        Computes the rotation rate tensor on each point of the grid and stores that in
        self.W. The rotation rate tensor is the anti-symmetric part of the velocity 
        gradient. It writes self.W = 1/2 (L - L').                                                     
        '''                                                                              
     
        # Print stuff
        print('Compute the Rotation Rate Tensor')

        # initialize the strain tensor
        self.W = np.zeros((self.DeltaV.shape))                                           
        
        # Loop over the pixels
        for j in range(self.D.shape[0]):                                                 
            
            # Get the velocity gradient
            L = self.DeltaV[j,:].reshape((3,3))                                               
            
            # Compute the symmetric part                                                 
            w = 0.5 * (L-L.T)                                                            
            
            # store it in self.D
            self.W[j,:] = w.flatten()
        
        # All done                                                                       
        return

    def computeDilatationRate(self):
        '''
        Computes the dilatation rate from the strain rate tensor.
        This is defined as the trace of the strain rate tensor.
        '''

        # Print stuff
        print('Compute the dilatation rate')

        # Compute the strain rate tensor
        if not hasattr(self, 'D'):
            self.computeStrainRateTensor()

        # Initialize it
        self.dilatation = np.zeros((self.DeltaV.shape[0],))

        # Loop over the pixels
        for j in range(self.dilatation.shape[0]):

            # Get the strain tensor
            D = self.D[j,:].reshape((3,3))

            # Get the trace
            self.dilatation[j] = np.trace(D)

        # All done 
        return

    def projectVelocities(self, name, angle):
        '''
        Projects the velocity field along a certain angle.
        The output is stored in the self.velproj dictionary and has a name

        Args:
            * name      : Name of the projected velocity field
            * angle     : azimuth of the projection 
        '''

        print('Project Velocities onto the direction {} degrees from North'.format(angle))

        # If the dictionary does not exist
        if not hasattr(self, 'velproj'):
            self.velproj = {}

        # Initialize
        self.velproj[name] = {}
        self.velproj[name]['Angle'] = angle
        self.velproj[name]['Projected Velocity'] = np.zeros((self.vel_east.shape))

        # Create the projection vector
        i = np.sin(angle*np.pi/180.0)
        j = np.cos(angle*np.pi/180.0)
        vec = np.array([i,j])

        # Loop on the velocities
        for i in range(self.vel_east.shape[0]):
            for j in range(self.vel_east.shape[1]):
                # Create the velocity vector
                vel = np.array([self.vel_east[i,j], self.vel_north[i,j]])
                # project
                self.velproj[name]['Projected Velocity'][i,j] = np.dot(vec, vel)

        # all done
        return

    def projectStrainRateTensor(self, name, angle):
        '''
        Projects the strain rate tensor onto a vector that has an angle 'angle'
        with the north. The unit vector is :

            V = [      0      ]
                [ -cos(angle) ]
                [  sin(angle) ]

        The projection is obtained by doing D.V on each grid point. We then get
        the scalar product of the projection with V.
        '''

        print('Project the Strain Rate Tensor onto the direction {} degrees from North'.format(angle))

        # Check if the strain rate tensor has been computed.
        if not hasattr(self, 'D'):
            self.computeStrainRateTensor()

        # Check if the dictionary for strain rate projection exists
        if not hasattr(self, 'Dproj'):
            self.Dproj = {}

        # Initialize 
        self.Dproj[name] = {}
        self.Dproj[name]['Angle'] = angle
        self.Dproj[name]['Projected Strain Rate'] = np.zeros((self.D.shape[0],))

        # Create the projection vector (in spherical coordinates, with theta 0 at the north pole, 180 at the south pole)
        V = np.zeros((3,))
        V[0] = 0
        V[1] = -1.0*np.sin(angle*np.pi/180.)
        V[2] = np.sin(angle*np.pi/180.)

        print('Vecteur : ( {} ; {} ; {} )'.format(V[0], V[1], V[2]))
    
        # Loop on the grid points
        for i in range(self.D.shape[0]):

            # Get the strain rate tensor at this grid point
            d = self.D[i,:].reshape((3,3))

            # Do the scalar product
            vp = np.dot( np.dot(d,V),V )

            # Store it in the projection
            self.Dproj[name]['Projected Strain Rate'][i] = vp

        # all done
        return
            
    def getprofile(self, name, loncenter, latcenter, length, azimuth, width, data='dilatation', comp=None):
        '''
        Project the wanted quantity onto a profile. Works on the lat/lon coordinates system.

        Args:
            * name              : Name of the profile.
            * loncenter         : Profile origin along longitude.
            * latcenter         : Profile origin along latitude.
            * length            : Length of profile.
            * azimuth           : Azimuth in degrees.
            * width             : Width of the profile
            
        Kwargs:
            * data              : name of the data to use ('dilatation', 'veast', 'vnorth', 'projection')
            * comp              : if data is 'projection', comp is the name of the desired projection.

        Returns:
            * None
        '''

        print('Get the profile called {}'.format(name))

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}

        # Which value are we going to use
        if data is 'veast':
            val = self.vel_east
        elif data is 'vnorth':
            val = self.vel_north
        elif data is 'dilatation':
            if not hasattr(self, 'dilatation'):
                self.computeDilatationRate()
            val = self.dilatation
        elif data is 'projection':
            val = self.velproj[comp]['Projected Velocity'].flatten()
        elif data is 'strainrateprojection':
            val = self.Dproj[comp]['Projected Strain Rate']
        else:
            print('Keyword unknown. Please implement it...')
            return

        # Mask the data
        i = np.where(self.mask.value.flatten()==1)
        val[i] = np.nan

        # Azimuth into radians
        alpha = azimuth*np.pi/180.

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.lonlat2xy(loncenter, latcenter)

        # Copmute the across points of the profile
        xa1 = xc - (width/2.)*np.cos(alpha)
        ya1 = yc + (width/2.)*np.sin(alpha)
        xa2 = xc + (width/2.)*np.cos(alpha)
        ya2 = yc - (width/2.)*np.sin(alpha)

        # Compute the endpoints of the profile
        xe1 = xc + (length/2.)*np.sin(alpha)
        ye1 = yc + (length/2.)*np.cos(alpha)
        xe2 = xc - (length/2.)*np.sin(alpha)
        ye2 = yc - (length/2.)*np.cos(alpha)

        # Convert the endpoints
        elon1, elat1 = self.xy2lonlat(xe1, ye1)
        elon2, elat2 = self.xy2lonlat(xe2, ye2)

        # Design a box in the UTM coordinate system.
        x1 = xe1 - (width/2.)*np.cos(alpha)
        y1 = ye1 + (width/2.)*np.sin(alpha)
        x2 = xe1 + (width/2.)*np.cos(alpha)
        y2 = ye1 - (width/2.)*np.sin(alpha)
        x3 = xe2 + (width/2.)*np.cos(alpha)
        y3 = ye2 - (width/2.)*np.sin(alpha)
        x4 = xe2 - (width/2.)*np.cos(alpha)
        y4 = ye2 + (width/2.)*np.sin(alpha)

        # Convert the box into lon/lat for further things
        lon1, lat1 = self.xy2lonlat(x1, y1)
        lon2, lat2 = self.xy2lonlat(x2, y2)
        lon3, lat3 = self.xy2lonlat(x3, y3)
        lon4, lat4 = self.xy2lonlat(x4, y4)

        # make the box 
        box = []
        box.append([x1, y1])
        box.append([x2, y2])
        box.append([x3, y3])
        box.append([x4, y4])

        # make latlon box
        boxll = []
        boxll.append([lon1, lat1])
        boxll.append([lon2, lat2])
        boxll.append([lon3, lat3])
        boxll.append([lon4, lat4])

        # Get the points in this box.
        # 1. import shapely and nxutils
        import shapely.geometry as geom
        import matplotlib.nxutils as mnu

        # 2. Create an array with the positions
        STRXY = np.vstack((self.x, self.y)).T

        # 3. Find those who are inside
        Bol = mnu.points_inside_poly(STRXY, box)

        # 4. Get these values
        xg = self.x[Bol]
        yg = self.y[Bol]
        val = val[Bol]

        # 5. Get the sign of the scalar product between the line and the point
        vec = np.array([xe1-xc, ye1-yc])
        sarxy = np.vstack((xg-xc, yg-yc)).T
        sign = np.sign(np.dot(sarxy, vec))

        # 6. Compute the distance (along, across profile) and get the velocity
        # Create the list that will hold these values
        Dacros = []; Dalong = []; 
        # Build lines of the profile
        Lalong = geom.LineString([[xe1, ye1], [xe2, ye2]])
        Lacros = geom.LineString([[xa1, ya1], [xa2, ya2]])
        # Build a multipoint
        PP = geom.MultiPoint(np.vstack((xg,yg)).T.tolist())
        # Loop on the points
        for p in range(len(PP.geoms)):
            Dalong.append(Lacros.distance(PP.geoms[p])*sign[p])
            Dacros.append(Lalong.distance(PP.geoms[p]))

        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name]
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['data'] = val
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]

        # All done
        return

    def writeProfile2File(self, name, filename, fault=None):
        '''
        Writes the profile named 'name' to the ascii file filename.

        Args:
            * name      : name of the profile to use
            * filename  : output file name

        Kwargs:
            * fault     : add a fault

        Returns:    
            * None
        '''

        # open a file
        fout = open(filename, 'w')

        # Get the dictionary
        dic = self.profiles[name]

        # Write the header
        fout.write('#---------------------------------------------------\n')
        fout.write('# Profile Generated with StaticInv\n')
        fout.write('# Center: {} {} \n'.format(dic['Center'][0], dic['Center'][1]))
        fout.write('# Endpoints: \n')
        fout.write('#           {} {} \n'.format(dic['EndPoints'][0][0], dic['EndPoints'][0][1]))
        fout.write('#           {} {} \n'.format(dic['EndPoints'][1][0], dic['EndPoints'][1][1]))
        fout.write('# Box Points: \n')
        fout.write('#           {} {} \n'.format(dic['Box'][0][0],dic['Box'][0][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][1][0],dic['Box'][1][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][2][0],dic['Box'][2][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][3][0],dic['Box'][3][1]))
        
        # Place faults in the header                                                     
        if fault is not None:
            if fault.__class__ is not list:                                             
                fault = [fault]
            fout.write('# Fault Positions: \n')                                          
            for f in fault:
                d = self.intersectProfileFault(name, f)
                fout.write('# {}           {} \n'.format(f.name, d))
        
        fout.write('#---------------------------------------------------\n')

        # Write the values
        for i in range(len(dic['Distance'])):
            d = dic['Distance'][i]
            Dp = dic['data'][i]
            if np.isfinite(Dp):
                fout.write('{} {} \n'.format(d, Dp))

        # Close the file
        fout.close()

        # all done
        return

    def plotprofile(self, name, data='veast', fault=None, comp=None):
        '''
        Plot profile.

        Args:
            * name      : Name of the profile.

        Kwargs:
            * data      : Which data to se
            * fault     : add a fault instance
            * comp      : ??

        Returns:
            * None
        '''

        # open a figure
        fig = plt.figure()
        carte = fig.add_subplot(121)
        prof = fig.add_subplot(122)
        
        # Get the data we want to plot
        if data is 'veast':
            dplot = self.vel_east.value.flatten()
        elif data is 'vnorth':
            dplot = self.vel_north.value.flatten()
        elif data is 'dilatation':
            if not hasattr(self, 'dilatation'):
                self.computeDilatationRate()
            dplot = self.dilatation
        elif data is 'projection':
            dplot = self.velproj[comp]['Projected Velocity'].flatten()
        elif data is 'strainrateprojection':
            dplot = self.Dproj[comp]['Projected Strain Rate']
        else:
            print('Keyword Unknown, please implement it....')
            return

        # Mask the data
        i = np.where(self.mask.value.flatten()==0)
        dplot = dplot[i]
        x = self.x.flatten()[i]
        y = self.y.flatten()[i]

        # Get min and max
        MM = np.abs(dplot).max()

        # Prepare a color map for insar
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        cmap = plt.get_cmap('seismic')
        cNorm = colors.Normalize(vmin=-1.0*MM, vmax=MM)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # plot the StrainField Points on the Map
        carte.scatter(x, y, s=20, c=dplot, cmap=cmap, vmin=-1.0*MM, vmax=MM, linewidths=0.0)
        scalarMap.set_array(dplot)
        plt.colorbar(scalarMap)

        # plot the box on the map
        b = self.profiles[name]['Box']
        bb = np.zeros((5, 2))
        for i in range(4):
            x, y = self.lonlat2xy(b[i,0], b[i,1])
            bb[i,0] = x
            bb[i,1] = y
        bb[4,0] = bb[0,0]
        bb[4,1] = bb[0,1]
        carte.plot(bb[:,0], bb[:,1], '.k')
        carte.plot(bb[:,0], bb[:,1], '-k')

        # plot the profile
        x = self.profiles[name]['Distance']
        y = self.profiles[name]['data']
        p = prof.plot(x, y, label=data, marker='.', linestyle='')

        # If a fault is here, plot it
        if fault is not None:
            # If there is only one fault
            if fault.__class__ is not list:
                fault = [fault]
            # Loop on the faults
            for f in fault:
                carte.plot(f.xf, f.yf, '-')
                # Get the distance
                d = self.intersectProfileFault(name, f)
                if d is not None:
                    ymin, ymax = prof.get_ylim()
                    prof.plot([d, d], [ymin, ymax], '--', label=f.name)

        # plot the legend
        prof.legend()

        # axis of the map
        carte.axis('equal')

        # Show to screen 
        plt.show()

        # All done
        return

    def plot(self, data='veast', faults=None, gps=None, figure=123, ref='utm', legend=False, comp=None):
        '''
        Plot one component of the strain field.

        Args:
            * data      : Type of data to plot. Can be 'dilatation', 'veast', 'vnorth'
            * faults    : list of faults to plot.
            * gps       : list of gps networks to plot.
            * figure    : figure number
            * ref       : utm or lonlat
            * legend    : add a legend
            * comp      : ??

        Returns:
            * None
        '''

        # Get the data we want to plot
        if data is 'veast':
            dplot = self.vel_east.value.flatten()
        elif data is 'vnorth':
            dplot = self.vel_north.value.flatten()
        elif data is 'dilatation':
            if not hasattr(self, 'dilatation'):
                self.computeDilatationRate()
            dplot = self.dilatation
        elif data is 'projection':
            dplot = self.velproj[comp]['Projected Velocity'].flatten()
        elif data is 'strainrateprojection':
            dplot = self.Dproj[comp]['Projected Strain Rate']
        else:
            print('Keyword Unknown, please implement...')
            return

        # Creates the figure
        fig = plt.figure(figure)
        ax = fig.add_subplot(111)

        # Set the axes
        if ref is 'utm':
            ax.set_xlabel('Easting (km)')
            ax.set_ylabel('Northing (km)')
        else:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

        # Mask the data
        i = np.where(self.mask.value.flatten()==0)
        dplot = dplot[i]
        x = self.x.flatten()[i]
        y = self.y.flatten()[i]
        lon = self.lon.flatten()[i]
        lat = self.lat.flatten()[i]

        # Get min and max
        MM = np.abs(dplot).max()

        # prepare a color map for the strain
        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        cmap = plt.get_cmap('seismic')
        cNorm  = colors.Normalize(vmin=-1.0*MM, vmax=MM)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # plot the wanted data
        if ref is 'utm':
            ax.scatter(x, y, s=20, c=dplot.flatten(), cmap=cmap, vmin=-1.0*MM, vmax=MM, linewidths=0.)
        else:
            ax.scatter(lon, lat, s=20, c=dplot.flatten(), cmap=cmap, vmin=-1.0*MM, vmax=MM, linewidths=0.)

        # Plot the surface fault trace if asked
        if faults is not None:
            if faults.__class__ is not list:
                faults = [faults]
            for fault in faults:
                if ref is 'utm':
                    ax.plot(fault.xf, fault.yf, '-b', label=fault.name)
                else:
                    ax.plot(fault.lon, fault.lat, '-b', label=fault.name)

        # Plot the gps if asked
        if gps is not None:
            if gps.__class__ is not list:
                gps = [gps]
            for g in gps:
                if ref is 'utm':
                        ax.quiver(g.x, g.y, g.vel_enu[:,0], g.vel_enu[:,1], label=g.name)
                else:
                        ax.quiver(g.lon, g.lat, g.vel_enu[:,0], g.vel_enu[:,1], label=g.name)

        # Legend
        if legend:
            ax.legend()

        # axis equal
        ax.axis('equal')

        # Colorbar
        scalarMap.set_array(dplot.flatten())
        plt.colorbar(scalarMap)

        # Show
        plt.show()

        # all done
        return

    def intersectProfileFault(self, name, fault):
        '''
        Gets the distance between the fault/profile intersection and the profile center.

        Args:
            * name      : name of the profile.
            * fault     : fault instance

        Returns:
            * None
        '''

        # Import shapely
        import shapely.geometry as geom

        # Grab the fault trace
        xf = fault.xf
        yf = fault.yf

        # Grab the profile
        prof = self.profiles[name]

        # import shapely
        import shapely.geometry as geom

        # Build a linestring with the profile center
        Lp = geom.LineString(prof['EndPoints'])

        # Build a linestring with the fault
        ff = []
        for i in range(len(xf)):
            ff.append([xf[i], yf[i]])
        Lf = geom.LineString(ff)

        # Get the intersection
        if Lp.crosses(Lf):
            Pi = Lp.intersection(Lf)
            p = Pi.coords[0]
        else:
            return None

        # Get the center
        lonc, latc = prof['Center']
        xc, yc = self.lonlat2xy(lonc, latc)

        # Get the sign 
        xa,ya = prof['EndPoints'][0]
        vec1 = [xa-xc, ya-yc]
        vec2 = [p[0]-xc, p[1]-yc]
        sign = np.sign(np.dot(vec1, vec2))

        # Compute the distance to the center
        d = np.sqrt( (xc-p[0])**2 + (yc-p[1])**2)*sign

        # All done
        return d

    def output2GRD(self, outfile, data='dilatation', comp=None):
        '''
        Output the desired field to a grd file.

        Args:
            * outfile       : Name of the outputgrd file.
            * data          : Type of data to output. Can be 'veast', 'vnorth', 'dilatation', 'projection', 'strainrateprojection'
            * comp          : if data is projection or 'strainrateprojection', give the name of the projection you want.

        Returns:
            * None
        '''

        # Get the data we want to plot
        if data is 'veast':
            dplot = self.vel_east.value
            units = 'mm/yr'
        elif data is 'vnorth':
            dplot = self.vel_north.value
            units = 'mm/yr'
        elif data is 'dilatation':
            if not hasattr(self, 'dilatation'):
                self.computeDilatationRate()
            dplot = self.dilatation.reshape((self.length, self.width))
            units = ' '
        elif data is 'projection':
            dplot = self.velproj[comp]['Projected Velocity']
            units = ' '
        elif data is 'strainrateprojection':
            dplot = self.Dproj[comp]['Projected Strain Rate'].reshape((self.length, self.width))
            units = ' '
        else:
            print('Keyword Unknown, please implement it....')
            return

        # Import netcdf
        import scipy.io.netcdf as netcdf
        fid = netcdf.netcdf_file(outfile,'w')

        # Create a dimension variable
        fid.createDimension('side',2)
        fid.createDimension('xysize',np.prod(z.shape))

        # Range variables
        fid.createVariable('x_range','d',('side',))
        fid.variables['x_range'].units = 'degrees'

        fid.createVariable('y_range','d',('side',))
        fid.variables['y_range'].units = 'degrees'

        fid.createVariable('z_range','d',('side',))
        fid.variables['z_range'].units = units

        # Spacing
        fid.createVariable('spacing','d',('side',))
        fid.createVariable('dimension','i4',('side',))

        fid.createVariable('z','d',('xysize',))
        fid.variables['z'].long_name = data
        fid.variables['z'].scale_factor = 1.0
        fid.variables['z'].add_offset = 0.0
        fid.variables['z'].node_offset=0

        # Fill the name
        fid.title = data
        fid.source = 'StaticInv.strainfield'

        # Filing 
        fid.variables['x_range'][0] = self.corners[0][0]
        fid.variables['x_range'][1] = self.corners[1][0]
        fid.variables['spacing'][0] = self.deltaLon

        fid.variables['y_range'][0] = self.corners[0][1]
        fid.variables['y_range'][1] = self.corners[3][1]
        fid.variables['spacing'][1] = -1.0*self.deltaLat

        #####Range
        zmin = np.nanmin(dplot)
        zmax = np.nanmax(dplot)

        fid.variables['z_range'][0] = zmin
        fid.variables['z_range'][1] = zmax

        fid.variables['dimension'][:] = z.shape[::-1]
        fid.variables['z'][:] = np.flipud(dplot).flatten()
        fid.sync()
        fid.close()

        return
#EOF
