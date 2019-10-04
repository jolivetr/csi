'''
A class that deals with StressField data.

Written by R. Jolivet, Feb 2014.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
try:
    import h5py
except:
    print('No hdf5 capabilities detected')

# Personals
from .SourceInv import SourceInv
from . import okadafull as okada
from . import csiutils as utils

class stressfield(SourceInv):
    '''
    A class that handles a stress field. Not used in a long time, untested, could be incorrect.

    Args:
        * name          : Name of the StressField dataset.

    Kwargs:
        * utmzone       : UTM zone. Default is 10 (Western US).
        * lon0          : Longitude of the custom utmzone
        * lat0          : Latitude of the custom utmzone
        * ellps         : ellipsoid
        * verbose       : talk to me

    '''

    def __init__(self, name, utmzone=None, lon0=None, lat0=None, ellps='WGS84', verbose=True):

        # Base class init
        super(stressfield, self).__init__(name,
                                          utmzone=utmzone, 
                                          lon0=lon0, lat0=lat0,
                                          ellps=ellps)

        # Initialize the data set 
        self.name = name
        self.dtype = 'strainfield'

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize StressField data set {}".format(self.name))
        self.verbose=verbose

        # Initialize some things
        self.lon = None
        self.lat = None
        self.x = None
        self.y = None
        self.depth = None
        self.Stress = None
        self.trace = None

        # All done
        return

    def setXYZ(self, x, y, z):
        '''
        Sets the values of x, y and z.

        Args:
            * x     : array of floats (km)
            * y     : array of floats (km)
            * z     : array of floats (km)

        Returns:
            * None
        '''

        # Set
        self.x = x
        self.y = y
        self.depth = z

        # Set lon lat
        lon, lat = self.xy2ll(x, y)
        self.lon = lon
        self.lat = lat
    
        # All done
        return

    def setLonLatZ(self, lon, lat , z):
        '''
        Sets longitude, latitude and depth.

        Args:
            * lon     : array of floats (km)
            * lat     : array of floats (km)
            * z       : array of floats (km)

        Returns:
            * None
        '''

        # Set 
        self.lon = lon
        self.lat = lat
        self.depth = z

        # XY
        x, y = self.ll2xy(lon, lat)
        self.x = x
        self.y = y

        # All done
        return

    def Fault2Stress(self, fault, factor=0.001, mu=30e9, nu=0.25, slipdirection='sd', force_dip=None, stressonpatches=False, verbose=False):
        '''
        Takes a fault, or a list of faults, and computes the stress change associated with the slip on the fault.

        Args:   
            * fault             : Fault object (RectangularFault).

        Kwargs:
            * factor            : Conversion factor between the slip units and distance units. Usually, distances are in Km. Therefore, if slip is in mm, then factor=1e-6.
            * slipdirection     : any combination of s, d, and t.
            * mu                : Shear Modulus (default is 30GPa).
            * nu                : Poisson's ratio (default is 0.25).
            * stressonpatches   : Re-sets the station locations to be where the center of the patches are.
            * force_dip         : Specify the dip angle of the patches
            * verbos            : talk to me

        Returns:
            * None
        '''

        # Verbose?
        if verbose:
            print('Computing stress changes from fault {}'.format(fault.name))

        # Check if fault type corresponds
        assert len(fault.patch[0])==4, 'Fault is not made of rectangular patches'

        # Get a number
        nPatch = len(fault.patch)

        # Build Arrays
        xc = np.zeros((nPatch,))
        yc = np.zeros((nPatch,))
        zc = np.zeros((nPatch,))
        width = np.zeros((nPatch,))
        length = np.zeros((nPatch,)) 
        strike = np.zeros((nPatch,)) 
        dip = np.zeros((nPatch,))
        strikeslip = np.zeros((nPatch,))
        dipslip = np.zeros((nPatch,))
        tensileslip = np.zeros((nPatch,))

        # Build the arrays for okada
        for ii in range(len(fault.patch)):
            xc[ii], yc[ii], zc[ii], width[ii], length[ii], strike[ii], dip[ii] = fault.getpatchgeometry(fault.patch[ii], center=True)
            strikeslip[ii], dipslip[ii], tensileslip[ii] = fault.slip[ii,:]

        # Don't invert zc (for the patches, we give depths, so it has to be positive)
        # Apply the conversion factor
        strikeslip *= factor
        dipslip *= factor
        tensileslip *= factor

        # Set slips
        if 's' not in slipdirection:
            strikeslip[:] = 0.0
        if 'd' not in slipdirection:
            dipslip[:] = 0.0
        if 't' not in slipdirection:
            tensileslip[:] = 0.0

        # Get the stations
        if not stressonpatches:
            xs = self.x
            ys = self.y
            zs = -1.0*self.depth            # Okada wants the z position, we have the depth, so x-1.
        else:
            xs = xc
            ys = yc
            zs = -1.0*zc                    # Okada wants the z position, we have the depth, so x-1.

        # If force dip
        if force_dip is not None:
            dip[:] = force_dip

        # Get the Stress
        self.stresstype = 'total'
        self.Stress, flag, flag2 = okada.stress(xs, ys, zs, 
                                                xc, yc, zc, 
                                                width, length, 
                                                strike, dip,
                                                strikeslip, dipslip, tensileslip, 
                                                mu, nu, 
                                                full=True)

        self.flag = flag
        self.flag2 = flag2

        # All done
        return

    def computeTrace(self):
        '''
        Computes the Trace of the stress tensor.
        '''
        
        # Assert there is something to do
        assert (self.Stress is not None), 'There is no stress tensor...'

        # Check something
        if self.stresstype in ('deviatoric'):
            print('You should not compute the trace of the deviatoric tensor...')
            print('     Previous Trace value erased...')

        # Get it
        self.trace = np.trace(self.Stress, axis1=0, axis2=1)

        # All done
        return

    def total2deviatoric(self):
        '''
        Computes the deviatoric stress tensor dS = S - Tr(S)I
        '''

        # Check 
        if self.stresstype in ('deviatoric'):
            print('Stress tensor is already a deviatoric tensor...')
            return

        # Trace
        if self.trace is None:
            self.computeTrace()

        # Remove Trace
        self.Stress[0,0,:] -= self.trace
        self.Stress[1,1,:] -= self.trace
        self.Stress[2,2,:] -= self.trace

        # Change
        self.stresstype = 'deviatoric'

        # All done
        return

    def computeTractions(self, strike, dip):
        '''
        Computes the tractions given a plane with a given strike and dip.

        Args:
            * strike            : Strike (radians). 
            * dip               : Dip (radians).

        If these are floats, all the tensors will be projected on that plane. Otherwise, they need to be the size ofthe number of tensors.

        Positive Normal Traction means extension. Positive Shear Traction means left-lateral.
        '''

        # Get number of data points
        Np = self.Stress.shape[2]

        # Check the sizes
        strike = np.ones((Np,))*strike
        dip = np.ones((Np,))*dip

        # Create the normal vectors
        n1, n2, n3 = self.strikedip2normal(strike, dip)

        # Compute the stress vectors
        T = [ np.dot(n1[:,i], self.Stress[:,:,i]) for i in range(Np)]

        # Compute the Shear Stress and Normal Stress
        Sigma = np.array([ np.dot(T[i],n1[:,i]) for i in range(Np) ])
        TauStrike = np.array([ np.dot(T[i],n2[:,i]) for i in range(Np) ])
        TauDip = np.array([ np.dot(T[i],n3[:,i]) for i in range(Np) ])

        # All done
        return n1, n2, n3, T, Sigma, TauStrike, TauDip

    def getTractions(self, strike, dip):
        '''
        Just a wrapper around computeTractions to store the result, if necessary.

        Args:
            * strike            : Strike (radians). 
            * dip               : Dip (radians).

        If these are floats, all the tensors will be projected on that plane. Otherwise, they need to be the size ofthe number of tensors.

        Positive Normal Traction means extension. Positive Shear Traction means left-lateral.

        '''

        # Compute tractions
        n1, n2, n3, T, Sigma, TauStrike, TauDip = self.computeTractions(strike, dip)

        # Store everything
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.T = T
        self.Sigma = Sigma
        self.TauStrike = TauStrike
        self.TauDip = TauDip
        
        # All done
        return

    def strikedip2normal(self, strike, dip):
        '''
        Returns a vector normal to a plane with a given strike and dip (radians).

        Args:
            * strike    : strike angle in radians
            * dip       : dip angle in radians

        Returns:
            * tuple of unit vectors
        '''
        
        # Compute normal
        n1 = np.array([np.sin(dip)*np.cos(strike), -1.0*np.sin(dip)*np.sin(strike), np.cos(dip)])

        # Along Strike
        n2 = np.array([np.sin(strike), np.cos(strike), np.zeros(strike.shape)])

        # Along Dip
        n3 = np.cross(n1, n2, axisa=0, axisb=0).T

        # All done
        if len(n1.shape)==1:
            return n1.reshape((3,1)), n2.reshape((3,1)), n3.reshape((3,1))
        else:
            return n1, n2, n3

    def getprofile(self, name, loncenter, latcenter, length, azimuth, width, data='trace'):
        '''
        Project the wanted quantity onto a profile. Works on the lat/lon coordinates system.

        Args:
            * name              : Name of the profile.
            * loncenter         : Profile origin along longitude.
            * latcenter         : Profile origin along latitude.
            * length            : Length of profile.
            * azimuth           : Azimuth in degrees.
            * width             : Width of the profile.

        Kwargs:
            * data              : name of the data to use ('trace')

        Returns:
            * None
        '''

        print('Get the profile called {}'.format(name))

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}

        # Which value are we going to use
        try:
            val = self.__getattribute__(data)
        except:
            print('Keyword unknown. Please implement it...')
            return

        # Mask the data
        if hasattr(self, 'mask'):
            i = np.where(self.mask.value.flatten()==1)
            val[i] = np.nan

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.ll2xy(loncenter, latcenter)

        # Get the profile
        Dalong, Dacros, Bol, boxll, box, xe1, ye1, xe2, ye2, lon, lat = utils.coord2prof(self, xc, yc, length, azimuth, width)


        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name]
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['data'] = val[Bol]
        dic['Depth'] = self.depth[Bol]
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]

        # All done
        return

    def writeProfile2File(self, name, filename, fault=None):
        '''
        Writes the profile named 'name' to the ascii file filename.

        Args:
            * name      : name of the profile to work with
            * filename  : output file name

        Kwargs:
            * fault     : fualt object

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
            * data      : which data to plot
            * fault     : fault object
            * comp      : ??

        Returns:
            * None
        '''

        # open a figure
        fig = plt.figure()
        carte = fig.add_subplot(121)
        prof = fig.add_subplot(122)
        
        # Get the data we want to plot
        if data is 'trace':
            dplot = self.trace
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

        # plot the StressField Points on the Map
        carte.scatter(x, y, s=20, c=dplot, cmap=cmap, vmin=-1.0*MM, vmax=MM, linewidths=0.0)
        scalarMap.set_array(dplot)
        plt.colorbar(scalarMap)

        # plot the box on the map
        b = self.profiles[name]['Box']
        bb = np.zeros((5, 2))
        for i in range(4):
            x, y = self.ll2xy(b[i,0], b[i,1])
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

    def plot(self, data='trace', faults=None, gps=None, figure=123, ref='utm', legend=False, comp=None):
        '''
        Plot one component of the strain field.

        Kwargs:
            * data      : Type of data to plot. Can be 'trace'
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
        if data is 'trace':
            dplot = self.trace
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
        if hasattr(self, 'mask'):
            i = np.where(self.mask.value.flatten()==0)
            dplot = dplot[i]
            x = self.x.flatten()[i]
            y = self.y.flatten()[i]
            lon = self.lon.flatten()[i]
            lat = self.lat.flatten()[i]
        else:
            x = self.x.flatten()
            y = self.y.flatten()
            lon = self.lon.flatten()
            lat = self.lat.flatten()

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
            * fault     : fault object.

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
        xc, yc = self.ll2xy(lonc, latc)

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

        Kwargs:
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

        # Open the file
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
