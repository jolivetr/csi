'''
Class that plots the class verticalfault, gps and insar in 3D.

Written by R. Jolivet and Z. Duputel, April 2013.

Edited by T. Shreve, May 2019. Commented out lines 386, 391, 460, 485, 1121, 1131 because plotting fault patches was incorrect...
Added plotting option for pressure sources

July 2019: R Jolivet replaced basemap by cartopy.
'''

# Numerics
import numpy as np
import scipy.interpolate as sciint

# Os
import os, copy, sys

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as colls

# Cartopy 
import cartopy 
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

# CSI
from .SourceInv import SourceInv


class geodeticplot(object):
    '''
    A class to create plots of geodetic data with faults. Geographic representation is based on cartopy.
    Two figures are created. One with a map and the other with a 3D view of the fault.

    Args:
        * lonmin    : left-boundary of the map
        * lonmax    : Right-boundary of the map
        * latmin    : Bottom-boundary of the map
        * latmax    : Top of the map

    Kwargs:
        * figure        : Figure number
        * pbaspect      : ??
        * resolution    : Resolution of the mapped coastlines, lakes, rivers, etc. See cartopy for details
        * figsize       : tuple of the size of the 2 figures

    Returns:
        * None
    '''

    def __init__(self,lonmin, latmin, lonmax, latmax,
                 figure=None, pbaspect=None,resolution='auto',
                 figsize=[None,None]):
        #projection='cyl',

        # Save
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.latmin = latmin
        self.latmax = latmax

        # Lon0 lat0
        self.lon0 = lonmin + (lonmax-lonmin)/2.
        self.lat0 = latmin + (latmax-latmin)/2.

        # Projection
        #self.projection = ccrs.TransverseMercator(central_longitude=self.lon0, 
        #                                          central_latitude=self.lat0)
        self.projection = ccrs.PlateCarree()

        # Open a figure
        fig1 = plt.figure(figure, figsize=figsize[0])
        faille = fig1.add_subplot(111, projection='3d')
        if figure is None:
            nextFig = np.max(plt.get_fignums())+1
        else:
            nextFig=figure+1
        fig2  = plt.figure(nextFig, figsize=figsize[1])
        carte = fig2.add_subplot(111, projection=self.projection)
        carte.set_extent([self.lonmin, self.lonmax, self.latmin, self.latmax], crs=self.projection)

        # Set the axes
        faille.set_xlabel('Longitude')
        faille.set_ylabel('Latitude')
        faille.set_zlabel('Depth (km)')
        carte.set_xlabel('Longitude')
        carte.set_ylabel('Latitude')

        # store plots
        self.faille = faille
        self.fig1 = fig1
        self.carte  = carte
        self.fig2 = fig2

        # All done
        return

    def close(self, fig2close=['map', 'fault']):
        '''
        Closes all the figures

        Kwargs:
            * fig2close : a list of 'map' and 'fault'. By default, closes both figures

        Returns:
            * None
        '''

        # Check
        if type(fig2close) is not list:
            fig2close = [fig2close]

        # Figure 1
        if 'fault' in fig2close:
            plt.close(self.fig1)
        if 'map' in fig2close:
            plt.close(self.fig2)

        # All done
        return

    def show(self, mapaxis=None, triDaxis=None, showFig=['fault', 'map'], fitOnBox=True):
        '''
        Show to screen

        Kwargs:
            * mapaxis   : Specify the axis type for the map (see matplotlib)
            * triDaxis  : Specify the axis type for the 3D projection (see mpl_toolkits)
            * showFig   : List of plots to show on screen ('fault' and/or 'map')
            * fitOnBox  : If True, fits the horizontal axis to the one asked at initialization

        Returns:
            * None
        '''

        # Change axis of the map
        if mapaxis is not None:
            self.carte.axis(mapaxis)

        # Change the axis of the 3d projection
        if triDaxis is not None:
            self.faille.axis(triDaxis)

        # Fits the horizontal axis to the asked values
        if fitOnBox:
            if self.lonmin>180.:
                self.lonmin -= 360.
            if self.lonmax>180.:
                self.lonmax -= 360.
            self.carte.set_extent([self.lonmin, self.lonmax, self.latmin, self.latmax])
            self.faille.set_xlim(self.carte.get_xlim())
            self.faille.set_ylim(self.carte.get_ylim())

        # Delete figures
        if 'map' not in showFig:
            plt.close(self.fig2)
        if 'fault' not in showFig:
            plt.close(self.fig1)

        # Show
        plt.show()

        # All done
        return

    def savefig(self, prefix, mapaxis='equal', ftype='pdf', dpi=None, bbox_inches=None, 
                      triDaxis='auto', saveFig=['fault', 'map']):
        '''
        Save to file.

        Args:
            * prefix    : Prefix used for filenames

        Kwargs:
            * mapaxis       : 'equal' or 'auto'
            * ftype         : 'eps', 'pdf', 'png'
            * dpi           : whatever dot-per-inche you'd like
            * bbox_inches   : pdf details
            * triDaxis      : 3D axis scaling
            * saveFig       : Which figures to save

        Returns:
            * None
        '''

        # Change axis of the map
        if mapaxis is not None:
            self.carte.axis(mapaxis)

        # Change the axis of the 3d proj
        if triDaxis is not None:
            self.faille.axis(triDaxis)

        # Save
        if (ftype is 'png') and (dpi is not None) and (bbox_inches is not None):
            if 'fault' in saveFig:
                self.fig1.savefig('%s_fault.png' % (prefix),
                        dpi=dpi, bbox_inches=bbox_inches)
            if 'map' in saveFig:
                self.fig2.savefig('%s_map.png' % (prefix),
                        dpi=dpi, bbox_inches=bbox_inches)
        else:
            if 'fault' in saveFig:
                self.fig1.savefig('{}_fault.{}'.format(prefix, ftype))
            if 'map' in saveFig:
                self.fig2.savefig('{}_map.{}'.format(prefix, ftype))

        # All done
        return

    def clf(self):
        '''
        Clears the figures

        Returns:
            * None
        '''
        self.fig1.clf()
        self.fig2.clf()
        return

    def titlemap(self, titre):
        '''
        Sets the title of the map.

        Args:
            * titre : title of the map

        Returns:
            * None
        '''

        self.carte.set_title(titre, y=1.08)

        # All done
        return

    def titlefault(self, titre):
        '''
        Sets the title of the fault model.

        Args:
            * titre : title of the fault

        Returns:
            * None
        '''

        self.faille.set_title(titre, title=1.08)

        # All done
        return

    def set_view(self, elevation, azimuth):
        '''
        Sets azimuth and elevation angle for the 3D plot.

        Args:
            * elevation     : Point of view elevation angle
            * azimuth       : Point of view azimuth angle

        Returns:
            * None
        '''
        # Set angles
        self.faille.view_init(elevation,azimuth)

        #all done
        return

    def equalize3dAspect(self):
        """
        Make the 3D axes have equal aspect. Not working yet (maybe never).

        Returns:
            * None
        """

        xlim = self.faille.get_xlim3d()
        ylim = self.faille.get_ylim3d()
        zlim = self.faille.get_zlim3d()

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]

        x0 = 0.5 * (xlim[1] + xlim[0])
        y0 = 0.5 * (ylim[1] + ylim[0])
        z0 = 0.5 * (zlim[1] + zlim[0])

        max_range = 0.5 * np.array([x_range, y_range, z_range]).max()

        self.faille.set_xlim3d([x0-max_range, x0+max_range])
        self.faille.set_ylim3d([y0-max_range, y0+max_range])
        self.faille.set_zlim3d(zlim)

        self.fig1.set_size_inches((14,6))

        return

    def set_xymap(self, xlim, ylim):
        '''
        Sets the xlim and ylim on the map.

        Args:
            * xlim  : tuple of the ongitude limits
            * ylim  : tuple of the latitude limits

        Returns:
            * None
        '''

        self.carte.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])

        # All done
        return

    def drawCoastlines(self, color='k', linewidth=1.0, linestyle='solid',
            resolution='auto', drawLand=True, drawMapScale=None,
            parallels=4, meridians=4, drawOnFault=False, drawCountries=True,
            zorder=1):
        '''
        Draws the coast lines in the desired area.

        Kwargs:
            * color         : Color of lines
            * linewidth     : Width of lines
            * linestyle     : Style of lines
            * resolution    : Resolution of the coastline. Can be auto, intermediate, coarse, low, high
            * drawLand      : Fill the continents (True/False)
            * drawMapScale  : Draw a map scale (None or length in km)
            * drawCountries : Draw County boundaries?
            * parallels     : If int, number of parallels. If float, spacing in degrees between parallels. If np.array, array of parallels
            * meridians     : Number of meridians to draw or array of meridians
            * drawOnFault   : Draw on 3D fault as well
            * zorder        : matplotlib order of plotting

        Returns:
            * None
        '''

        # Draw continents?
        landcolor = None
        if drawLand: landcolor = 'lightgrey'

        
        # Resolution 
        if resolution == 'auto' or resolution == 'intermediate':
            resolution = '50m'
        elif resolution == 'coarse' or resolution == 'low':
            resolution = '110m'
        elif resolution == 'fine':
            resolution = '10m'
        else:
            assert False, 'Unknown resolution : {}'.format(resolution)

        # coastlines in cartopy are multipolygon objects. Polygon has exterior, which has xy
        self.coastlines = cfeature.GSHHSFeature(scale='auto', edgecolor=color, facecolor=landcolor, 
                                                linewidth=linewidth, linestyle=linestyle, zorder=zorder, alpha=0.6)

        ## MapScale
        if drawMapScale is not None:
            assert False, 'Cannot draw a map scale yet. To be implemented...'

        # Draw and get the line object
        self.carte.add_feature(self.coastlines)
        ##### NOT WORKING YET ####
        #if drawOnFault:
        #    segments = []
        #    if type(self.coastlines.geometries()) is not list:
        #        geoms = [self.coastlines.geometries()]
        #    else:
        #        geoms = self.coastlines.geometries()
        #    for geom in geoms:
        #        for poly in geom:
        #            x = np.array(poly.exterior.xy[0])
        #            y = np.array(poly.exterior.xy[1])
        #            segments.append(np.vstack((x,y,np.zeros(x.shape))).T)
        #    if len(segments)>0:
        #        cote = art3d.Line3DCollection(segments)
        #        cote.set_edgecolor(landcolor)
        #        cote.set_linestyle(linestyle)
        #        cote.set_linewidth(linewidth)
        #        self.faille.add_collection3d(cote)
        ##### NOT WORKING YET ####

        # Draw countries
        if drawCountries:
            self.countries = cfeature.NaturalEarthFeature(scale=resolution, category='cultural', name='admin_0_countries', 
                                                          linewidth=linewidth/2., edgecolor='k', facecolor='lightgray', 
                                                          alpha=0.6, zorder=zorder)
            self.carte.add_feature(self.countries)
            ##### NOT WORKING YET ####
            #if drawOnFault:
            #    segments = []
            #    if type(self.countries.geometries()) is not list:
            #        geoms = [self.countries.geometries()]
            #    else:
            #        geoms = self.countries.geometries()
            #    for geom in geoms:
            #        for poly in geom:
            #            x = np.array(poly.exterior.xy[0])
            #            y = np.array(poly.exterior.xy[1])
            #            segments.append(np.vstack((x,y,np.zeros(x.shape))).T)
            #    if len(segments)>0:
            #        border = art3d.Line3DCollection(segments)
            #        border.set_edgecolor(None)
            #        border.set_linestyle(linestyle)
            #        border.set_linewidth(linewidth/2.)
            #        self.faille.add_collection3d(border)
            ##### NOT WORKING YET ####

        # Parallels
        lmin,lmax = self.latmin, self.latmax
        if type(parallels) is int:
            parallels = np.linspace(lmin, lmax, parallels+1)
        elif type(parallels) is float:
            parallels = np.arange(lmin, lmax+parallels, parallels)
        parallels = np.round(parallels, decimals=2)

        # Meridians
        lmin,lmax = self.lonmin, self.lonmax
        if type(meridians) is int:
            meridians = np.linspace(lmin, lmax, meridians+1)
        elif type(meridians) is float:
            meridians = np.arange(lmin, lmax+meridians, meridians)
        meridians = np.round(meridians, decimals=2)

        # Draw them
        gl = self.carte.gridlines(color='gray', xlocs=meridians, ylocs=parallels, linestyle=(0, (1, 1)))

        #if drawOnFault and parDir!={}:
        #    segments = []
        #    colors = []
        #    linestyles = []
        #    linewidths = []
        #    for p in parDir:
        #        par = parDir[p][0][0]
        #        segments.append(np.hstack((par.get_path().vertices,np.zeros((par.get_path().vertices.shape[0],1)))))
        #        colors.append(par.get_color())
        #        linestyles.append(par.get_linestyle())
        #        linewidths.append(par.get_linewidth())
        #    parallel = art3d.Line3DCollection(segments, colors=colors, linestyles=linestyles, linewidths=linewidths)
        #    self.faille.add_collection3d(parallel)

        #merDir = self.carte.drawmeridians(meridians, labels=[0,0,1,0], linewidth=0.4, color='gray', zorder=zorder)
        #if drawOnFault and merDir!={}:
        #    segments = []
        #    colors = []
        #    linestyles = []
        #    linewidths = []
        #    for m in merDir:
        #        mer = merDir[m][0][0]
        #        segments.append(np.hstack((mer.get_path().vertices,np.zeros((mer.get_path().vertices.shape[0],1)))))
        #        colors.append(mer.get_color())
        #        linestyles.append(mer.get_linestyle())
        #        linewidths.append(mer.get_linewidth())
        #    meridian = art3d.Line3DCollection(segments, colors=colors, linestyles=linestyles, linewidths=linewidths)
        #    self.faille.add_collection3d(meridian)

        # Restore axis
        self.faille.set_xlim(self.carte.get_xlim())
        self.faille.set_ylim(self.carte.get_ylim())

        # All done
        return

    def faulttrace(self, fault, color='r', add=False, discretized=False, zorder=4):
        '''
        Plots a fault trace.

        Args:
            * fault         : Fault instance

        Kwargs:
            * color         : Color of the fault.
            * add           : plot the faults in fault.addfaults
            * discretized   : Plot the discretized fault
            * zorder        : matplotlib order of plotting

        Returns:
            * None
        '''

        # discretized?
        if discretized:
            lon = fault.loni
            lat = fault.lati
        else:
            lon = fault.lon
            lat = fault.lat
        # Plot the added faults before
        if add:
            for f in fault.addfaults:
                #f[0][f[0]<0.] += 360.
                self.carte.plot(f[0], f[1], '-k', zorder=zorder)
            for f in fault.addfaults:
                if self.faille_flag:
                    self.faille.plot(f[0], f[1], '-k')

        # Plot the surface trace
        #lon[lon<0] += 360.
        #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        self.faille.plot(lon, lat, '-{}'.format(color), linewidth=2)
        self.carte.plot(lon, lat, '-{}'.format(color), linewidth=2, zorder=2)

        # All done
        return

    def faultpatches(self, fault, slip='strikeslip', norm=None, colorbar=True,
                     plot_on_2d=False, revmap=False, linewidth=1.0, cmap='jet',
                     transparency=0.0, factor=1.0, zorder=3):
        '''
        Plot the fualt patches

        Args:
            * fault         : Fault instance

        Kwargs:
            * slip          : Can be 'strikeslip', 'dipslip', 'tensile', 'total' or 'coupling'
            * norm          : Limits for the colorbar.
            * colorbar      : if True, plots a colorbar.
            * plot_on_2d    : if True, adds the patches on the map.
            * revmap        : Revert the colormap
            * linewidth     : Width of the edges of the patches
            * cmap          : Colormap (any of the matplotlib ones)
            * transparency  : 1 - alpha
            * factor        : scale factor for fault slip values
            * zorder        : matplotlib order of plotting

        Returns:
            * None
        '''

        # Get slip
        if slip in ('strikeslip'):
            slip = fault.slip[:,0].copy()
        elif slip in ('dipslip'):
            slip = fault.slip[:,1].copy()
        elif slip in ('tensile'):
            slip = fault.slip[:,2].copy()
        elif slip in ('total'):
            slip = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        elif slip in ('coupling'):
            slip = fault.coupling.copy()
        else:
            print ("Unknown slip direction")
            return
        slip *= factor

        # norm
        if norm is None:
            vmin=slip.min()
            vmax=slip.max()
        else:
            vmin=norm[0]
            vmax=norm[1]

        # set z axis
        try:
            self.setzaxis(fault.depth+5., zticklabels=fault.z_patches)
        except:
            print('Warning: Depth cannot be determined automatically. Please set z-axis limit manually')

        # set color business
        if revmap:
            cmap = plt.get_cmap(cmap)
        else:
            cmap = plt.get_cmap(cmap)
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        Xs = np.array([])
        Ys = np.array([])
        for p in range(len(fault.patch)):
            ncorners = len(fault.patchll[0])
            x = []
            y = []
            z = []
            for i in range(ncorners):
                x.append(fault.patchll[p][i][0])
                y.append(fault.patchll[p][i][1])
                z.append(-1.0*fault.patchll[p][i][2])
            verts = []
            for xi,yi,zi in zip(x,y,z):
                #if xi<0.: xi += 360.
                verts.append((xi,yi,zi))
            rect = art3d.Poly3DCollection([verts])
            rect.set_facecolor(scalarMap.to_rgba(slip[p]))
            rect.set_edgecolors('gray')
            alpha = 1.0 - transparency
            if alpha<1.0:
                rect.set_alpha(alpha)
            rect.set_linewidth(linewidth)
            self.faille.add_collection3d(rect)

        # Reset x- and y-lims
        self.faille.set_xlim([self.lonmin,self.lonmax])
        self.faille.set_ylim([self.latmin,self.latmax])

        # If 2d.
        if plot_on_2d:
            for p, patch in zip(range(len(fault.patchll)), fault.patchll):
                x = []
                y = []
                for i in range(ncorners):
                    x.append(patch[i][0])
                    y.append(patch[i][1])
                verts = []
                for xi,yi in zip(x,y):
                    #if xi<0.: xi += 360.
                    verts.append((xi,yi))
                rect = colls.PolyCollection([verts])
                rect.set_facecolor(scalarMap.to_rgba(slip[p]))
                rect.set_edgecolors('gray')
                rect.set_linewidth(linewidth)
                rect.set_zorder(zorder)
                self.carte.add_collection(rect)

        # put up a colorbar
        if colorbar:
            scalarMap.set_array(slip)
            self.fphbar = self.fig1.colorbar(scalarMap, shrink=0.3, orientation='horizontal')

        # All done
        return

    def pressuresource(self, fault, delta='pressure', norm=None, colorbar=True, revmap=False, linewidth=1.0, cmap='jet',
                     transparency=0.0, factor=1.0, zorder=3, plot_on_2d=False):
        '''
        Plots a pressure source.

        Args:
            * fault         : Pressure source instance

        Kwargs:
            * delta         : Can be 'pressure' or 'volume'
            * norm          : Limits for the colorbar.
            * colorbar      : if True, plots a colorbar.
            * revmap        : Reverts the colormap
            * linewidth     : width of the edhe of the source
            * cmap          : matplotlib colormap
            * transparency  : 1 - alpha
            * plot_on_2d    : if True, adds the patches on the map.
            * factor        : scale factor for pressure values
            * zorder        : matplotlib plotting order

        Returns:
            * None
        '''

        # Get slip
        if delta is 'pressure':
            delta = fault.deltapressure
        elif delta is 'volume':
            delta = fault.deltavolume
        else:
            print ("Unknown slip direction")
            return
        delta *= factor

        # norm
        if norm is None:
            vmin=0
            vmax=delta
        else:
            vmin=norm[0]
            vmax=norm[1]

        # set z axis
        self.setzaxis(fault.ellipshape['z0']+5., zticklabels=None)

        # set color business
        if revmap:
            cmap = plt.get_cmap(cmap)
        else:
            cmap = plt.get_cmap(cmap)
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        Xs = np.array([])
        Ys = np.array([])

        #plot 3-d spheroid here

        #Radii -- assuming prolate spheroid (z-axis is the semi-major axis when dip = 90, y-axis is semi-major axis when strike = 0), in km
        rx, ry, rz = fault.ellipshape['a']*fault.ellipshape['A']/1000.,fault.ellipshape['a']/1000.,fault.ellipshape['a']*fault.ellipshape['A']/1000.
        #All spherical angles
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100        )

        #xyz coordinates for spherical angles
        ex = rx * np.outer(np.cos(u), np.sin(v))
        ey = ry * np.outer(np.sin(u), np.sin(v))
        ez = rz * np.outer(np.ones_like(u), np.cos(v))

        #strike and dip in radians, multiplied by -1 because we want clockwise rotations
        strike = -1*fault.ellipshape['strike']* np.pi / 180.
        dip = -1*fault.ellipshape['dip']* np.pi / 180.

        #??? NEEDS TO BE CONFIRMED ---(could use scipy.spatial.transformations, but would require an additional package installation)
        #This conforms to the dMODELS formulation of Yang
        #All of this should probably be double, triple-checked to make sure we are plotting the source correctly
        rotex = np.asarray([[1,0,0],[0,np.cos(dip),-np.sin(dip)],[0,np.sin(dip),np.cos(dip)]])
        #rotey = np.asarray([[np.cos(dip), 0., np.sin(dip)],[0., 1.,0.],[-np.sin(dip), 0, np.cos(dip)]])
        rotez = np.asarray([[np.cos(strike), -np.sin(strike), 0.],[np.sin(strike), np.cos(strike),0.],[0.,0.,1.]])
        #Rotate around y-axis (or x-axis???) first, then z-axis
        xyz = np.dot(np.array([ex,ey,ez]).T,np.dot(rotex,rotez))

        ex_ll,ey_ll = np.array(fault.xy2ll(xyz[:,:,0]+fault.xf,xyz[:,:,1]+fault.yf))
        ez_ll = xyz[:,:,2]-(fault.ellipshape['z0']/1000.)

        #x0 and y0 are in lat/lon, depth is negative and in km
        self.faille.plot_surface(ex_ll,ey_ll,ez_ll,color=scalarMap.to_rgba(delta))


        # Reset x- and y-lims
        self.faille.set_xlim([self.lonmin,self.lonmax])
        self.faille.set_ylim([self.latmin,self.latmax])

        # # If 2d. Just plots center for now, should plot ellipse projection onto surface
        if plot_on_2d:
             self.carte.scatter(fault.ellipshape['x0'],fault.ellipshape['y0'])

        # put up a colorbar
        if colorbar:
            scalarMap.set_array(delta)
            self.fphbar = self.fig1.colorbar(scalarMap, shrink=0.3, orientation='horizontal')

        # All done
        return

    def faultTents(self, fault,
                   slip='strikeslip', norm=None, colorbar=True,
                   method='scatter', cmap='jet', plot_on_2d=False,
                   revmap=False, factor=1.0, npoints=10,
                   xystrides=[100, 100], zorder=0,
                   vertIndex=False):
        '''
        Plot a fault with tents.

        Args:
            * fault         : TriangularTent fault instance

        Kwargs:
            * slip          : Can be 'strikeslip', 'dipslip', 'tensile', 'total' or 'coupling'
            * norm          : Limits for the colorbar.
            * colorbar      : if True, plots a colorbar.
            * method        : Can be 'scatter' (plots all the sub points as a colored dot) or 'surface' (interpolates a 3D surface which can be ugly...)
            * cmap          : matplotlib colormap
            * plot_on_2d    : if True, adds the patches on the map.
            * revmap        : Reverse the default colormap
            * factor        : Scale factor for fault slip values
            * npoints       : Number of subpoints per patch. This number is only indicative of the actual number of points that is picked out by the dropSourcesInPatch function of EDKS.py. It only matters to make the interpolation finer. Default value is generally alright.
            * xystrides     : If method is 'surface', then xystrides is going to be the number of points along x and along y used to interpolate the surface in 3D and its color.
            * vertIndex     : Writes the index of the vertices

        Returns:
            * None
        '''

        # Get slip
        if slip in ('strikeslip'):
            slip = fault.slip[:,0].copy()
        elif slip in ('dipslip'):
            slip = fault.slip[:,1].copy()
        elif slip in ('tensile'):
            slip = fault.slip[:,2].copy()
        elif slip in ('total'):
            slip = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
        elif slip in ('coupling'):
            slip = fault.coupling.copy()
        else:
            print ("Unknown slip direction")
            return
        slip *= factor

        # norm
        if norm is None:
            vmin=slip.min()
            vmax=slip.max()
        else:
            vmin=norm[0]
            vmax=norm[1]

        # set z axis
        self.setzaxis(fault.depth+5., zticklabels=fault.z_patches)

        # set color business
        if revmap:
            cmap = plt.get_cmap('{}_r'.format(cmap))
        else:
            cmap = plt.get_cmap(cmap)
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # Get the variables we need
        vertices = fault.Vertices.tolist()
        vertices_ll = fault.Vertices_ll.tolist()
        patches = fault.patch
        faces = fault.Faces

        # Plot the triangles
        for face in faces:
            verts = [vertices_ll[f] for f in face]
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            z = [-1.0*v[2] for v in verts]
            x.append(x[0]); y.append(y[0]); z.append(z[0])
            x = np.array(x); x[x<0.] += 360.
            self.faille.plot3D(x, y, z, '-', color='gray', linewidth=1)
            if plot_on_2d:
                self.carte.plot(x, y, '-', color='gray', linewidth=1, zorder=zorder)

        # Plot the color for slip
        # 1. Get the subpoints for each triangle
        from .EDKSmp import dropSourcesInPatches as Patches2Sources
        if hasattr(fault, 'edksSources') and not hasattr(fault, 'plotSources'):
            fault.plotSources = copy.deepcopy(fault.edksSources)
            fault.plotSources[1] /= 1e3
            fault.plotSources[2] /= 1e3
            fault.plotSources[3] /= 1e3
            fault.plotSources[4] /= 180./np.pi
            fault.plotSources[5] /= 180./np.pi
            fault.plotSources[6] /= 1e6
        if hasattr(fault, 'plotSources'):
            print('Using precomputed sources for plotting')
        else:
            fault.sourceNumber = npoints
            Ids, xs, ys, zs, strike, dip, Areas = Patches2Sources(fault, verbose=False)
            fault.plotSources = [Ids, xs, ys, zs, strike, dip, Areas]

        # Get uhem
        Ids = fault.plotSources[0]
        X = fault.plotSources[1]
        Y = fault.plotSources[2]
        Z = fault.plotSources[3]

        # 2. Interpolate the slip on each subsource
        Slip = fault._getSlipOnSubSources(Ids, X, Y, Z, slip)

        # Check Method:
        if method is 'surface':

            # Do some interpolation
            intpZ = sciint.LinearNDInterpolator(np.vstack((X, Y)).T, Z, fill_value=np.nan)
            intpC = sciint.LinearNDInterpolator(np.vstack((X, Y)).T, Slip, fill_value=np.nan)
            x = np.linspace(np.nanmin(X), np.nanmax(X), xystrides[0])
            y = np.linspace(np.nanmin(Y), np.nanmax(Y), xystrides[1])
            x,y = np.meshgrid(x,y)
            z = intpZ(np.vstack((x.flatten(), y.flatten())).T).reshape(x.shape)
            slip = intpC(np.vstack((x.flatten(), y.flatten())).T).reshape(x.shape)

            # Do the surface plot
            cols = np.empty(x.shape, dtype=tuple)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    cols[i,j] = scalarMap.to_rgba(slip[i,j])

            lon, lat = fault.xy2ll(x, y)
            lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
            self.faille.plot_surface(lon, lat, -1.0*z, facecolors=cols, rstride=1, cstride=1, antialiased=True, linewidth=0)

            # On 2D?
            if plot_on_2d:
                lon, lat = fault.xy2ll(X, Y)
                lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
                self.carte.scatter(lon, lat, c=Slip, cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax, zorder=zorder)

            # Color Bar
            if colorbar:
                scalarMap.set_array(slip)
                self.fphbar = self.fig1.colorbar(scalarMap, shrink=0.3, orientation='horizontal')

        elif method is 'scatter':
            # Do the scatter ploto
            lon, lat = fault.xy2ll(X, Y)
            lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
            cb = self.faille.scatter3D(lon, lat, zs=-1.0*Z, c=Slip, cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax)

            # On 2D?
            if plot_on_2d:
                self.carte.scatter(lon, lat, c=Slip, cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax, zorder=zorder)
                if vertIndex:
                    for ivert,vert in enumerate(fault.Vertices_ll):
                        x,y = self.carte(vert[0], vert[1])
                        if x<360.: x+= 360.
                        plt.annotate('{}'.format(ivert), xy=(x,y),
                                     xycoords='data', xytext=(x,y), textcoords='data')

            # put up a colorbar
            if colorbar:
                self.fphbar = self.fig1.colorbar(cb, shrink=0.3, orientation='horizontal')

        # All done
        return lon, lat, Z, Slip

    def setzaxis(self, depth, zticklabels=None):
        '''
        Set the z-axis.

        Args:
            * depth     : Maximum depth.

        Kwargs:
            * ztickslabel   : which labels to use

        Returns:
            * None
        '''

        self.faille.set_zlim3d([-1.0*(depth+5), 0])
        if zticklabels is None:
            zticks = []
            zticklabels = []
            for z in np.linspace(0,depth,5):
                zticks.append(-1.0*z)
                zticklabels.append(z)
        else:
            zticks = []
            for z in zticklabels:
                zticks.append(-1.0*z)
        self.faille.set_zticks(zticks)
        self.faille.set_zticklabels(zticklabels)

        # All done
        return

    def surfacestress(self, stress, component='normal', linewidth=0.0, norm=None, colorbar=True):
        '''
        Plots the stress on the map.

        Args:
            * stress        : Stressfield object.

        Kwargs:
            * component     : If string, can be normal, shearstrike, sheardip. If tuple or list, can be anything specifying the indexes of the Stress tensor.
            * linewidth     : option of scatter.
            * norm          : Scales the color bar.
            * colorbar      : if true, plots a colorbar

        Returns:
            * None
        '''

        # Get values
        if component.__class__ is str:
            if component in ('normal'):
                val = stress.Sigma
            elif component in ('shearstrike'):
                val = stress.TauStrike
            elif component in ('sheardip'):
                val = stress.TauDip
            else:
                print ('Unknown component of stress to plot')
                return
        else:
            val = stress.Stress[component[0], component[1]]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')

        # norm
        if norm is not None:
            vmin = norm[0]
            vmax = norm[1]
        else:
            vmin = val.min()
            vmax = val.max()

        # Plot
        lon = stress.lon
        lat = stress.lat
        lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        sc = self.carte.scatter(xloc, yloc, s=20, c=val, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth)

        # colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.3, orientation='horizontal')

        # All don
        return

    def gps(self, gps, data=['data'], color=['k'], scale=None, scale_units=None, legendscale=10., linewidths=.1, name=False, zorder=5):
        '''
        Args:
            * gps           : gps object from gps.

        Kwargs:
            * data          : List of things to plot. Can be any list of 'data', 'synth', 'res', 'strain', 'transformation'
            * color         : List of the colors of the gps velocity arrows. Must be the same size as data
            * scale         : Scales the arrows
            * scale_units   : 'width', 'height', 'dots' or 'inches'
            * legendscale   : Length of the scale.
            * linewidths    : Width of the arrows.
            * name          : Plot the name of the stations (True/False).
            * zorder        : Order of the matplotlib rendering

        Returns:
            * None
        '''

        # Assert
        if (type(data) is not list) and (type(data) is str):
            data = [data]
        if (type(color) is not list):
            color = [color]
        if len(color)==1 and len(data)>1:
            color = [color[0] for d in data]

        # Check
        if scale is None:
            assert len(data)==1, 'If multiple data are plotted, need to provide scale'

        # Get lon lat
        lon = gps.lon
        lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = gps.lat

        # Make the dictionary of the things to plot
        Data = {}
        for dtype,col in zip(data, color):
            if dtype is 'data':
                dName = '{} Data'.format(gps.name)
                Values = gps.vel_enu
            elif dtype is 'synth':
                dName = '{} Synth.'.format(gps.name)
                Values = gps.synth
            elif dtype is 'res':
                dName = '{} Res.'.format(gps.name)
                Values = gps.vel_enu - gps.synth
            elif dtype is 'strain':
                dName = '{} Strain'.format(gps.name)
                Values = gps.Strain
            elif dtype is 'transformation':
                dName = '{} Trans.'.format(gps.name)
                Values = gps.transformation
            else:
                assert False, 'Data name not recognized'
            Data[dName] = {}
            Data[dName]['Values'] = Values
            Data[dName]['Color'] = col

        # Plot these
        for dName in Data:
            values = Data[dName]['Values']
            c = Data[dName]['Color']
            p = self.carte.quiver(lon, lat,
                                  values[:,0], values[:,1],
                                  width=0.005, color=c,
                                  scale=scale, scale_units=scale_units,
                                  linewidths=linewidths, zorder=zorder)
            # TODO TODO TODO TODO TODO
            #if np.isfinite(self.err_enu[:,0]).all() and np.isfinite(self.err_enu[:,1]).all():
                # Extract the location of the arrow head
                # Create an ellipse of the good size at that location
                # Add it to collection, under the arrow
            # TODO TODO TODO TODO TODO

        # Plot Legend
        q = plt.quiverkey(p, 0.1, 0.1,
                          legendscale, '{}'.format(legendscale),
                          coordinates='axes', color='k', zorder=10)

        # Plot the name of the stations if asked
        if name:
            font = {'family' : 'serif',
                    'color'  : 'k',
                    'weight' : 'normal',
                    'size'   : 10}
            for lo, la, sta in zip(lon.tolist(), lat.tolist(), gps.station):
                # Do it twice, I don't know why text is screwed up...
                self.carte.text(lo, la, sta, zorder=20, fontdict=font)
                self.carte.text(lo-360., la, sta, zorder=20, fontdict=font)

        # All done
        return

    def gpsverticals(self, gps, norm=None, colorbar=True,
                     data=['data'], markersize=[10], linewidth=0.1,
                     zorder=4, cmap='jet', marker='o'):
        '''
        Scatter plot of the vertical displacement of the GPS.

        Args:
            * gps       : gps instance

        Kwargs:
            * norm          : limits of the colormap 
            * colorbar      : True/False
            * data          : list of 'data', 'synth', 'res'
            * markersize    : Size of the markers. List of the same size as data
            * linewidth     : width of the edge of the markers
            * cmap          : Colormap from matplotlib
            * marker        : type of marker
            * zorder        : plotting order in matplotlib

        Returns:
            * None
        '''

        # Assert
        if (type(data) is not list) and (type(data) is str):
            data = [data]
        if (type(markersize) is not list):
            markersize = [markersize]
        if len(markersize)==1 and len(data)>1:
            markersize = [markersize[0]*np.random.rand()*10. for d in data]

        # Get lon lat
        lon = gps.lon
        lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = gps.lat

        # Initiate
        vmin = 999999999.
        vmax = -999999999.

        # Make the dictionary of the things to plot
        from collections import OrderedDict
        Data = OrderedDict()
        for dtype,mark in zip(data, markersize):
            if dtype is 'data':
                dName = '{} Data'.format(gps.name)
                Values = gps.vel_enu[:,2]
            elif dtype is 'synth':
                dName = '{} Synth.'.format(gps.name)
                Values = gps.synth[:,2]
            elif dtype is 'res':
                dName = '{} Res.'.format(gps.name)
                Values = gps.vel_enu[:,2] - gps.synth[:,2]
            elif dtype is 'strain':
                dName = '{} Strain'.format(gps.name)
                Values = gps.Strain[:,2]
            elif dtype is 'transformation':
                dName = '{} Trans.'.format(gps.name)
                Values = gps.transformation[:,2]
            Data[dName] = {}
            Data[dName]['Values'] = Values
            Data[dName]['Markersize'] = mark
            vmin = np.min([vmin, np.min(Values)])
            vmax = np.max([vmax, np.max(Values)])

        # Get a colormap
        cmap = plt.get_cmap(cmap)

        # norm
        if norm is not None:
            vmin = norm[0]
            vmax = norm[1]

        # Plot that on the map
        for dName in Data:
            mark = Data[dName]['Markersize']
            V = Data[dName]['Values']
            sc = self.carte.scatter(lon, lat, s=mark, c=V, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth, zorder=zorder)

        # Colorbar
        if colorbar:
            cbar = self.fig2.colorbar(sc, orientation='horizontal', shrink=0.3)
            cbar.ax.tick_params(labelsize=4)

        return

    def gpsprojected(self, gps, norm=None, colorbar=True, zorder=4, cmap='jet'):
        '''
        Plot the gps data projected in the LOS

        Args:
            * gps       : gps instance

        Kwargs:
            * norm      : List of lower and upper bound of the colorbar.
            * colorbar  : activates the plotting of the colorbar.
            * cmap      : Colormap, by default 'jet'
            * zorder    : order of the plot in matplotlib

        Returns:
            * None
        '''

        # Get the data
        d = gps.vel_los
        lon = gps.lon;
        lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = gps.lat

        # Prepare the color limits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap(cmap)

        # Plot
        sc = self.carte.scatter(lon, lat, s=100, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, zorder=zorder)

        # plot colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.3, orientation='horizontal')

        # All done
        return

    def earthquakes(self, earthquakes, plot='2d3d', color='k', markersize=5, norm=None, colorbar=False, zorder=2):
        '''
        Plot earthquake catalog

        Args:
            * earthquakes   : object from seismic locations.

        Kwargs:
            * plot          : any combination of 2d and 3d.
            * color         : color of the earthquakes. Can be an array.
            * markersize    : size of each dots. Can be an array.
            * norm          : upper and lower bound for the color code.
            * colorbar      : Draw a colorbar.
            * zorder    : order of the plot in matplotlib

        Returns:
            * None
        '''

        # set vmin and vmax
        vmin = None
        vmax = None
        if (color.__class__ is np.ndarray):
            if norm is not None:
                vmin = norm[0]
                vmax = norm[1]
            else:
                vmin = color.min()
                vmax = color.max()
            import matplotlib.cm as cmx
            import matplotlib.colors as colors
            cmap = plt.get_cmap('jet')
            cNorm = colors.Normalize(vmin=color.min(), vmax=color.max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
            scalarMap.set_array(color)
        else:
            cmap = None

        # Get lon lat
        lon = earthquakes.lon
        lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = earthquakes.lat

        # plot the earthquakes on the map if ask
        if '2d' in plot:
            sc = self.carte.scatter(lon, lat, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.1, zorder=zorder)
            if colorbar:
                self.fig2.colorbar(sc, shrink=0.3, orientation='horizontal')

        # plot the earthquakes in the volume if ask
        if '3d' in plot:
            sc = self.faille.scatter3D(lon, lat, -1.*earthquakes.depth, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=0.1)
            if colorbar:
                self.fig1.colorbar(sc, shrink=0.3, orientation='horizontal')

        # All done
        return

    def faultsimulation(self, fault, norm=None, colorbar=True, direction='north'):
        '''
        Plot a fault simulation, Not tested in a long time... Might be obsolete.

        Args:
            * fault     : fault object 

        Kwargs:
            * norm      : List of lower and upper bound of the colorbar.
            * colorbar  : activates the plotting of the colorbar.
            * direction : which direction do we plot ('east', 'north', 'up', 'total' or a given azimuth in degrees, or an array to project in the LOS).

        Returns:
            * None
        '''

        if direction is 'east':
            d = fault.sim.vel_enu[:,0]
        elif direction is 'north':
            d = fault.sim.vel_enu[:,1]
        elif direction is 'up':
            d = fault.sim.vel_enu[:,2]
        elif direction is 'total':
            d = np.sqrt(fault.sim.vel_enu[:,0]**2 + fault.sim.vel_enu[:,1]**2 + fault.sim.vel_enu[:,2]**2)
        elif direction.__class__ is float:
            d = fault.sim.vel_enu[:,0]/np.sin(direction*np.pi/180.) + fault.sim.vel_enu[:,1]/np.cos(direction*np.pi/180.)
        elif direction.shape[0]==3:
            d = np.dot(fault.sim.vel_enu,direction)
        else:
            print ('Unknown direction')
            return

        # Prepare the color limits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')

        # Get lon lat
        lon = fault.sim.lon
        lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = fault.sim.lat

        # Plot the insar
        sc = self.carte.scatter(lon, lat, s=30, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.1)

        # plot colorbar
        if colorbar:
            self.fig2.colorbar(sc, shrink=0.3, orientation='horizontal')

        # All done
        return

    def insar(self, insar, norm=None, colorbar=True, data='data',
                       plotType='decimate',
                       decim=1, zorder=3, edgewidth=1):
        '''
        Plot an insar object

        Args:
            * insar     : insar object from insar.

        Kwargs:
            * norm      : lower and upper bound of the colorbar.
            * colorbar  : plot the colorbar (True/False).
            * data      : plot either 'data' or 'synth' or 'res'.
            * plotType  : Can be 'decimate' or 'scatter'
            * decim     : In case plotType='scatter', decimates the data by a factor decim.
            * edgewidth : width of the patches in case plotTtype is decimate
            * zorder    : order of the plot in matplotlib

        Returns:
            * None
        '''

        # Assert
        assert data in ('data', 'synth', 'res', 'poly'), 'Data type to plot unknown'
        # Choose data type
        if data == 'data':
            assert insar.vel is not None, 'No data to plot'
            d = insar.vel
        elif data == 'synth':
            assert insar.synth is not None, 'No Synthetics to plot'
            d = insar.synth
        elif data == 'res':
            assert insar.synth is not None and insar.vel is not None, \
                    'Cannot compute residuals'
            d = insar.vel - insar.synth
        elif data == 'poly':
            assert insar.orb is not None, 'No Orbital correction to plot'
            d = insar.orb
        else:
            print('Unknown data type')
            return

        # Prepare the colorlimits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        if plotType is 'decimate':
            for corner, disp in zip(insar.corner, d):
                x = []
                y = []
                # upper left
                x.append(corner[0])
                y.append(corner[1])
                # upper right
                x.append(corner[2])
                y.append(corner[1])
                # down right
                x.append(corner[2])
                y.append(corner[3])
                # down left
                x.append(corner[0])
                y.append(corner[3])
                verts = []
                for xi,yi in zip(x,y):
                    #if xi<0.: xi += 360.
                    verts.append((xi,yi))
                rect = colls.PolyCollection([verts],linewidth=edgewidth)
                rect.set_color(scalarMap.to_rgba(disp))
                rect.set_edgecolors('k')
                rect.set_zorder(zorder)
                self.carte.add_collection(rect)

        elif plotType is 'scatter':

            lon = insar.lon
            #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
            lat = insar.lat
            sc = self.carte.scatter(lon[::decim], lat[::decim], s=10,
                                    c=d[::decim], cmap=cmap, vmin=vmin, vmax=vmax,
                                    linewidth=0.0, zorder=zorder)

        else:
            print('Unknown plot type: {}'.format(plotType))
            return

        # plot colorbar
        if colorbar:
            scalarMap.set_array(d)
            plt.colorbar(scalarMap,shrink=0.3, orientation='horizontal')


        # All done
        return

    def opticorr(self, corr, norm=None, colorbar=True, data='dataEast',
                plotType='decimate', decim=1, zorder=3):
        '''
        Plot opticorr instance

        Args:
            * corr      : instance of the class opticorr

        Kwargs:
            * norm      : lower and upper bound of the colorbar.
            * colorbar  : plot the colorbar (True/False).
            * data      : plot either 'dataEast', 'dataNorth', 'synthNorth', 'synthEast', 'resEast', 'resNorth', 'data', 'synth' or 'res'
            * plotType  : plot either rectangular patches (decimate) or scatter (scatter)
            * decim     : decimation factor if plotType='scatter'
            * zorder    : order of the plot in matplotlib

        Returns:
            * None
        '''

        # Assert
        assert data in ('dataEast', 'dataNorth', 'synthEast', 'synthNorth', 'resEast', 'resNorth', 'data', 'synth', 'res'), 'Data type to plot unknown'

        # Choose the data
        if data == 'dataEast':
            d = corr.east
        elif data == 'dataNorth':
            d = corr.north
        elif data == 'synthEast':
            d = corr.east_synth
        elif data == 'synthNorth':
            d = corr.north_synth
        elif data == 'resEast':
            d = corr.east - corr.east_synth
        elif data == 'resNorth':
            d = corr.north - corr.north_synth
        elif data == 'data':
            d = np.sqrt(corr.east**2+corr.north**2)
        elif data == 'synth':
            d = np.sqrt(corr.east_synth**2 + corr.north_synth**2)
        elif data == 'res':
            d = np.sqrt( (corr.east - corr.east_synth)**2 + \
                    (corr.north - corr.north_synth)**2 )

        # Prepare the colorlimits
        if norm is None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        if plotType is 'decimate':
            for corner, disp in zip(corr.corner, d):
                x = []
                y = []
                # upper left
                x.append(corner[0])
                y.append(corner[1])
                # upper right
                x.append(corner[2])
                y.append(corner[1])
                # down right
                x.append(corner[2])
                y.append(corner[3])
                # down left
                x.append(corner[0])
                y.append(corner[3])
                verts = []
                for xi,yi in zip(x,y):
                    #if xi<0.: xi += 360.
                    verts.append((xi,yi))
                rect = colls.PolyCollection([verts])
                rect.set_color(scalarMap.to_rgba(disp))
                rect.set_edgecolors('k')
                rect.set_zorder(zorder)
                self.carte.add_collection(rect)

        elif plotType is 'scatter':
            lon = corr.lon
            lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
            lat = corr.lat
            self.carte.scatter(lon[::decim], lat[::decim], s=10., c=d[::decim], cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.0, zorder=zorder)

        else:
            assert False, 'unsupported plot type. Must be rect or scatter'

        # plot colorbar
        if colorbar:
            scalarMap.set_array(d)
            plt.colorbar(scalarMap, shrink=0.3, orientation='horizontal')

        # All done
        return

    def slipdirection(self, fault, linewidth=1., color='k', scale=1.):
        '''
        Plots segments representing the direction of slip along the fault.

        Args:
            * fault : Instance of a fault

        Kwargs:
            * linewidth : Width of the line
            * color     : Color of the line
            * scale     : Multiply slip 

        Returns:
            * None
        '''

        # Check utmzone
        assert self.utmzone==fault.utmzone, 'Fault object {} not in the same utmzone...'.format(fault.name)

        # Check if it exists
        if not hasattr(fault,'slipdirection'):
            fault.computeSlipDirection(scale=scale)

        # Loop on the vectors
        for v in fault.slipdirection:
            # Z increase downward
            v[0][2] *= -1.0
            v[1][2] *= -1.0
            # Make lists
            x, y, z = zip(v[0],v[1])
            # Plot
            self.faille.plot3D(x, y, z, color=color, linewidth=linewidth)

        # All done
        return

#EOF
