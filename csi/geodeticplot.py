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
import scipy.ndimage as sciim

# Geography
import pyproj as pp

# Os
import os, copy, sys

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as colls
from matplotlib.patches import PathPatch
import matplotlib.patches as patches
import matplotlib.transforms as transforms

# Cartopy 
import cartopy 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import PostprocessedRasterSource, LocatedImage
from cartopy.io.srtm import SRTM1Source, SRTM3Source
from cartopy.io import srtm

# mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

# CSI
from .SourceInv import SourceInv
from .gebco import GebcoSource


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
                 figsize=[None,None], Map=True, Fault=True):

        # Save
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.latmin = latmin
        self.latmax = latmax

        # Lon0 lat0
        self.lon0 = lonmin + (lonmax-lonmin)/2.
        self.lat0 = latmin + (latmax-latmin)/2.

        # Projection
        self.projection = ccrs.PlateCarree()

        # Open a figure
        if Fault:
            figFaille = plt.figure(figure, figsize=figsize[0])
            faille = figFaille.add_subplot(111, projection='3d')
        
        # Chec figure number
        if figure == None:
            fignums = plt.get_fignums()
            if len(fignums)>0:
                nextFig = np.max(plt.get_fignums())+1
            else:
                nextFig = 1
        else:
            nextFig=figure+1

        # Open another one
        if Map:
            figCarte  = plt.figure(nextFig, figsize=figsize[1])
            carte = figCarte.add_subplot(111, projection=self.projection)
            carte.set_extent([self.lonmin, self.lonmax, self.latmin, self.latmax], crs=self.projection)

            # Gridlines (there is something wrong with the gridlines class...)
            gl = carte.gridlines(crs=self.projection, draw_labels=True, alpha=0.5, zorder=0)
            gl.xlabel_style = {'size': 'large', 'color': 'k', 'weight': 'bold'}
            gl.ylabel_style = {'size': 'large', 'color': 'k', 'weight': 'bold'}
            self.cartegl = gl
        #carte.set_xticks(carte.get_xticks())
        #carte.set_yticks(carte.get_yticks())
        #carte.tick_params(axis='both', which='major', labelsize='large')

        # Set the axes
        if Fault:
            faille.set_xlabel('Longitude')
            faille.set_ylabel('Latitude')
            faille.set_zlabel('Depth (km)')

        # store plots
        if Fault:
            self.faille = faille
            self.figFaille = figFaille
        else:
            self.faille = None
            self.figFaille = None
        if Map:
            self.carte  = carte
            self.figCarte = figCarte
        else:
            self.carte = None
            self.figCarte = None

        # All done
        return

    def drawScaleBar(self, scalebar, csiobj, lonlat=None, zorder=0, textoffset=10., linewidth=1, color='k', fontdict=None):
        '''
        Draw a {scalebar} km bar for scale at {lonlat}.
        '''

        # Check
        assert type(scalebar) == float, 'scalebar should be float: {}'.format(type(scalebar))

        # Chose where to put the bar
        if lonlat is not None:
            lonc, latc = lonlat
        else:
            lonc = self.lonmin + (self.lonmax-self.lonmin)/10.
            latc = self.latmin + (self.latmax-self.latmin)/10.

        # Convert to xy and build the end points
        xc,yc = csiobj.ll2xy(lonc,latc)

        # End points
        x1 = xc-scalebar/2.
        x2 = xc+scalebar/2.

        # Convert
        lon1,lat1 = csiobj.xy2ll(x1,yc)
        lon2,lat2 = csiobj.xy2ll(x2,yc)
        lonc,latc = csiobj.xy2ll(xc,yc+textoffset)

        # Show me
        self.carte.plot([lon1, lon2], [lat1,lat2], '-', color=color, 
                        linewidth=linewidth, zorder=zorder)
        self.carte.text(lonc,latc,'{} km'.format(scalebar), fontdict=fontdict,
                        horizontalalignment='center',zorder=zorder)

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
            plt.close(self.figFaille)
        if 'map' in fig2close:
            plt.close(self.figCarte)

        # All done
        return

    def show(self, mapaxis=None, triDaxis=None, showFig=['fault', 'map'], fitOnBox=False):
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
        if mapaxis != None and self.carte is not None:
            self.carte.axis(mapaxis)

        # Change the axis of the 3d projection
        if triDaxis != None and self.faille is not None:
            self.faille.axis(triDaxis)

        # Fits the horizontal axis to the asked values
        if fitOnBox:
            if self.lonmin>180.:
                self.lonmin -= 360.
            if self.lonmax>180.:
                self.lonmax -= 360.
            if self.carte is not None:  self.carte.set_extent([self.lonmin, self.lonmax, self.latmin, self.latmax])
            if self.faille is not None:
                self.faille.set_xlim(self.carte.get_xlim())
                self.faille.set_ylim(self.carte.get_ylim())

        # Delete figures
        if 'map' not in showFig:
            plt.close(self.figCarte)
        if 'fault' not in showFig:
            plt.close(self.figFaille)

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
        if mapaxis is not None and self.carte is not None:
            self.carte.axis(mapaxis)

        # Change the axis of the 3d proj
        if triDaxis is not None and self.faille is not None:
            self.faille.axis(triDaxis)

        # Save
        if (ftype == 'png') and (dpi is not None) and (bbox_inches is not None):
            if 'fault' in saveFig and self.faille is not None:
                self.figFaille.savefig('%s_fault.png' % (prefix),
                        dpi=dpi, bbox_inches=bbox_inches)
            if 'map' in saveFig and self.carte is not None:
                self.figCarte.savefig('%s_map.png' % (prefix),
                        dpi=dpi, bbox_inches=bbox_inches)
        else:
            if 'fault' in saveFig and self.faille is not None:
                self.figFaille.savefig('{}_fault.{}'.format(prefix, ftype))
            if 'map' in saveFig and self.carte is not None:
                self.figCarte.savefig('{}_map.{}'.format(prefix, ftype))

        # All done
        return

    def addColorbar(self, values, scalarMap, cbaxis, cborientation, figure, cblabel=''):
        '''
        Adds a colorbar for a given {scalarMap} to the chosen {figure}

        Args:
            * values        : Values used fr the plot
            * scalaMap      : Scalar Mappable from matplotlib
            * cbaxis        : [Left, Bottom, Width, Height] shape of the axis in which the colorbar is plot
            * cborientation : 'horizontal' or 'vertical'
            * figure        : Figure object

        Kwargs:
            * cblabel       : Label the colorbar
        '''

        scalarMap.set_array(values)
        cax = figure.add_axes(cbaxis)
        cb = plt.colorbar(scalarMap, cax=cax, orientation=cborientation)
        cb.set_label(label=cblabel, weight='bold')
        
        # Save this axis
        self.cbax = cb 

        # All done
        return

    def clf(self):
        '''
        Clears the figures

        Returns:
            * None
        '''
        self.figFaille.clf()
        self.figCarte.clf()
        return

    def titlemap(self, titre, y=1.1):
        '''
        Sets the title of the map.

        Args:
            * titre : title of the map

        Returns:
            * None
        '''

        self.carte.set_title(titre, y=y)

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

        if self.faille is None:
            print('No fault figure work on')
            return

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

    def set_view(self, elevation, azimuth, shape=(1., 1., 1.)):
        '''
        Sets azimuth and elevation angle for the 3D plot.

        Args:
            * elevation     : Point of view elevation angle
            * azimuth       : Point of view azimuth angle

        Kwargs:
            * shape         : [scale_x, scale_y, scale_z] (0-1 each)

        Returns:
            * None
        '''

        # Check 
        if self.faille is None:
            print('No Fault figure to work on')
            return

        # Set angles
        self.faille.view_init(elevation,azimuth)

        # Thank you stackoverflow
        self.faille.get_proj = lambda: np.dot(Axes3D.get_proj(self.faille), 
                np.diag([shape[0], shape[1], shape[2], 1]))

        #all done
        return

    def equalize3dAspect(self):
        """
        Make the 3D axes have equal aspect. Not working yet (maybe never).

        Returns:
            * None
        """

        # Check 
        if self.faille is None:
            print('No Fault figure to work on')
            return

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

        self.figFaille.set_size_inches((14,6))

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
        
        # check
        if self.carte is None: return

        self.carte.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])

        # All done
        return

    def shadedTopography(self, source='gebco', smooth=3, azimuth=140, altitude=45, 
                               alpha=1., zorder=1, cmap='Greys', norm=None,
                               gebcotype='sub_ice_topo', gebcoyear=2022,
                               srtmversion=1):
        '''
        if source == 'gebco':
            Plots the shaded topography Gebco.
            Needs user to download Gebco data and store that in ~/.local/share/cartopy/GEBCO
        if source == 'srtm':
            Plots the shaded topography from SRTM. Thanks to Thomas Lecocq.
            Needs user to download the SRTM tiles and unzip them beforehand (until the day cartopy guru's 
            manage to input a login and password to access directly SRTM data).
            Tiles must be stored at ~/.local/share/cartopy/SRTM/SRTMGL1 or in the directory given
            by the environment variable CARTOPY_DATA_DIR (which must be set before importing cartopy)

        Args:
            * smooth        : Smoothing factor in pixels of SRTM data (3 is nice)
            * azimuth       : Azimuth of the sun
            * elevation     : Sun elevation
            * alpha         : Alpha
            * srtmversion   : 1 or 3
            * gebcotype     : 'sub_ice_topo'
            * gebcoyear     : 2022

        Returns:
            * None
        '''

        # check
        if self.carte is None: return

        # Define toposhading (thanks Thomas Lecocq and the Cartopy website)
        def shade(located_elevations):
            """
            Given an array of elevations in a LocatedImage, add a relief (shadows) to
            give a realistic 3d appearance.
        
            """
            new_img = srtm.add_shading(sciim.gaussian_filter(located_elevations.image, smooth),
                                       azimuth=azimuth, altitude=altitude)
            return LocatedImage(new_img, located_elevations.extent)

        if source == 'srtm':

            # Version
            if srtmversion == 1:
                source = SRTM1Source(max_nx=8, max_ny=8)
            elif srtmversion == 3:
                source = SRTM3Source(max_nx=8, max_ny=8)

        elif source == 'gebco':

            # Get it
            source = GebcoSource(dtype=gebcotype, year=gebcoyear)

        # Build the raster
        if smooth>0.:
            shaded_topo = PostprocessedRasterSource(source, shade)
        else:
            shaded_topo = source

        # Add the raster
        self.carte.add_raster(shaded_topo, cmap=cmap, alpha=alpha, zorder=zorder, clim=norm)

        # All done
        return

    def drawCoastlines(self, color='k', linewidth=1.0, linestyle='solid',
            resolution='10m', landcolor='lightgrey', seacolor=None, drawMapScale=None,
            parallels=None, meridians=None, drawOnFault=False, 
            alpha=0.5, zorder=1):
        '''
        Draws the coast lines in the desired area.

        Kwargs:
            * color         : Color of lines
            * linewidth     : Width of lines
            * linestyle     : Style of lines
            * resolution    : Resolution of the coastline. Can be 10m, 50m or 110m
            * drawLand      : Fill the continents (True/False)
            * drawMapScale  : Draw a map scale (None or length in km)
            * parallels     : If int, number of parallels. If float, spacing in degrees between parallels. If np.array, array of parallels
            * meridians     : Number of meridians to draw or array of meridians
            * drawOnFault   : Draw on 3D fault as well
            * zorder        : matplotlib order of plotting

        Returns:
            * None
        '''

        # check
        if self.carte is None: return

        # Scale bar
        if drawMapScale is not None:
            self.drawScaleBar(drawMapScale, lonlat=None)

        # Ocean color (not really nice since this colors everything in the background)
        if seacolor=='image':
            self.carte.stock_img()
        else:
            if seacolor is not None: 
                self.carte.add_feature(cfeature.NaturalEarthFeature('physical', 
                                                                    'ocean', 
                                                                    scale=resolution,
                                                                    edgecolor=color, 
                                                                    facecolor=seacolor, 
                                                                    zorder=np.max([zorder-1,0])))

        # coastlines in cartopy are multipolygon objects. Polygon has exterior, which has xy
        self.coastlines = cfeature.NaturalEarthFeature('physical', 'land', scale=resolution,
                                                edgecolor=color, 
                                                facecolor=landcolor, 
                                                linewidth=linewidth, 
                                                linestyle=linestyle, 
                                                zorder=zorder, alpha=alpha)

        # Draw and get the line object
        self.carte.add_feature(self.coastlines)

        ## MapScale
        if drawMapScale is not None:
            assert False, 'Cannot draw a map scale yet. To be implemented...'

        # Parallels
        if parallels is not None:
            lmin,lmax = self.latmin, self.latmax
            if type(parallels) is int:
                parallels = np.linspace(lmin, lmax, parallels+1)
            elif type(parallels) is float:
                parallels = np.arange(lmin, lmax+parallels, parallels)
            parallels = np.round(parallels, decimals=2)

        # Meridians
        if meridians is not None:
            lmin,lmax = self.lonmin, self.lonmax
            if type(meridians) is int:
                meridians = np.linspace(lmin, lmax, meridians+1)
            elif type(meridians) is float:
                meridians = np.arange(lmin, lmax+meridians, meridians)
            meridians = np.round(meridians, decimals=2)

            # Draw them
        if meridians is not None and parallels is not None:
            gl = self.carte.gridlines(color='gray', xlocs=meridians, ylocs=parallels, linestyle=(0, (1, 1)))

        # All done
        return

    def drawCountries(self, resolution='10m', linewidth=1., edgecolor='gray', facecolor='lightgray', alpha=1., zorder=0):
        '''
        See the cartopy manual for options
        '''

        # Check
        if self.carte is not None:
            self.countries = cfeature.NaturalEarthFeature(scale=resolution, category='cultural', name='admin_0_countries', 
                                                          linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor,
                                                          alpha=alpha, zorder=zorder)
            self.carte.add_feature(self.countries)
        
        # All done
        return


    def faulttrace(self, fault, color='r', add=False, discretized=False, linewidth=1, zorder=1):
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
                self.carte.plot(f[0], f[1], '-k', zorder=zorder, linewidth=linewidth)
            for f in fault.addfaults:
                if self.faille_flag:
                    self.faille.plot(f[0], f[1], '-k', linewidth=linewidth)

        # Plot the surface trace
        #lon[lon<0] += 360.
        #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        if hasattr(fault, 'color'): color = fault.color
        if hasattr(fault, 'linewidth'): linewidth = fault.linewidth
        if self.faille is not None: self.faille.plot(lon, lat, '-{}'.format(color), linewidth=linewidth)
        if self.carte is not None: self.carte.plot(lon, lat, '-{}'.format(color), zorder=zorder, 
                                                   linewidth=linewidth)

        # All done
        return

    def faultpatches(self, fault, slip='strikeslip', norm=None, colorbar=True,
                     cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='',
                     plot_on_2d=False, revmap=False, linewidth=1.0, cmap='jet', offset=None,
                     alpha=1.0, factor=1.0, zorder=3, edgecolor='slip', colorscale='normal',stdfault=None):
        '''
        Plot the fualt patches

        Args:
            * fault         : Fault instance

        Kwargs:
            * slip          : Can be 'strikeslip', 'dipslip', 'tensile', 'total' or 'coupling'
            * norm          : Limits for the colorbar.
            * colorbar      : if True, plots a colorbar.
            * cbaxis        : [Left, Bottom, Width, Height] of the colorbar axis
            * cborientation : 'horizontal' (default) 
            * cblabel       : Write something next to the colorbar.
            * plot_on_2d    : if True, adds the patches on the map.
            * revmap        : Revert the colormap
            * linewidth     : Width of the edges of the patches
            * cmap          : Colormap (any of the matplotlib ones)
            * factor        : scale factor for fault slip values
            * zorder        : matplotlib order of plotting
            * edgecolor     : either a color or 'slip'
            * colorscale    : 'normal' or 'log'
            * alpha         : constant transparency value
            * stdfault      : to plot slip with transparency varying as a function of the std/mean relation for each patch. To use with positive values of slip and colorbars that are lighter near zero.
        Returns:
            * None
        '''

# Get slip
        if slip in ('strikeslip'):
            slip = fault.slip[:,0].copy()
            if stdfault!=None:
                stdslip = stdfault.slip[:,0].copy()
        elif slip in ('dipslip'):
            slip = fault.slip[:,1].copy()
            if stdfault!=None:
                stdslip = stdfault.slip[:,1].copy()
        elif slip in ('tensile'):
            slip = fault.slip[:,2].copy()
            if stdfault!=None:
                stdslip = stdfault.slip[:,2].copy()
        elif slip in ('total'):
            slip = np.sqrt(fault.slip[:,0]**2 + fault.slip[:,1]**2 + fault.slip[:,2]**2)
            if stdfault!=None:
                stdslip = np.sqrt(stdfault.slip[:,0]**2 + stdfault.slip[:,1]**2 + stdfault.slip[:,2]**2)
        else:
            print ("Unknown slip direction")
            return
        slip *= factor

        # norm
        if norm == None:
            vmin=slip.min()
            vmax=slip.max()
        else:
            vmin=norm[0]
            vmax=norm[1]

        # Potential offset of the fault wrt. its true position 
        if offset is None:
            offset = [0., 0., 0.]

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
        if colorscale in ('normal', 'n'):
            cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        elif colorscale in ('log', 'l', 'lognormal'):
            cNorm = colors.LogNorm(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        # fault figure
        if self.faille is not None:
            Xs = np.array([])
            Ys = np.array([])
            for p in range(len(fault.patch)):
                ncorners = len(fault.patchll[0])
                x = []
                y = []
                z = []
                for i in range(ncorners):
                    x.append(fault.patchll[p][i][0]+offset[0])
                    y.append(fault.patchll[p][i][1]+offset[1])
                    z.append(-1.0*fault.patchll[p][i][2]+offset[2])
                verts = []
                for xi,yi,zi in zip(x,y,z):
                    #if xi<0.: xi += 360.
                    verts.append((xi,yi,zi))
                rect = art3d.Poly3DCollection([verts])
                rect.set_facecolor(scalarMap.to_rgba(slip[p]))
                if edgecolor=='slip': 
                    rect.set_edgecolors(scalarMap.to_rgba(slip[p]))
                else:
                    rect.set_edgecolors(edgecolor)
                if alpha<1.0:
                    rect.set_alpha(alpha)
                if (stdfault!=None):
                    if (stdslip[p]>0) and (slip[p]!=0):
                        rect.set_alpha(np.min([1.,np.max([slip[p],0.])/stdslip[p]]))
                rect.set_linewidth(linewidth)
                self.faille.add_collection3d(rect)

            # Reset x- and y-lims
            self.faille.set_xlim([self.lonmin,self.lonmax])
            self.faille.set_ylim([self.latmin,self.latmax])

        # If 2d.
        if plot_on_2d and self.carte is not None:
            for p, patch in zip(range(len(fault.patchll)), fault.patchll):
                x = []
                y = []
                ncorners = len(fault.patchll[0])
                for i in range(ncorners):
                    x.append(patch[i][0]+offset[0])
                    y.append(patch[i][1]+offset[1])
                verts = []
                for xi,yi in zip(x,y):
                    #if xi<0.: xi += 360.
                    verts.append((xi,yi))
                rect = colls.PolyCollection([verts])
                rect.set_facecolor(scalarMap.to_rgba(slip[p]))
                if edgecolor=='slip': 
                    rect.set_edgecolors(scalarMap.to_rgba(slip[p]))
                else:
                    rect.set_edgecolors(edgecolor)
                rect.set_linewidth(linewidth)
                if stdfault!=None:
                    if (stdslip[p]>0) and (slip[p]!=0):
                        rect.set_alpha(np.min([1.,np.max([slip[p],0.])/stdslip[p]]))
                if alpha<1.0:
                    rect.set_alpha(alpha)
                rect.set_zorder(zorder)
                self.carte.add_collection(rect)

        # put up a colorbar
        if colorbar:
            if self.faille is not None: self.addColorbar(slip, scalarMap, cbaxis, cborientation, self.figFaille, cblabel=cblabel) 
            if plot_on_2d and self.carte is not None: 
                self.addColorbar(slip, scalarMap, cbaxis, cborientation,self.figCarte, cblabel=cblabel)

        # All done
        return

    def pressuresource(self, fault, delta='pressure', norm=None, colorbar=True, 
                     cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='',
                     revmap=False, linewidth=1.0, cmap='jet',
                     alpha=1.0, factor=1.0, zorder=3, plot_on_2d=False):
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
            * plot_on_2d    : if True, adds the patches on the map.
            * factor        : scale factor for pressure values
            * zorder        : matplotlib plotting order

        Returns:
            * None
        '''

        # Get slip
        if delta == 'pressure':
            delta = fault.deltapressure
        elif delta == 'volume':
            delta = fault.deltavolume
        else:
            print ("Unknown slip direction")
            return
        delta *= factor

        # norm
        if norm == None:
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
        rx, ry, rz = fault.ellipshape['ax']/1000.,fault.ellipshape['ay']/1000.,fault.ellipshape['az']/1000.
        #All spherical angles
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

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

        if self.faille is not None:
            #x0 and y0 are in lat/lon, depth is negative and in km
            self.faille.plot_surface(ex_ll,ey_ll,ez_ll,color=scalarMap.to_rgba(delta))

            # Reset x- and y-lims
            self.faille.set_xlim([self.lonmin,self.lonmax])
            self.faille.set_ylim([self.latmin,self.latmax])

        # # If 2d. Just plots center for now, should plot ellipse projection onto surface
        if plot_on_2d and self.carte is not None:
             self.carte.scatter(fault.ellipshape['x0'],fault.ellipshape['y0'])

        # put up a colorbar
        if colorbar:
            if self.faille is not None: self.addColorbar(delta, scalarMap, cbaxis, cborientation, self.figFaille, cblabel=cblabel)
            if plot_on_2d and self.carte is not None:
                self.addColorbar(delta, scalarMap, cbaxis, cborientation, self.figCarte, cblabel=cblabel)

        # All done
        return

    def faultTents(self, fault,
                   slip='strikeslip', norm=None, colorbar=True, alpha=1.0,
                   cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='',
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
        elif slip in ('sensitivity'):
            slip = fault.sensitivity.copy()
        else:
            print ("Unknown slip direction")
            return
        slip *= factor

        # norm
        if norm == None:
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
            x = np.array(x); #x[x<0.] += 360.
            if self.faille is not None: self.faille.plot3D(x, y, z, '-', color='gray', linewidth=1, alpha=alpha)
            if plot_on_2d and self.carte is not None: self.carte.plot(x, y, '-', color='gray', linewidth=1, zorder=zorder, alpha=alpha)

        # Plot the color for slip
        # 1. Get the subpoints for each triangle
        from .EDKSmp import dropSourcesInPatches as Patches2Sources
        #if hasattr(fault, 'edksSources') and not hasattr(fault, 'plotSources'):
        #    fault.plotSources = copy.deepcopy(fault.edksSources)
        #    fault.plotSources[1] /= 1e3
        #    fault.plotSources[2] /= 1e3
        #    fault.plotSources[3] /= 1e3
        #    fault.plotSources[4] /= 180./np.pi
        #    fault.plotSources[5] /= 180./np.pi
        #    fault.plotSources[6] /= 1e6
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
        if method == 'surface':

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
            #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
            if self.faille is not None: self.faille.plot_surface(lon, lat, -1.0*z, facecolors=cols, rstride=1, cstride=1, antialiased=True, linewidth=0, alpha=alpha)

            # On 2D?
            if plot_on_2d and self.carte is not None:
                lon, lat = fault.xy2ll(X, Y)
                #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
                self.carte.scatter(lon, lat, c=Slip, cmap=cmap, linewidth=0, s=2, vmin=vmin, vmax=vmax, zorder=zorder, alpha=alpha)

        elif method == 'scatter':
            # Do the scatter ploto
            lon, lat = fault.xy2ll(X, Y)
            #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
            if self.faille is not None: cb = self.faille.scatter3D(lon, lat, zs=-1.0*Z, c=Slip, cmap=cmap, linewidth=0, s=2, vmin=vmin, vmax=vmax, alpha=alpha)

            # On 2D?
            if plot_on_2d and self.carte is not None:
                self.carte.scatter(lon, lat, c=Slip, cmap=cmap, linewidth=0, vmin=vmin, vmax=vmax, zorder=zorder, alpha=alpha)
                if vertIndex:
                    for ivert,vert in enumerate(fault.Vertices_ll):
                        x,y = self.carte(vert[0], vert[1])
                        #if x<360.: #x+= 360.
                        plt.annotate('{}'.format(ivert), xy=(x,y),
                                     xycoords='data', xytext=(x,y), textcoords='data')

        # put up a colorbar
        if colorbar:
            if self.faille is not None: self.addColorbar(Slip, scalarMap, cbaxis, cborientation, self.figFaille, cblabel=cblabel) 
            if plot_on_2d and self.carte is not None:
                self.addColorbar(Slip, scalarMap, cbaxis, cborientation, self.figCarte, cblabel=cblabel)

        # All done
        return lon, lat, Z, Slip

    def surfacestress(self, stress, component='normal', linewidth=0.0, norm=None, 
                      colorbar=True, cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal',
                      cblabel=''):
        '''
        Plots the stress on the map.

        Args:
            * stress        : Stressfield object.

        Kwargs:
            * component     : If string, can be normal, shearstrike, sheardip. If tuple or list, can be anything specifying the indexes of the Stress tensor.
            * linewidth     : option of scatter.
            * norm          : Scales the color bar.
            * colorbar      : if true, plots a colorbar
            * cbaxis        : [Left, Bottom, Width, Height] of the colorbar axis
            * cborientation : 'horizontal' (default) 
            * cblabel       : Write something next to the colorbar.

        Returns:
            * None
        '''

        if self.carte is None:
            print('No Map figure to work on')
            return

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
        #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        sc = self.carte.scatter(xloc, yloc, s=20, c=val, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth)

        # colorbar
        if colorbar:
            self.addColorbar(val, sc, cbaxis, cborientation, self.figCarte, cblabel=cblabel) 

        # All don
        return

    def gps(self, gps, data=['data'], color=['k'], scale=None, 
            legendscale=10., legendunit='', linewidths=.1, name=False, error=True,
            zorder=5, alpha=1., width=0.005, headwidth=3, headlength=5, 
            headaxislength=4.5, minshaft=1, minlength=1, quiverkeypos=(0.1, 0.1)):
        '''
        Args:
            * gps           : gps object from gps.

        Kwargs:
            * data          : List of things to plot. Can be any list of 'data', 'synth', 'res', 'strain', 'transformation'
            * color         : List of the colors of the gps velocity arrows. Must be the same size as data
            * scale         : Scales the arrows
            * legendscale   : Length of the scale.
            * linewidths    : Width of the arrows.
            * name          : Plot the name of the stations (True/False).
            * zorder        : Order of the matplotlib rendering

        Returns:
            * None
        '''

        # Check
        if self.carte is None:
            print('No Map figure to work on ')
            return

        # Assert
        if (type(data) is not list) and (type(data) is str):
            data = [data]
        if (type(color) is not list):
            color = [color]
        if len(color)==1 and len(data)>1:
            color = [color[0] for d in data]

        # Check
        if scale == None:
            assert len(data)==1, 'If multiple data are plotted, need to provide scale'

        # Get lon lat
        lon = gps.lon
        #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = gps.lat

        # Make the dictionary of the things to plot
        Data = {}
        for dtype,col in zip(data, color):
            if dtype == 'data':
                dName = '{} Data'.format(gps.name)
                Values = gps.vel_enu
            elif dtype == 'synth':
                dName = '{} Synth.'.format(gps.name)
                Values = gps.synth
            elif dtype == 'res':
                dName = '{} Res.'.format(gps.name)
                Values = gps.vel_enu - gps.synth
            elif dtype == 'strain':
                dName = '{} Strain'.format(gps.name)
                Values = gps.Strain
            elif dtype == 'transformation':
                dName = '{} Trans.'.format(gps.name)
                Values = gps.transformation
            else:
                assert hasattr(gps, dtype), f'{dtype} is not available for plotting'
                dName = f'{dtype}'
                Values = getattr(gps, dtype)
            Data[dName] = {}
            Data[dName]['Values'] = Values
            Data[dName]['Color'] = col

            if dtype in ('data', 'res') and np.isfinite(gps.err_enu[:,:2]).all():
                Data[dName]['Error'] = gps.err_enu

        # Plot these
        for dName in Data:
            values = Data[dName]['Values']
            c = Data[dName]['Color']
            p = self.carte.quiver(lon, lat,
                                  values[:,0], values[:,1],
                                  color=c, width=width, headwidth=headwidth, 
                                  headlength=headlength, 
                                  headaxislength=headaxislength, 
                                  minshaft=minshaft, 
                                  minlength=minlength,
                                  scale=scale, scale_units='xy',
                                  linewidths=linewidths, 
                                  zorder=zorder, alpha=alpha)

            if 'Error' in Data[dName] and error:

                sigma = Data[dName]['Error']

                if scale is None:
                    print('Cannot plot ellipses or quiverkey if scale is None')
                    return

                self.carte.ellipses = []

                for vel, err, lo, la in zip(values, sigma, lon, lat):

                    # Found this on stackoverflow. Thanks!
                    # Basic ellipse definition
                    ellipse = patches.Ellipse((0, 0),
                            width=(err[0]),
                            height=(err[1]),
                            facecolor='none',
                            edgecolor=c, zorder=zorder)

                    # Transformation of the ellipse according to external parameters (obtained from various statistics on the data)
                    # We have a 360° issue with the ellipse that I don't understand here
                    if lo>180.: lo = lo - 360.
                    center=(lo+vel[0]/scale, la+vel[1]/scale)
                    transf = transforms.Affine2D().scale(1/scale, 1/scale).translate(center[0], center[1])
                    ellipse.set_transform(transf + self.carte.transData)

                    # Plot of the ellipse
                    self.carte.add_patch(ellipse)

        # Plot Legend
        if quiverkeypos is not None:
            q = self.carte.quiverkey(p, quiverkeypos[0], quiverkeypos[1],
                                     legendscale, '{} {}'.format(legendscale, legendunit),
                                     coordinates='axes', color=c, 
                                     zorder=max([_.zorder for _ in self.carte.get_children()]))

        # Plot the name of the stations if asked
        if name:
            font = {'family' : 'serif',
                    'color'  : 'k',
                    'weight' : 'normal',
                    'size'   : 10}
            lonmin, lonmax, latmin, latmax = self.carte.get_extent()
            for lo, la, sta in zip(lon.tolist(), lat.tolist(), gps.station):
                # Do it twice, I don't know why text is screwed up...
                if lo<lonmax and lo>lonmin and la>latmin and la<latmax: self.carte.text(lo, la, sta, zorder=20, fontdict=font)
                #self.carte.text(lo-360., la, sta, zorder=20, fontdict=font)

        # All done
        return

    def gpsverticals(self, gps, norm=None, colorbar=True,
                     cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='',
                     data=['data'], markersize=[10], linewidth=0.1,
                     zorder=4, cmap='jet', marker='o', alpha=1.):
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

        # Check
        if self.carte is None:
            print('No Map figure to work on')
            return

        # Assert
        if (type(data) is not list) and (type(data) is str):
            data = [data]
        if (type(markersize) is not list):
            markersize = [markersize]
        if len(markersize)==1 and len(data)>1:
            markersize = [markersize[0]*np.random.rand()*10. for d in data]

        # Get lon lat
        lon = gps.lon
        #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = gps.lat

        # Initiate
        vmin = 999999999.
        vmax = -999999999.

        # Make the dictionary of the things to plot
        from collections import OrderedDict
        Data = OrderedDict()
        for dtype,mark in zip(data, markersize):
            if dtype == 'data':
                dName = '{} Data'.format(gps.name)
                Values = gps.vel_enu[:,2]
            elif dtype == 'synth':
                dName = '{} Synth.'.format(gps.name)
                Values = gps.synth[:,2]
            elif dtype == 'res':
                dName = '{} Res.'.format(gps.name)
                Values = gps.vel_enu[:,2] - gps.synth[:,2]
            elif dtype == 'strain':
                dName = '{} Strain'.format(gps.name)
                Values = gps.Strain[:,2]
            elif dtype == 'transformation':
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
            sc = self.carte.scatter(lon, lat, s=mark, c=V, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=linewidth, zorder=zorder, alpha=alpha)

        # Colorbar
        if colorbar:
            self.addColorbar(V, sc, cbaxis, cborientation, self.figCarte, cblabel=cblabel) 

        return

    def gpsprojected(self, gps, norm=None, colorbar=True, 
                     cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='',
                     zorder=4, cmap='jet', alpha=1.):
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

        # Check 
        if self.carte is None:
            print('No Map figure to work on')
            return

        # Get the data
        d = gps.vel_los
        lon = gps.lon;
        #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = gps.lat

        # Prepare the color limits
        if norm == None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap(cmap)

        # Plot
        sc = self.carte.scatter(lon, lat, s=100, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.5, zorder=zorder, alpha=alpha)

        # plot colorbar
        if colorbar:
            self.addColorbar(d, sc, cbaxis, cborientation, self.figCarte, cblabel=cblabel)

        # All done
        return

    def earthquakes(self, earthquakes, plot='2d3d', color='k', markersize=5, norm=None, 
                    colorbar=False, cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', 
                    cblabel='', cmap='jet', zorder=2, alpha=1., linewidth=0.1):
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
            * cbaxis        : [Left, Bottom, Width, Height] of the colorbar axis
            * cborientation : 'horizontal' (default) 
            * cblabel       : Write something next to the colorbar.
            * zorder    : order of the plot in matplotlib

        Returns:
            * None
        '''

        # Color
        if color.__class__ is str:
            color = earthquakes.__getattribute__(color)
    
        # Get lon lat
        lon = earthquakes.lon
        #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = earthquakes.lat

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
            cmap = plt.get_cmap(cmap)
            cNorm = colors.Normalize(vmin=color.min(), vmax=color.max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
            scalarMap.set_array(color)
        else:
            cmap = None

        # plot the earthquakes on the map if ask
        if '2d' in plot and self.carte is not None:
            sc = self.carte.scatter(lon, lat, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=linewidth, edgecolor='k', zorder=zorder, alpha=alpha)
            if colorbar:
                self.addColorbar(color, sc, cbaxis, cborientation, self.figCarte, cblabel=cblabel) 

        # plot the earthquakes in the volume if ask
        if '3d' in plot and self.faille is not None:
            sc = self.faille.scatter3D(lon, lat, -1.*earthquakes.depth, s=markersize, c=color, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=linewidth, edgecolor='k', alpha=alpha)
            if colorbar:
                self.addColorbar(color, sc, cbaxis, cborientation, self.figFaille, cblabel=cblabel) 

        # All done
        return

    def beachball(self, fm, xy, facecolor='k', bgcolor='white', edgecolor='k', linewidth=2,
                        width=200, size=100, alpha=1.0, zorder=3):
        '''
        Plots a beach ball from a GCMT moment tensor. See obspy.imaging.beachball.beach to have a description of the arguments.

        Args:
            * fm        : Focal mechanism (M11, M22, M33, M12, M13, M23, Harvard convention)
            * xy        : Tuple of (lon, lat)

        Returns:
            * None:
        '''

        # Check
        if self.carte is None:
            print('No Map figure to work on')
            return

        # Import obspy, we just need it here so no general import
        from .beachball import beach

        # Get the beach ball
        bb = beach(fm, xy=xy, linewidth=linewidth, edgecolor=edgecolor, bgcolor=bgcolor, facecolor=facecolor, width=width, size=size, alpha=alpha, zorder=zorder)

        # Add it 
        self.carte.add_collection(bb)

        # All done
        return 

    def faultsimulation(self, fault, norm=None, colorbar=True, 
                        cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='',
                        direction='north'):
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

        # Check
        if self.carte is None:
            print('No Map figure to work on')
            return

        if direction == 'east':
            d = fault.sim.vel_enu[:,0]
        elif direction == 'north':
            d = fault.sim.vel_enu[:,1]
        elif direction == 'up':
            d = fault.sim.vel_enu[:,2]
        elif direction == 'total':
            d = np.sqrt(fault.sim.vel_enu[:,0]**2 + fault.sim.vel_enu[:,1]**2 + fault.sim.vel_enu[:,2]**2)
        elif direction.__class__ is float:
            d = fault.sim.vel_enu[:,0]/np.sin(direction*np.pi/180.) + fault.sim.vel_enu[:,1]/np.cos(direction*np.pi/180.)
        elif direction.shape[0]==3:
            d = np.dot(fault.sim.vel_enu,direction)
        else:
            print ('Unknown direction')
            return

        # Prepare the color limits
        if norm == None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')

        # Get lon lat
        lon = fault.sim.lon
        #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
        lat = fault.sim.lat

        # Plot the insar
        sc = self.carte.scatter(lon, lat, s=30, c=d, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.1)

        # plot colorbar
        if colorbar:
            self.addColorbar(d, sc, cbaxis, cborientation, self.figCarte, cblabel=cblabel) 

        # All done
        return

    def insar(self, insar, norm=None, colorbar=True, markersize=1, lognorm=False,
                    cbaxis=[0.2,0.2,0.1,0.01], cborientation='horizontal', cblabel='',
                    data='data', plotType='scatter', cmap='jet', los=None,
                    decim=1, zorder=3, edgewidth=1, alpha=1.):
        '''
        Plot an insar object

        Args:
            * insar     : insar object from insar.

        Kwargs:
            * norm      : lower and upper bound of the colorbar.
            * colorbar  : plot the colorbar (True/False).
            * cbaxis        : [Left, Bottom, Width, Height] of the colorbar axis
            * cborientation : 'horizontal' (default) 
            * cblabel       : Write something next to the colorbar.
            * data      : plot either 'data' or 'synth' or 'res' or 'err'.
            * plotType  : Can be 'decimate' or 'scatter'
            * los       : [lon, lat, length, fontsize (default=16), mutation_scale (default=90), 'flip'] 
            * decim     : In case plotType='scatter', decimates the data by a factor decim.
            * edgewidth : width of the patches in case plotTtype is decimate
            * zorder    : order of the plot in matplotlib

        Returns:
            * None
        '''

        # Check
        if self.carte is None:
            print('No Map figure to work on')
            return

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
        elif data in ['trans', 'transformation']:
            assert insar.orbit is not None, 'No Transformation available'
            d = insar.orbit
        elif data == 'err':
            assert insar.err is not None, 'No Error to plot'
            d = insar.err
        else:
            if hasattr(insar, data):
                assert len(getattr(insar, data)) == len(insar.vel), f'Attribute {data} has incorrect length ({len(getattr(insar, data))} instead of {len(insar.vel)}'
                d = getattr(insar, data)
            else:
                print('Unknown data type')
                return

        # Prepare the colorlimits
        if norm == None:
            vmin = np.nanmin(d)
            vmax = np.nanmax(d)
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap(cmap)
        if lognorm:
            cNorm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        if plotType == 'decimate':
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
                rect.set_alpha(alpha)
                self.carte.add_collection(rect)

        elif plotType == 'scatter':

            lon = insar.lon
            #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
            lat = insar.lat
            sc = self.carte.scatter(lon[::decim], lat[::decim], s=markersize,
                                    c=d[::decim], cmap=cmap, norm=cNorm, 
                                    linewidth=0.0, zorder=zorder, alpha=alpha)

        elif plotType == 'flat':
        
            # Is there NaNs:
            if np.isfinite(d).all(): print('Carefull: there is no NaNs, the interpolation might be a whole load of garbage...')
            # Get coordinantes
            x,y = insar.x,insar.y
            # Check if insar has the nx ny attributes
            if hasattr(insar, 'nx') and hasattr(insar, 'ny'):
                lon = insar.lon.reshape((insar.ny,insar.nx))
                lat = insar.lat.reshape((insar.ny,insar.nx))
                data = d.reshape((insar.ny,insar.nx))
            else:
                # Build an interpolator
                sarint = sciint.LinearNDInterpolator(np.vstack((x,y)).T, d, fill_value=np.nan)
                # Interpolate
                xx = np.linspace(self.lonmin, self.lonmax, 1000)
                yy = np.linspace(self.latmin, self.latmax, 1000)
                xx,yy = np.meshgrid(xx,yy)
                xx,yy = insar.ll2xy(xx,yy)
                data = sarint(xx,yy)
                lon,lat = insar.xy2ll(xx,yy)
            # Plot
            sc = self.carte.pcolormesh(lon, lat, data, cmap=cmap, norm=cNorm,
                                       zorder=zorder, alpha=alpha)

        else:
            print('Unknown plot type: {}'.format(plotType))
            return

        # plot colorbar
        if colorbar:
            self.addColorbar(d, scalarMap, cbaxis, cborientation, self.figCarte, cblabel=cblabel) 

        # Plot LOS
        if los is not None:
            lonl = los[0]
            latl = los[1]
            loslength= los[2]
            if los[3] is not None:
                fontsize = los[3]
            else:
                fontsize = 16
            if los[4] is not None:
                mutscale = los[4]
            else:
                mutscale = 90.
            x1,y1 = insar.ll2xy(lonl,latl)
            dx = np.nanmean(insar.los[:,0])
            dy = np.nanmean(insar.los[:,1])
            x2 = x1-loslength*dx
            y2 = y1-loslength*dy
            xt = x1-loslength*dx/2.5
            yt = y1-loslength*dy/2.5
            angle = np.arctan2(dy,dx)*180./np.pi + 180.
            if los[5] == 'flip': angle+=180.
            lon1,lat1 = insar.xy2ll(x1,y1)
            lon2,lat2 = insar.xy2ll(x2,y2)
            lont,latt = insar.xy2ll(xt,yt)
            self.carte.annotate("LOS", xy=(lont, latt),
                                fontweight='bold', rotation=angle, fontsize=fontsize, 
                                ha='center', va='center',
                                zorder=11)
            self.carte.annotate("", xy=(lon2, lat2), xytext=(lon1, lat1), 
                                arrowprops=dict(arrowstyle="simple", mutation_scale=mutscale, 
                                                     facecolor='white', edgecolor='k'),
                                zorder=10)


        # All done
        return

    def opticorr(self, corr, norm=None, colorbar=True, 
                 cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='',
                 data='dataEast', plotType='decimate', decim=1, zorder=3):
        '''
        Plot opticorr instance

        Args:
            * corr      : instance of the class opticorr

        Kwargs:
            * norm      : lower and upper bound of the colorbar.
            * colorbar  : plot the colorbar (True/False).
            * cbaxis        : [Left, Bottom, Width, Height] of the colorbar axis
            * cborientation : 'horizontal' (default) 
            * cblabel       : Write something next to the colorbar.
            * data      : plot either 'dataEast', 'dataNorth', 'synthNorth', 'synthEast', 'resEast', 'resNorth', 'data', 'synth' or 'res'
            * plotType  : plot either rectangular patches (decimate) or scatter (scatter)
            * decim     : decimation factor if plotType='scatter'
            * zorder    : order of the plot in matplotlib

        Returns:
            * None
        '''

        # Check
        if self.carte is None:
            print('No Map figure to work on')
            return

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
        if norm == None:
            vmin = d.min()
            vmax = d.max()
        else:
            vmin = norm[0]
            vmax = norm[1]

        # Prepare the colormap
        cmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

        if plotType == 'decimate':
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

        elif plotType == 'scatter':
            lon = corr.lon
            #lon[np.logical_or(lon<self.lonmin, lon>self.lonmax)] += 360.
            lat = corr.lat
            self.carte.scatter(lon[::decim], lat[::decim], s=10., c=d[::decim], cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.0, zorder=zorder)

        else:
            assert False, 'unsupported plot type. Must be rect or scatter'

        # plot colorbar
        if colorbar:
            self.addColorbar(d, scalarMap, cbaxis, cborientation, self.figCarte, cblabel=cblabel) 

        # All done
        return

    def slipdirection(self, fault, linewidth=1., color='k', scale=1., zorder=10, markersize=10, factor=1.0):
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

        # Check if it exists
        fault.computeSlipDirection(scale=scale, factor=factor)

        # Loop on the vectors
        for v in fault.slipdirection:
            # Z increase downward
            v[0][2] *= -1.0
            v[1][2] *= -1.0
            # Make lists
            x, y, z = zip(v[0],v[1])
            lo,la = fault.xy2ll(np.array(x),np.array(y))
            # Plot
            if self.faille is not None: self.faille.plot3D(lo, la, z, color=color, linewidth=linewidth, zorder=zorder)
            # Plot map
            if self.carte is not None:
                self.carte.plot(lo,la,color=color, linewidth=linewidth, zorder=zorder)
                self.carte.plot(lo[0],la[0], '.', color=color, markersize=markersize)

        # All done
        return

#EOF
