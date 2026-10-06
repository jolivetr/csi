'''
Class that plots Kinematic faults.

Written by Z. Duputel, January 2014.
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as colls
import numpy as np

# Base class
from .geodeticplot import geodeticplot

class seismicplot(geodeticplot):

    '''
    A class to plot kinematic faults

    Kwargs:
        * figure        : Number of the figure.
        * ref           : 'utm' or 'lonlat'.
        * pbaspect      : XXXXX?????

    '''

    def __init__(self, figure=130, ref='utm', pbaspect=None):

        # Base class init
        projection = 'cyl'      
        super(seismicplot,self).__init__(figure,pbaspect,projection)
        
    def faultPatchesGrid(self, fault, slip='strikeslip', norm=None, colorbar=True, 
                         plot_on_2d=False, revmap=False, data=None, plotgrid=True):
        '''
        Plots a grid of fault patches

        Args:
            * fault         : Fault class from verticalfault.

        Kwargs:
            * slip          : Can be 'strikeslip', 'dipslip' or 'opening'
            * Norm          : Limits for the colorbar.
            * colorbar      : if True, plots a colorbar.
            * plot_on_2d    : if True, adds the patches on the map.
            * revmap        : Reverse color map
            * data          : add points in the x and y attributes of data
            * plotgrid      : Show grid points

        Returns:
            * None
        '''

        Xs,Ys = self.faultpatches(fault, slip=slip, norm=norm, colorbar=colorbar, 
                                  plot_on_2d=plot_on_2d, revmap=revmap, linewidth=1.0)

        # Plot Hypo
        if fault.hypo_x != None:
            self.faille.scatter3D(fault.hypo_x,fault.hypo_y,-fault.hypo_z+2.,color='k',
                                  marker=(5,1,0),s=150,zorder=1000)

        # Plot Grid
        if plotgrid:
            for p in range(len(fault.patch)):
                grid = np.array(fault.grid[p])      
                self.faille.scatter3D(grid[:,0],grid[:,1],-grid[:,2]+1.0,color='w',
                                      marker='o',s=10,zorder=1000)


        if data!=None:
            for x,y in zip(data.x,data.y):
                self.faille.scatter3D(x,y,0.,color='b',marker='v',s=20,zorder=1000)
                Xs = np.append(Xs,x)
                Ys = np.append(Ys,y)

        # All done
        return Xs,Ys


    def faulttrace(self, fault, color='r', add=False, data=None):
        '''
        Plot the fault trace

        Args:
            * fault         : Fault class from verticalfault.

        Kwargs:
            * color         : Color of the fault.
            * add           : plot the faults in fault.addfaults    
            * data          : Add locations of the x and y attribute of data

        Returns:
            * None
        '''

        # Plot the added faults before
        if add and (self.ref=='utm'):
            for f in fault.addfaultsxy:
                self.faille.plot(f[0], f[1], '-k')
                self.carte.plot(f[0], f[1], '-k')
        elif add and (self.ref!='utm'):
            for f in fault.addfaults:
                self.faille.plot(f[0], f[1], '-k')
                self.carte.plot(f[0], f[1], '-k')

        # Plot the surface trace
        print(fault.top)
        if self.ref=='utm':
            if fault.xf is None:
                fault.trace2xy()
            self.faille.plot3D(fault.xf, fault.yf,-fault.top, '-{}'.format(color), linewidth=2)
            self.carte.plot(fault.xf, fault.yf, '-{}'.format(color), linewidth=2)
        else:
            self.faille.plot3D(fault.lon, fault.lat,-fault.top, '-{}'.format(color), linewidth=2)
            self.carte.plot(fault.lon, fault.lat, '-{}'.format(color), linewidth=2)
            
            
        if data!=None:
            self.carte.plot(data.x,data.y,'bv',zorder=1000)

        # All done
        return





#EOF
