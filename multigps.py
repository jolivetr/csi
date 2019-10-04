''' 
A class that deals with multiple gps objects

Written by R. Jolivet, May 2014.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import os
import copy
import sys

# Personals
from .gps import gps
from .gpstimeseries import gpstimeseries

class multigps(gps):

    '''
    A class that handles several networks of gps stations

    Args:
       * name      : Name of the dataset.

    Kwargs:
       * gpsobjects: list of instances of the gps class
       * obs_per_station: how many data points per site
       * utmzone   : UTM zone  (optional, default=None)
       * lon0      : Longitude of the center of the UTM zone
       * lat0      : Latitude of the center of the UTM zone
       * ellps     : ellipsoid (optional, default='WGS84')
       * verbose   : Speak to me (default=True)

    '''
    def __init__(self, name, gpsobjects=None, utmzone=None, ellps='WGS84', obs_per_station=2, lon0=None, lat0=None):

        # Base class init
        super(multigps,self).__init__(name,
                                      utmzone=utmzone,
                                      ellps=ellps,
                                      lon0=lon0,
                                      lat0=lat0,
                                      verbose=False) 
        
        # Set things
        self.dtype = 'multigps'
        self.obs_per_station = obs_per_station
 
        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Initialize Multiple GPS array {}".format(self.name))

        # Initialize things
        self.vel_enu = None
        self.err_enu = None
        self.rot_enu = None
        self.synth = None

        # Set objects
        if gpsobjects is not None:
            self.setgps(gpsobjects)

        # All done
        return
    
    def setgps(self, gpsobjects):
        '''
        Takes list of gps and set it 

        Args:
            * gpsobjects    : List of gps objects.

        Returns:
            * None
        '''

        # Get the list
        self.gpsobjects = gpsobjects

        # Loop over the stations to get the number of stations 
        ns = 0
        for gp in gpsobjects:
            ns += gp.station.shape[0]

        # Get the factor
        self.factor = gpsobjects[0].factor

        # Create the arrays
        self.station = np.zeros(ns).astype('|S4')
        self.lon = np.zeros(ns)
        self.lat = np.zeros(ns)
        self.x = np.zeros(ns)
        self.y = np.zeros(ns)
        self.vel_enu = np.zeros((ns,3))
        self.err_enu = np.zeros((ns,3))

        # obs per stations
        obs_per_station = []

        # Loop over the gps objects to feed self
        ns = 0
        for gp in gpsobjects:
            
            # Assert
            assert gp.factor==self.factor, 'GPS object have a different factor: Object {}'.format(gp.name)
            assert gp.utmzone==self.utmzone, 'UTM zone is not compatible: Object {}'.format(gp.utmzone)

            # Set starting and ending points
            st = ns
            ed = ns + gp.station.shape[0]

            # Get stations
            self.station[st:ed] = gp.station

            # Feed in 
            self.lon[st:ed] = gp.lon
            self.lat[st:ed] = gp.lat
            self.x[st:ed] = gp.x
            self.y[st:ed] = gp.y
            self.vel_enu[st:ed,:] = gp.vel_enu
            self.err_enu[st:ed,:] = gp.err_enu

            # Force number of observation per station
            gp.obs_per_station = self.obs_per_station

            # Update ns
            ns += gp.station.shape[0]

        # All done
        return

    def getNumberOfTransformParameters(self, transformation):
        '''
        Returns the number of transform parameters for the given transformation.

        Args:
            * transformation: List [main_transformation, [transfo_subnet1, transfo_subnet2, ....] ]. Each can be 'strain', 'full', 'strainnorotation', 'strainnotranslation', 'strainonly'
        
        Returns:
            * int
        '''

        # Separate 
        mainTrans = transformation[0]
        subTrans = transformation[1]

        # Assert
        assert type(mainTrans) in (str, type(None)), 'First Item of transformation list needs to be string'
        assert type (subTrans) is list, 'Second Item of transformation list needs to be a list'

        # Nhere
        nMain = super(multigps,self).getNumberOfTransformParameters(transformation[0])
        
        # Each subnetwork
        nSub = 0
        for trans, gp in zip(subTrans, self.gpsobjects):
            nSub += gp.getNumberOfTransformParameters(trans)

        # Sum
        Npo = nSub + nMain

        # All done
        return Npo

    def getTransformEstimator(self, transformation, computeNormFact=True):
        '''
        Returns the estimator for the transform.

        Args:
            * transformation : List [main_transformation, [transfo_subnet1, transfo_subnet2, ....] ]. Each item can be 'strain', 'full', 'strainnorotation', 'strainnotranslation', 'strainonly'

        Kwargs:
            * computeNormFact: compute and store the normalizing factor

        Returns:
            * 2D array
        '''

        # Separate
        mainTrans = transformation[0]
        subTrans = transformation[1]

        # Assert
        assert type(mainTrans) in (str, type(None)), 'First Item of transformation list needs to be string'
        assert type (subTrans) is list, 'Second Item of transformation list needs to be a list: {}'.format(subTrans)

        # Need the number of columns and lines
        nc = self.getNumberOfTransformParameters(transformation)
        ns = self.station.shape[0]*self.obs_per_station

        # Create the big holder
        orb = np.zeros((ns,nc))

        # Get the main transforms
        Morb = super(multigps,self).getTransformEstimator(mainTrans, 
                                                          computeNormFact=computeNormFact)

        # Put it where it should be
        if Morb is not None:
            cst = Morb.shape[1]
            orb[:, :cst] = Morb
        else:
            cst = 0
        
        # Loop over the subnetworks
        lst_east = 0
        lst_north = self.station.shape[0]
        lst_up = 2*self.station.shape[0]
        for trans, gp in zip(subTrans, self.gpsobjects):
            # Get the transform
            Sorb = gp.getTransformEstimator(trans, computeNormFact=computeNormFact)
            if Sorb is not None:
                # Set the indexes right
                ced = cst + Sorb.shape[1]
                led_east = lst_east + gp.station.shape[0]
                led_north = lst_north + gp.station.shape[0]
                # Put it where it should be 
                orb[lst_east:led_east, cst:ced] = Sorb[:gp.station.shape[0],:]
                orb[lst_north:led_north, cst:ced] = Sorb[gp.station.shape[0]:2*gp.station.shape[0],:]
                # Deal with verticals if needed
                if self.obs_per_station==3:
                    led_up = lst_up + gp.station.shape[0]
                    orb[lst_up:led_up, cst:ced] = Sorb[2*gp.station.shape[0]:,:]
                # Update column
                cst += Sorb.shape[1]
            # update lines
            lst_east += gp.station.shape[0]
            lst_north += gp.station.shape[0]

        # All done
        return orb

    def computeTransformation(self, fault, custom=False):
        '''
        Computes the transformation that is stored with a particular fault.

        Args:
            * fault: instance of a fault class that has a {polysol} attribute

        Kwargs:
            * custom: True/False. Do we have to account for custom GFs?

        Returns:
            * None. Transformation is stroed in attribute {transformation}
        
        '''

        # Get the transformation 
        transformation = fault.poly[self.name]

        # Separate
        mainTrans = transformation[0]
        subTrans = transformation[1]

        # Get the solution
        Tvec = fault.polysol[self.name]['{}'.format(transformation)]

        # Assert
        assert type(mainTrans) in (str, type(None)), 'First Item of transformation list needs to be string'
        assert type(subTrans) is list, 'Second Item of transformation list needs to be a list'

        # Get the estimator
        orb = self.getTransformEstimator(transformation)

        # Get the starting point
        st = self.getNumberOfTransformParameters([mainTrans,[None for i in range(len(subTrans))]])

        # Loop over the transformations
        for trans, gp in zip(subTrans, self.gpsobjects):
            
            # Put the transform in the fault
            fault.poly[gp.name] = trans

            # Put the solution in the fault
            nP = gp.getNumberOfTransformParameters(trans)     
            ed = st + nP
            fault.polysol[gp.name] = Tvec[st:ed]

            # Edit st
            st += nP

        # make the array
        self.transformation = np.zeros(self.vel_enu.shape)

        # Check
        if orb is None:
            return

        # Compute the synthetics
        tmpsynth = np.dot(orb, Tvec)

        # Fill it
        no = self.vel_enu.shape[0]
        self.transformation[:,0] = tmpsynth[:no]
        self.transformation[:,1] = tmpsynth[no:2*no]
        if self.obs_per_station==3:
            self.transformation[:,2] = tmpsynth[2*no:]

        # Add custom
        if custom:
            self.computeCustom(fault)
            if self.obs_per_station==1:
                self.transformation[:,2] += self.custompred[:,2]
            if self.obs_per_station>=2:
                self.transformation[:,0] += self.custompred[:,0]
                self.transformation[:,1] += self.custompred[:,1]
            if self.obs_per_station==3:
                self.transformation[:,2] += self.custompred[:,2]

        # All done
        return

#EOF
