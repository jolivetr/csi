'''
A class that deals with creepmeter data

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import os

# Personals
from .SourceInv import SourceInv

class creepmeters(SourceInv):

    '''
    A class that handles creepmeter data

    Args:
       * name      : Name of the dataset.

    Kwargs:
       * utmzone   : UTM zone  (optional, default=None)
       * lon0      : Longitude of the center of the UTM zone
       * lat0      : Latitude of the center of the UTM zone
       * ellps     : ellipsoid (optional, default='WGS84')
       * verbose   : Speak to me (default=True)

    '''

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Initialize the data set 
        self.dtype = 'creepmeters'

        # Print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print (" Initialize Creepmeters data set {}".format(self.name))
        self.verbose = verbose

        # Base class init
        super(creepmeters,self).__init__(name,
                                         utmzone = utmzone,
                                         ellps = ellps,
                                         lon0 = lon0,
                                         lat0 = lat0) 

        # Initialize some things
        self.data = {}

        # All done
        return

    def readStationList(self, filename):
        '''
        Reads the list of Stations. Input file format is

        +--------------+-----+-----+
        | Station Name | Lon | Lat |
        +--------------+-----+-----+

        Args:
            * filename        : Input file.

        Returns:
            * None
        '''

        # open the file
        fin = open(filename, 'r')

        # Read all lines
        Text = fin.readlines()
        fin.close()

        # Create lists
        self.station = []
        self.lon = []
        self.lat = []

        # Loop
        for t in Text:
            tex = t.split()
            self.station.append(tex[0])
            self.lon.append(np.float(tex[2]))
            self.lat.append(np.float(tex[1]))

        # translate to array
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)

        # Convert to utm
        self.lonlat2xy()

        # All done
        return

    def position(self, station):
        ''' 
        Returns lon,lat of a station.

        Args:
            * station   : Name of a station.

        Returns:
            * lon, lat 
        '''
        
        # Find it
        u = np.flatnonzero(np.array(self.station)==station)
        u = u[0]

        # Get lon, lat and return
        return self.lon[u], self.lat[u]

    def distance(self, station, point, direction):
        '''
        Computes the distance between a station and a point.

        Args:
            * station   : Name of a station.
            * point     : [Lon, Lat].
            * direction : Direction of the positive sign.

        Returns:
            * None. Distance is stored in the {data} attribute under the station name.

        '''

        # Check
        if 'Distance' not in self.data[station].keys():
            self.data[station]['Distance'] = []

        # Get station lon,lat
        lon, lat = self.position(station)
        x, y = self.ll2xy(lon,lat)

        # Transfert point
        x0, y0 = self.ll2xy(point[0],point[1])

        # Computes the sign
        Dir = np.array([np.cos(direction*np.pi/180.), np.sin(direction*np.pi/180.)])
        vec = np.array([x0-x, y0-y])
        sign = np.sign(np.dot(Dir, vec))

        # Compute distance
        d = np.sqrt( (x0-x)**2 +(y0-y)**2 ) * sign

        # Stores it
        self.data[station]['Distance'].append([[lon,lat], d])

        # all done
        return

    def deleteStation(self, station):
        '''
        Removes a station.

        Args:
            * station   : Name of the station to remove

        Returns:
            * None
        '''

        # find it
        u = np.flatnonzero(np.array(self.station)==station)
        
        # delete it
        del self.station[u]
        self.lon = np.delete(self.lon, u)
        self.lat = np.delete(self.lat, u)

        # All done 
        return

    def readAllStations(self, directory='.'):
        '''
        Reads all the station files.

        Kwargs:
            * directory : directory where to find the station files

        Returns:
            * None
        '''

        for station in self.station:
            self.readStationData(station,directory=directory)

        # stations to delete
        sta = []
        for station in self.station:
            if self.data[station] == {}:
                sta.append(station)
        for s in sta:
            self.deleteStation(s)

        # all done
        return

    def readStationData(self, station, directory='.'):
        '''
        From the name of a station, reads what is in station.day.

        Args:
            * station   : name of the station

        Kwargs:
            * directory : directory where to find the station.day file.

        Returns:
            * None
        '''

        # Filename
        filename = '{}/{}.day'.format(directory,station)
        if not os.path.exists(filename):
            filename = '{}/{}.m'.format(directory,station)
            if not os.path.exists(filename):
                self.data[station] = {}
                return

        # Create the storage
        self.data[station] = {}
        self.data[station]['Time'] = []
        self.data[station]['Offset'] = []
        t = self.data[station]['Time']
        o = self.data[station]['Offset']

        # Open
        fin = open(filename, 'r')

        # Read everything in it
        Text = fin.readlines()
        fin.close()

        # Loop 
        for text in Text:

            # Get values
            yr = np.int(text.split()[0])
            da = np.int(text.split()[1])
            of = np.float(text.split()[2])

            # Compute the time 
            time = dt.datetime.fromordinal(dt.datetime(yr, 1, 1).toordinal() + da)

            # Append
            t.append(time)
            o.append(of)

        # Arrays
        self.data[station]['Time'] = np.array(self.data[station]['Time'])
        self.data[station]['Offset'] = np.array(self.data[station]['Offset'])

        # All done
        return

    def selectbox(self, minlon, maxlon, minlat, maxlat):
        ''' 
        Select the earthquakes in a box defined by min and max, lat and lon.
        
        Args:
            * minlon        : Minimum longitude.
            * maxlon        : Maximum longitude.
            * minlat        : Minimum latitude.
            * maxlat        : Maximum latitude.

        Returns:
            * None
        '''

        # Store the corners
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat

        # Select on latitude and longitude
        print( "Selecting the earthquakes in the box Lon: {} to {} and Lat: {} to {}".format(minlon, maxlon, minlat, maxlat))
        u = np.flatnonzero((self.lat>minlat) & (self.lat<maxlat) & (self.lon>minlon) & (self.lon<maxlon))

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.station = sellf.station[u]

        # All done
        return

    def lonlat2xy(self):
        '''
        Converts the lat lon positions into utm coordinates.

        Returns:
            * None
        '''

        self.x, self.y = self.ll2xy(self.lon, self.lat)

        # all done
        return

    def fitLinearAllStations(self, period=None, directory='.'):
        '''
        Fits a linear trend to all the available stations. Can specify a period=[startdate, enddate]

        Args:
            * period    : list of 2 tuples (yyyy, mm, dd)

        Kwargs:
            * directory : If station files have not been read before, this is the directory where to find the station files

        Returns:
            * None
        '''

        # Loop 
        for station in self.station:
            print('Fitting Station {}'.format(station))
            self.fitLinear(station, period=period, directory=directory)

        # All done
        return

    def fitLinear(self, station, period=None, directory='.'):
        '''
        Fits a linear trend onto the offsets for the station 'station'.

        Args:
            * station   : station name

        Kwargs:
            * period    : list of 2 tuples (yyyy, mm, dd)
            * directory : If station files have not been read before, this is the directory where to find the station files

        Returns:
            * None
        '''

        # Check if the station has been read before
        if not (station in self.data.keys()):
            self.readStationData(station, directory=directory)
            if not (station in self.data.keys()):
                print('Cannot fit station {} ...'.format(station))
                return

        # Creates a storage
        self.data[station]['Linear'] = {}
        store = self.data[station]['Linear']

        # Create the dates
        if period is None:
            date1 = self.data[station]['Time'][0]
            date2 = self.data[station]['Time'][1]
        else:
            date1 = dt.datetime(period[0][0], period[0][1], period[0][2])
            date2 = dt.datetime(period[1][0], period[1][1], period[1][2])

        # Keep the period
        store['Period'] = []
        store['Period'].append(date1)
        store['Period'].append(date2)

        # Get the data we want
        time = self.data[station]['Time']
        offset = self.data[station]['Offset']

        # Get the dates we want
        u = np.flatnonzero(time>=date1)
        v = np.flatnonzero(time<=date2)
        w = np.intersect1d(u,v)
        if w.shape[0]<2:
            print('Not enough points for station {}'.format(station))
            store['Fit'] = None
            return
        ti = time[w]
        of = offset[w]

        # pass the dates into real numbers
        tr = np.array([self.date2real(ti[i]) for i in range(ti.shape[0])])

        # Make an array
        A = np.ones((tr.shape[0], 2))
        A[:,0] = tr

        # invert
        m, res, rank, s = np.linalg.lstsq(A, of)

        # Stores the results
        store['Fit'] = m

        # all done
        return

    def plotStation(self, station, figure=100, save=None):
        '''
        Plots one station evolution through time.

        Args:
            * station   : name of the station

        Kwargs:
            * figure    : figure numner
            * save      : name of the file if you want to save

        Returns:
            * None
        '''

        # Check if the station has been read
        if not (station in self.data.keys()):
            print('Read the data first...')
            return

        # Create figure
        fig = plt.figure(figure)
        ax = fig.add_subplot(111)

        # Title
        ax.set_title(station)

        # plot data
        t = self.data[station]['Time']
        o = self.data[station]['Offset']
        ax.plot(t, o, '.k')

        # plot fit
        if 'Linear' in self.data[station].keys():
            if self.data[station]['Linear']['Fit'] is not None:
                v = self.data[station]['Linear']['Fit'][0]
                c = self.data[station]['Linear']['Fit'][1]
                date1 = self.data[station]['Linear']['Period'][0]
                date2 = self.data[station]['Linear']['Period'][1]
                dr1 = self.date2real(date1)
                dr2 = self.date2real(date2)
                plt.plot([date1, date2],[c+dr1*v, c+dr2*v], '-r')

        # save
        if save is not None:
            plt.savefig(save)

        # Show
        plt.show()

        # All done 
        return


    def date2real(self, date):
        '''
        Pass from a datetime to a real number.
        Weird method... Who implemented that?

        Args:
            * date  : datetime instance

        Returns:
            * float
        '''

        yr = date.year
        yrordi = dt.datetime(yr, 1, 1).toordinal()
        ordi = date.toordinal()
        days = ordi - yrordi
        return yr + days/365.25

#EOF
