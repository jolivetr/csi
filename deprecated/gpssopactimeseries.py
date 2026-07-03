''' 
A class that deals with gps time series.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import sys

from .SourceInv import SourceInv

class gpstimeseries(SourceInv):

    def __init__(self, name, utmzone=None, lon0=None, lat0=None, ellps='WGS84', verbose=True):
        '''
        Args:
            * name      : Name of the dataset.
            * datatype  : can be 'gps' or 'insar' for now.
            * utmzone   : UTM zone. Default is 10 (Western US).
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize GPS array {}".format(self.name))
        self.verbose = verbose

        # Base class init
        super(gpstimeseries, self).__init__(name, 
                                            utmzone=utmzone,
                                            lon0=lon0,
                                            lat0=lat0,
                                            ellps=ellps)

        # Set things
        self.name = name
        self.dtype = 'gpstimeseries'

        # Initialize things
        self.data = None

        # All done
        return

    def get_station_list_from_sopac(self, coordfile):
        '''
        Reading velocities from Sopac file and converting to mm/yr.
        Args:
            * coordfile : File containing the coordinates.
        '''

        print ("Read data from file {} into data set {}".format(coordfile, self.name))

        # Keep the files, to remember
        self.coordfile = coordfile

        # open the files
        fcor = open(self.coordfile, 'r')

        # read them
        Cor = fcor.readlines()

        # Initialize things
        self.lon = []           # Longitude list
        self.lat = []           # Latitude list
        self.station = []       # List of the stations

        # Loop
        for c in range(len(Cor)):

            self.lon.append(float(Cor[c].split()[9]))
            self.lat.append(float(Cor[c].split()[8]))
            self.station.append(Cor[c].split()[0])

        # Make np array with that
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.station = np.array(self.station)

        # Pass to xy 
        self.ll2xy()

        # All done
        return

    def ll2xy(self):
        '''
        Pass the position into the utm coordinate system.
        '''

        x, y = self.putm(self.lon, self.lat)
        self.x = x/1000.
        self.y = y/1000.

        # All done
        return

    def select_stations(self, minlon, maxlon, minlat, maxlat):
        ''' 
        Select the stations in a box defined by min and max, lat and lon.
        
        Args:
            * minlon        : Minimum longitude.
            * maxlon        : Maximum longitude.
            * minlat        : Minimum latitude.
            * maxlat        : Maximum latitude.
        '''

        # Store the corners
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat

        # Select on latitude and longitude
        u = np.flatnonzero((self.lat>minlat) & (self.lat<maxlat) & (self.lon>minlon) & (self.lon<maxlon))

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.station = self.station[u]
        self.x = self.x[u]
        self.y = self.y[u]
        if self.data is not None:
            self.data = self.data[u,:]

        # All done
        return

    def reject_stations(self, station):
        '''
        Reject the stations named in stations.
        Args:
            * station   : name or list of names of station.
        '''

        if station.__class__ is str:

            # Get the concerned station
            u = np.flatnonzero(self.station == station)

            # Get the name
            sta = self.station[u]

            if u.size != 0:

                # Delete
                self.station = np.delete(self.station, u, axis=0)
                self.lon = np.delete(self.lon, u, axis=0)
                self.lat = np.delete(self.lat, u, axis=0)

            if (self.data is not None):
                if (sta in self.data.keys()):
                    del self.data[sta]

        elif station.__class__ is list:

            for sta in station:

                # Get the concerned station
                u = np.flatnonzero(self.station == sta)

                # get the name
                sta = self.station[u]

                if u.size != 0:

                    # Delete
                    self.station = np.delete(self.station, u, axis=0)
                    self.lon = np.delete(self.lon, u, axis=0)
                    self.lat = np.delete(self.lat, u, axis=0)
                
                if (self.data is not None):
                    if (sta in self.data.keys()):
                        del self.data[sta]

        # Update x and y
        self.ll2xy()

        # All done
        return

    def read_stations(self, directory='/Users/jolivetinsar/Documents/ParkfieldCreep/GPS/TimeSeries/Filtered', scale=1000.):
        '''
        Reads all the station time series from the sopac files.
        This fills in the self.data with gpstation objects.
        Args:
            * directory     : Where to find the GPS files.
            * scale         : Scale to mm.
        '''

        # Store things
        self.directory = directory
        self.scale = scale

        # Create a directory
        self.data = {}
        
        # Create a rejection map
        reject = []

        # Loop over the stations
        for sta in self.station:
            
            site = gpssta.gpsstation(sta, directory=directory)

            if site.valid:
                string = '\r Importing station {}'.format(sta)
                sys.stdout.write(string)
                sys.stdout.flush()
                site.read_sopac_timeseries()
                site.read_sopac_model()
                site.scaledisp(scale)
                self.data[sta] = site
            else:
                reject.append(sta)

        # Reject the bad stations
        self.reject_stations(reject)

        # Clean the screen
        sys.stdout.write('\n')
        sys.stdout.flush()

        # All done
        return

    def getlonlat(self, sta):
        '''
        Gets the longitude and latitude of a station.
        Args:
            * sta       : Name of the station.
        '''
        
        # Get index
        u = np.flatnonzero(self.station == sta)

        # All done
        return self.lon[u], self.lat[u]

    def velocity2file(self, filename, period=[2006, 2012]):
        '''
        Takes the velocity out of the sopac model for the period and makes a file:
        StationName | Lon | Lat | e_vel | n_vel | u_vel | e_err | n_err | u_err
        Args:
            * filename  : Name of the output file.
            * period    : Period for the velocity estimation.
        '''

        # open the output file
        fout = open(filename, 'w')

        # Loop over the stations
        for stn in self.data:
            
            # get the station
            sta = self.data[stn]

            # Get the velocity dictionary
            velo = sta.velo

            # Get lon and lat
            lon = sta.lon
            lat = sta.lat

            # Initialize velocities
            e_vel = np.nan
            e_err = np.nan
            n_vel = np.nan
            n_err = np.nan
            u_vel = np.nan
            u_err = np.nan

            # East
            v = velo['east']
            vitesse = 0
            erreur = 0
            n = 0
            for i in range(len(v)):
                win = v[i][2]
                if (win[0]<=period[0]) and (win[1]>=period[1]):
                    n += 1
                    vitesse += v[i][0]
                    erreur += v[i][1]
            if n>0:
                e_vel = vitesse / n
                e_err = erreur / n

            # North
            v = velo['north']
            vitesse = 0
            erreur = 0
            n = 0
            for i in range(len(v)):
                win = v[i][2]
                if (win[0]<=period[0]) and (win[1]>=period[1]):
                    n += 1
                    vitesse += v[i][0]
                    erreur += v[i][1]
            if n>0:
                n_vel = vitesse / n
                n_err = erreur / n

            # Up
            v = velo['up']
            vitesse = 0
            erreur = 0
            n = 0
            for i in range(len(v)):
                win = v[i][2]
                if (win[0]<=period[0]) and (win[1]>=period[1]):
                    n += 1
                    vitesse += v[i][0]
                    erreur += v[i][1]
            if n>0:
                u_vel = vitesse / n
                u_err = erreur / n

            # Create the string
            string = '{} {} {} {} {} {} {} {} {} \n'.format(sta.name, lon, lat, e_vel, n_vel, u_vel, e_err, n_err, u_err)
            fout.write(string)

        # Close file
        fout.close()

        # All done
        return

    def extract_velocities(self):
        '''
        Extracts the velocities for all the stations
        Velocity dictionaries for each station are stored in velo
        '''

        # Loop over the stations
        for sta in self.station:

            # Get station
            stn = self.data[sta]

            # Get lon lat
            lon, lat = self.getlonlat(sta)

            # Store these
            stn.lon = lon
            stn.lat = lat

            # Get the velocity dictionnary
            velo = stn.spitsopacvelocity()

            # Create the storage in the station
            self.data[sta].velo = velo

        # All done
        return

    def err_lower_bound(self, low):
        '''
        Take the errors that are smaller than "low" and substitute these by "low".
        Args:
            * low       : Threshold value.
        '''

        for sta in self.station:

            # Get station
            stn = self.data[sta]

            # Get the velo dictionary
            velo = stn.velo

            # East
            n = len(velo['east'])
            for i in range(n):
                if velo['east'][i][1] < low:
                    velo['east'][i][1] = low

            # North
            n = len(velo['north'])
            for i in range(n):
                if velo['north'][i][1] < low:
                    velo['north'][i][1] = low

            # Up
            n = len(velo['up'])
            for i in range(n):
                if velo['up'][i][1] < low:
                    velo['up'][i][1] = low

        # All done
        return

    def substitute_errors(self, gps):
        '''
        Takes the errors of a gps object and substitute these to the errors in place.
        Args:
            * gps      : gps object with errors OK.
        '''

        for sta in self.station:

            # Get station
            stn = self.data[sta]

            # Check if attribute
            if not hasattr(stn, 'velo'):
                velo = stn.spitsopacvelocity()
                stn.velo = velo

            # Get the velo dictionary
            velo = stn.velo

            # Modify the errors
            i = np.flatnonzero(gps.station == stn.name)
            if len(i)>0:
                i = i[0]
                # Get the values
                err_e = gps.err_enu[i, 0]
                err_n = gps.err_enu[i, 1]
                err_u = gps.err_enu[i, 2]
                # Store those
                n = len(velo['east'])
                for i in range(n):
                    velo['east'][i][1] = err_e
                n = len(velo['north'])
                for i in range(n):
                    velo['north'][i][1] = err_n
                n = len(velo['up'])
                for i in range(n):
                    velo['up'][i][1] = err_u

        # All done
        return

    def plotstation(self, station):
        '''
        Plots the time series for a station.
        Args:
            * station       : Name of the station.
        '''

        # check first
        if self.data is not None:
            self.data[sta].plot()

        # All done
        return

#EOF
