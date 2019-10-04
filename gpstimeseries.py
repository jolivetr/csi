''' 
A class that deals with gps time series.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import sys, os

# Personal
from .timeseries import timeseries
from .SourceInv import SourceInv

class gpstimeseries(SourceInv):

    '''
    A class that handles a time series of gps data

    Args:
       * name      : Name of the dataset.

    Kwargs:
       * utmzone   : UTM zone  (optional, default=None)
       * lon0      : Longitude of the center of the UTM zone
       * lat0      : Latitude of the center of the UTM zone
       * ellps     : ellipsoid (optional, default='WGS84')
       * verbose   : Speak to me (default=True)

    '''

    def __init__(self, name, utmzone=None, verbose=True, lon0=None, lat0=None, ellps='WGS84'):

        # Set things
        self.name = name
        self.dtype = 'gpstimeseries'
 
        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize GPS Time Series {}".format(self.name))
        self.verbose = verbose

        # Base class init
        super(gpstimeseries,self).__init__(name,
                                           utmzone=utmzone,
                                           lon0=lon0,
                                           lat0=lat0, 
                                           ellps=ellps)

        # All done
        return

    def lonlat2xy(self):
        '''
        Pass the position into the utm coordinate system.

        Returns:
            * None
        '''

        x, y = self.putm(self.lon, self.lat)
        self.x = x/1000.
        self.y = y/1000.

        # All done
        return

    def xy2lonlat(self):
        '''
        Pass the position from utm to lonlat.

        Returns:
            * None
        '''

        lon, lat = self.putm(x*1000., y*1000.)
        self.lon = lon
        self.lat = lat

        # all done
        return

    def read_from_file(self, filename, verbose=False):
        '''
        Reads the time series from a file which has been written by write2file

        Args:
            * filename      : name of the file

        Kwargs:
            * verbose       : talk to me

        Returns:
            * None
        '''

        # Open, read, close file
        fin = open(filename, 'r')
        Lines = fin.readlines() 
        fin.close()

        # Create values
        time = []
        east = []; north = []; up = []
        stdeast = []; stdnorth = []; stdup = []

        # Read these
        for line in Lines:
            values = line.split()
            if values[0][0] == '#':
                continue
            isotime = values[0]
            year  = int(isotime[:4])
            month = int(isotime[5:7])
            day   = int(isotime[8:10])
            hour = int(isotime[11:13])
            mins = int(isotime[14:16])
            secd = int(isotime[17:19])
            time.append(dt.datetime(year, month, day, hour, mins, secd))
            east.append(float(values[1]))
            north.append(float(values[2]))
            up.append(float(values[3]))
            stdeast.append(float(values[4]))
            stdnorth.append(float(values[5]))
            stdup.append(float(values[6]))

        # Initiate some timeseries
        self.north = timeseries('North',
                                utmzone=self.utmzone, 
                                lon0=self.lon0, lat0=self.lat0, 
                                ellps=self.ellps, verbose=verbose)
        self.east = timeseries('East', 
                               utmzone=self.utmzone, 
                               lon0=self.lon0, 
                               lat0=self.lat0, 
                               ellps=self.ellps, verbose=verbose)
        self.up = timeseries('Up', 
                             utmzone=self.utmzone, 
                             lon0=self.lon0, lat0=self.lat0, 
                             ellps=self.ellps, verbose=verbose)

        # Set time
        self.time = time
        self.north.time = self.time
        self.east.time = self.time
        self.up.time = self.time

        # Set values
        self.north.value = np.array(north)
        self.north.synth = None        
        self.north.error = np.array(stdnorth)
        self.east.value = np.array(east)
        self.east.synth = None        
        self.east.error = np.array(stdeast)
        self.up.value = np.array(up)
        self.up.synth = None        
        self.up.error = np.array(stdup)

        # All done
        return

    def read_from_renoxyz(self, filename, verbose=False):
        '''
        Reads the time series from a file which has been downloaded on
        http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html

        This was true on 2015.

        Args:
            * filename      : name of file

        Kwargs:
            * verbose       : talk to me

        Returns:
            * None
        '''

        # Get months description
        from .csiutils import months

        # Open, read, close file
        fin = open(filename, 'r')
        Lines = fin.readlines() 
        fin.close()

        # Create values
        time = []
        east = []; north = []; up = []
        stdeast = []; stdnorth = []; stdup = []

        # Read these
        for line in Lines:
            values = line.split()
            if values[0][0] == '#':
                continue
            isotime = values[1]
            yd = int(isotime[:2])
            if yd<80: year = yd + 2000
            if yd>=90: year = yd + 1900
            month = months[isotime[2:5]]
            day   = int(isotime[5:7])
            time.append(dt.datetime(year, month, day, 0, 0, 0))
            east.append(float(values[3]))
            north.append(float(values[4]))
            up.append(float(values[5]))
            stdeast.append(float(values[6]))
            stdnorth.append(float(values[7]))
            stdup.append(float(values[8]))

        # Initiate some timeseries
        self.north = timeseries('North',
                                utmzone=self.utmzone, 
                                lon0=self.lon0, lat0=self.lat0, 
                                ellps=self.ellps, verbose=verbose)
        self.east = timeseries('East', 
                               utmzone=self.utmzone, 
                               lon0=self.lon0, 
                               lat0=self.lat0, 
                               ellps=self.ellps, verbose=verbose)
        self.up = timeseries('Up', 
                             utmzone=self.utmzone, 
                             lon0=self.lon0, lat0=self.lat0, 
                             ellps=self.ellps, verbose=verbose)

        # Set time
        self.time = time
        self.north.time = self.time
        self.east.time = self.time
        self.up.time = self.time

        # Set values
        self.north.value = np.array(north)
        self.north.synth = None        
        self.north.error = np.array(stdnorth)
        self.east.value = np.array(east)
        self.east.synth = None        
        self.east.error = np.array(stdeast)
        self.up.value = np.array(up)
        self.up.synth = None        
        self.up.error = np.array(stdup)

        # All done
        return

    def read_from_JPL(self, filename):
        '''
        Reads the time series from a file which has been sent from JPL.
        Format is a bit awkward and you should not see that a lot.
        Look inside the code to find out...
        '''

        # Open, read, close file
        fin = open(filename, 'r')
        Lines = fin.readlines() 
        fin.close()

        # Create values
        time = []
        east = []; north = []; up = []
        stdeast = []; stdnorth = []; stdup = []

        # Read these
        for line in Lines:
            values = line.split()
            time.append(dt.datetime(int(values[11]), 
                                    int(values[12]),
                                    int(values[13]),
                                    int(values[14]),
                                    int(values[15]),
                                    int(values[16])))
            east.append(float(values[1]))
            north.append(float(values[2]))
            up.append(float(values[3]))
            stdeast.append(float(values[4]))
            stdnorth.append(float(values[5]))
            stdup.append(float(values[6]))

        # Initiate some timeseries
        self.north = timeseries('North', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.east = timeseries('East', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.up = timeseries('Up', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)

        # Set time
        self.time = time
        self.north.time = self.time
        self.east.time = self.time
        self.up.time = self.time

        # Set values
        self.north.value = np.array(north)
        self.north.synth = None
        self.north.error = np.array(stdnorth)
        self.east.value = np.array(east)
        self.east.synth = None        
        self.east.error = np.array(stdeast)
        self.up.value = np.array(up)
        self.up.synth = None        
        self.up.error = np.array(stdup)

        # All done
        return

    def read_from_sql(self, filename, 
                      tables={'e': 'east', 'n': 'north', 'u': 'up'},
                      sigma={'e': 'sigma_east', 'n': 'sigma_north', 'u': 'sigma_up'},
                      factor=1.):
        '''
        Reads the East, North and Up components of the station in a sql file.
        This follows the organization of M. Simons' group at Caltech. The sql file 
        has tables called as indicated in the dictionary tables and sigma.

        This method requires pandas and sqlalchemy

        Args:
            * filename  : Name of the sql file

        Kwargs:
            * tables    : Dictionary of the names of the table for the east, north and up displacement time series
            * sigma     : Dictionary of the names of the tables for the east, north and up uncertainties time series
            * factor    : scaling factor

        Returns:
            * None
        '''

        # Import necessary bits
        try:
            import pandas
            from sqlalchemy import create_engine
        except:
            assert False, 'Could not import pandas or sqlalchemy...'

        # Open the file
        assert os.path.exists(filename), 'File cannot be found'
        engine = create_engine('sqlite:///{}'.format(filename))
        east = pandas.read_sql_table(tables['e'], engine)
        north = pandas.read_sql_table(tables['n'], engine)
        up = pandas.read_sql_table(tables['u'], engine)
        sigmaeast = pandas.read_sql_table(sigma['e'], engine)
        sigmanorth = pandas.read_sql_table(sigma['n'], engine)
        sigmaup = pandas.read_sql_table(sigma['u'], engine)

        # Find the time
        assert (east['DATE'].values==north['DATE'].values).all(), \
                'There is something weird with the timeline of your station'
        ns = 1e-9 # Number of nanoseconds in a second
        self.time = [dt.datetime.utcfromtimestamp(t.astype(int)*ns) \
                              for t in east['DATE'].values]

        # Initiate some timeseries
        self.north = timeseries('North', utmzone=self.utmzone, verbose=self.verbose,
                                lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.east = timeseries('East', utmzone=self.utmzone, verbose=self.verbose,
                               lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)
        self.up = timeseries('Up', utmzone=self.utmzone, verbose=self.verbose,
                             lon0=self.lon0, lat0=self.lat0, ellps=self.ellps)

        # set time
        self.east.time = self.time
        self.north.time = self.time
        self.up.time = self.time

        # Set the values
        self.north.value = north[self.name].values*factor
        self.north.synth = None        
        self.north.error = sigmanorth[self.name].values*factor
        self.east.value = east[self.name].values*factor
        self.east.synth = None        
        self.east.error = sigmaeast[self.name].values*factor
        self.up.value = up[self.name].values*factor
        self.up.synth = None
        self.up.error = sigmaup[self.name].values*factor
        
        # All done
        return

    def read_from_caltech(self, filename):
        '''
        Reads the data from a time series file from CalTech (Avouac's group).
        Time is in decimal year...

        Args:
            * filename      : Input file

        Returns:
            * None
        '''

        # Open, read, close file
        fin = open(filename, 'r')
        Lines = fin.readlines() 
        fin.close()

        # Create values
        time = []
        east = []; north = []; up = []
        stdeast = []; stdnorth = []; stdup = []

        # Read these
        for line in Lines:
            values = line.split()
            year = np.floor(float(values[0]))
            doy = np.floor((float(values[0])-year)*365.24).astype(int)
            time.append(dt.datetime.fromordinal(dt.datetime(year.astype(int), 1, 1).toordinal() + doy))
            east.append(float(values[1]))
            north.append(float(values[2]))
            up.append(float(values[3]))
            stdeast.append(float(values[4]))
            stdnorth.append(float(values[5]))
            stdup.append(float(values[6]))

        # Initiate some timeseries
        self.north = timeseries('North', utmzone=self.utmzone, 
                                lon0=self.lon0, lat0=self.lat0, ellps=self.ellps,
                                verbose=self.verbose)
        self.east = timeseries('East', utmzone=self.utmzone, 
                               lon0=self.lon0, lat0=self.lat0, ellps=self.ellps,
                               verbose=self.verbose)
        self.up = timeseries('Up', utmzone=self.utmzone, 
                             lon0=self.lon0, lat0=self.lat0, ellps=self.ellps,
                             verbose=self.verbose)

        # Set time
        self.time = time
        self.north.time = self.time
        self.east.time = self.time
        self.up.time = self.time

        # Set values
        self.north.value = np.array(north)
        self.north.synth = None
        self.north.error = np.array(stdnorth)        
        self.east.value = np.array(east)
        self.east.synth = None        
        self.east.error = np.array(stdeast)
        self.up.value = np.array(up)
        self.up.synth = None                
        self.up.error = np.array(stdup)


        # All done
        return
    
    def removeNaNs(self):
        '''
        Remove NaNs in the time series

        Returns:
            * None
        '''

        # Get the indexes
        east = self.east.checkNaNs()
        north = self.north.checkNaNs()
        up = self.north.checkNaNs()

        # check
        enu = np.union1d(east, north)
        enu = np.union1d(enu, up)

        # Remove these guys
        self.east.removePoints(enu)
        self.north.removePoints(enu)
        self.up.removePoints(enu)

        # All done
        return

    def initializeTimeSeries(self, time=None, start=None, end=None, interval=1, los=False):
        '''
        Initializes the time series by creating whatever is necessary.

        Kwargs:
            * time              Time vector
            * starttime:        Begining of the time series.
            * endtime:          End of the time series.
            * interval:         In days.
            * los:              True/False
        '''
    
        # North-south time series
        self.north = timeseries('North', 
                                utmzone=self.utmzone, 
                                lon0=self.lon0, 
                                lat0=self.lat0, 
                                ellps=self.ellps, 
                                verbose=self.verbose)
        self.north.initialize(time=time, 
                              start=start, end=end, increment=interval)

        # East-west time series
        self.east = timeseries('East', 
                               utmzone=self.utmzone, 
                               lon0=self.lon0, 
                               lat0=self.lat0, 
                               ellps=self.ellps, 
                               verbose=self.verbose)
        self.east.initialize(time=time, 
                             start=start, end=end, increment=interval)

        # Vertical time series 
        self.up = timeseries('Up', 
                             utmzone=self.utmzone, 
                             lon0=self.lon0, 
                             lat0=self.lat0, 
                             ellps=self.ellps, 
                             verbose=self.verbose)
        self.up.initialize(time=time, 
                           start=start, end=end, increment=interval)

        # LOS time series
        if los:
            self.los = timeseries('LOS', 
                                  utmzone=self.utmzone, 
                                  lon0=self.lon0, 
                                  lat0=self.lat0, 
                                  ellps=self.ellps,
                                  verbose=self.verbose)
            self.los.initialize(time=time, 
                                start=start, end=end, increment=interval)

        # Time
        self.time = self.north.time

        # All done
        return

    def trimTime(self, start, end=dt.datetime(2100,1,1)):
        '''
        Keeps the epochs between start and end

        Args:
            * start: starting date (datetime instance)
        
        Kwargs:
            * end: ending date (datetime instance)

        Returns:
            * None

        '''
        
        # Trim
        self.north.trimTime(start, end=end)
        self.east.trimTime(start, end=end)
        self.up.trimTime(start, end=end)

        # Fix time
        self.time = self.up.time

        # All done
        return

    def addPointInTime(self, time, east=0.0, north=0.0, up = 0.0, std_east=0.0, std_north=0.0, std_up=0.0):
        '''
        Augments the time series by one point.

        Args:
            * time: datetime object.a

        Kwargs:
            * east, north, up   : Time series values. Default is 0
            * std_east, std_north, std_up: Uncertainty values. Default is 0

        Returns:
            * None
        '''

        # insert
        self.east.addPointInTime(time, value=east, std=std_east)
        self.north.addPointInTime(time, value=north, std=std_north)
        self.up.addPointInTime(time, value=up, std=std_up)
 
        # Time vector
        self.time = self.up.time

        # All done
        return

    def removePointsInTime(self, u):
        '''
        Remove points from the time series.

        Args:
            * u         : List or array of indexes to remove

        Returns:
            * None
        '''

        # Delete
        self.east._deleteDates(u)
        self.north._deleteDates(u)
        self.up._deleteDates(u)

        # Time
        self.time = self.up.time

        # All done
        return

    def fitFunction(self, function, m0, solver='L-BFGS-B', iteration=1000, tol=1e-8):
        '''
        Fits a function to the timeseries

        Args:
            * function  : Prediction function, 
            * m0        : Initial model

        Kwargs:
            * solver    : Solver type (see list of solver in scipy.optimize.minimize)
            * iteration : Number of iteration for the solver
            * tol       : Tolerance

        Returns:   
            * None. Parameters are stored in attribute {m} of each time series object
        '''

        # Do it for the three components
        self.east.fitFunction(function, m0, solver=solver, iteration=iteration, tol=tol)
        self.north.fitFunction(function, m0, solver=solver, iteration=iteration, tol=tol)
        self.up.fitFunction(function, m0, solver=solver, iteration=iteration, tol=tol)

        # All done
        return

    def fitTidalConstituents(self, steps=None, linear=False, tZero=dt.datetime(2000, 1, 1), 
            chunks=None, cossin=False, constituents='all'):
        '''
        Fits tidal constituents on the time series.

        Args:
            * steps     : list of datetime instances to add step functions in the estimation process.
            * linear    : estimate a linear trend.
            * tZero     : origin time (datetime instance).
            * chunks    : List [ [start1, end1], [start2, end2]] where the fit is performed.
            * cossin    : Add an  extra cosine+sine term (weird...)
            * constituents: list of constituents to fit (default is 'all')

        Returns:
            * None
        '''

        # Do it for each time series
        self.north.fitTidalConstituents(steps=steps, linear=linear, tZero=tZero, 
                chunks=chunks, cossin=cossin, constituents=constituents)
        self.east.fitTidalConstituents(steps=steps, linear=linear, tZero=tZero, 
                chunks=chunks, cossin=cossin, constituents=constituents)
        self.up.fitTidalConstituents(steps=steps, linear=linear, tZero=tZero, 
                chunks=chunks, cossin=cossin, constituents=constituents)

        # All done
        return

    def getOffset(self, date1, date2, nodate=np.nan, data='data'):
        '''
        Get the offset between date1 and date2.
        If the 2 dates are not available, returns NaN.

        Args:
            * date1       : datetime object
            * date2       : datetime object

        Kwargs:
            * data        : can be 'data' or 'std'
            * nodate      : If there is no date, return this value

        Returns:
            * tuple of floats
        '''

        # Get offsets
        east = self.east.getOffset(date1, date2, nodate=nodate, data=data)
        north = self.north.getOffset(date1, date2, nodate=nodate, data=data)
        up = self.up.getOffset(date1, date2, nodate=nodate, data=data)

        # all done
        return east, north, up

    def write2file(self, outfile, steplike=False):
        '''
        Writes the time series to a file.

        Args:   
            * outfile   : output file.

        Kwargs:
            * steplike  : doubles the output each time so that the plot looks like steps.

        Returns:
            * None
        '''

        # Open the file
        fout = open(outfile, 'w')
        fout.write('# Time | east | north | up | east std | north std | up std \n')

        # Loop over the dates
        for i in range(len(self.time)-1):
            t = self.time[i].isoformat()
            e = self.east.value[i]
            n = self.north.value[i]
            u = self.up.value[i]
            es = self.east.error[i]
            ns = self.north.error[i]
            us = self.up.error[i]
            if hasattr(self, 'los'):
                lo = self.los.value[i]
            else:
                lo = None
            fout.write('{} {} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us, lo))
            if steplike:
                e = self.east.value[i+1]
                n = self.north.value[i+1]
                u = self.up.value[i+1]
                es = self.east.error[i+1]
                ns = self.north.error[i+1]
                us = self.up.error[i+1]
                fout.write('{} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us))

        if not steplike:
            i += 1
            t = self.time[i].isoformat()
            e = self.east.value[i]
            n = self.north.value[i]
            u = self.up.value[i]
            es = self.east.error[i]
            ns = self.north.error[i]
            us = self.up.error[i]
            if hasattr(self, 'los'):
                lo = self.los.value[i]
            else:
                lo = None
            fout.write('{} {} {} {} {} {} {} {} \n'.format(t, e, n, u, es, ns, us, lo))

        # Done 
        fout.close()

        # All done
        return

    def project2InSAR(self, los):
        '''
        Projects the time series of east, north and up displacements into the 
        line-of-sight given as argument

        Args:
            * los       : list of three component. L2-norm of los must be equal to 1

        Returns:
            * None. Results are stored in attribute {losvector}
        '''

        # Create a time series
        self.los = timeseries('LOS', 
                              utmzone=self.utmzone, 
                              lon0=self.lon0, 
                              lat0=self.lat0, 
                              ellps=self.ellps,
                              verbose=self.verbose)

        # Make sure los is an array
        if type(los) is list:
            los = np.array(los)

        # Get the values and project
        self.los.time = self.time
        self.los.value = np.dot(np.vstack((self.east.value, 
                                           self.north.value, 
                                           self.up.value)).T, 
                                los[:,np.newaxis]).reshape((len(self.time), ))
        self.los.error = np.dot(np.vstack((self.east.error, 
                                           self.north.error, 
                                           self.up.error)).T, 
                                           los[:,np.newaxis]).reshape((len(self.time),))

        # Save the los vector
        self.losvector = los

        # All done
        return

    def reference2timeseries(self, timeseries, verbose=True):
        '''
        Removes to another gps timeseries the difference between self and timeseries

        Args:
            * timeseries        : Another gpstimeseries

        Kwargs:
            * verbose           : Talk to me

        Returns:
            * None
        '''

        # Verbose
        if verbose:
            print('---------------------------------')
            print('Reference time series {} to {}'.format(timeseries.name, self.name))

        # Do the reference for all the timeseries in there 
        north = self.north.reference2timeseries(timeseries.north)
        string = 'North offset: {} \n'.format(north) 
        east = self.east.reference2timeseries(timeseries.east)
        string += 'East offset: {} \n'.format(east)
        up = self.up.reference2timeseries(timeseries.up)
        string += 'Up offset: {} \n'.format(up)
        if hasattr(self, 'los') and hasattr(timeseries, 'los'):
            los = self.los.reference2timeseries(timeseries.los)
            string += 'LOS offset: {} \n'.format(los)

        # verbose
        if verbose:
            print(string)

        # All done
        return

    def plot(self, figure=1, styles=['.r'], show=True, data='data'):
        '''
        Plots the time series.

        Kwargs:
            * figure  :   Figure id number (default=1)
            * styles  :   List of styles (default=['.r'])
            * show    :   Show to me (default=True)
            * data    :   What do you show (data, synth)

        Returns:
            * None
        '''

        # list 
        if type(data) is not list:
            data = [data]

        # Create a figure
        fig = plt.figure(figure)

        # Number of plots
        nplot = 311
        if hasattr(self, 'los'):
            nplot += 100

        # Create axes
        axnorth = fig.add_subplot(nplot)
        axeast = fig.add_subplot(nplot+1)
        axup = fig.add_subplot(nplot+2)
        if nplot > 350:
            axlos = fig.add_subplot(nplot+3)

        # Plot
        self.north.plot(figure=fig, subplot=axnorth, styles=styles, data=data, show=False)
        self.east.plot(figure=fig, subplot=axeast, styles=styles, data=data, show=False)
        self.up.plot(figure=fig, subplot=axup, styles=styles, data=data, show=False)
        if nplot > 350:
            self.los.plot(figure=fig, subplot=axlos, styles=styles, data=data, show=False)

        # show
        if show:
            plt.show()

        # All done
        return

#EOF
