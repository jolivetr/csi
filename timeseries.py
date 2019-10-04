''' 
A class that deals with time series of one variable.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import sys

# Personal
from .functionfit import functionfit
from .tidalfit import tidalfit
from .SourceInv import SourceInv

class timeseries(SourceInv):

    '''
    A class that handles generic time series

    Args:
       * name      : Name of the dataset.

    Kwargs:
       * utmzone   : UTM zone  (optional, default=None)
       * lon0      : Longitude of the center of the UTM zone
       * lat0      : Latitude of the center of the UTM zone
       * ellps     : ellipsoid (optional, default='WGS84')
       * verbose   : Talk to me 

    '''

    def __init__(self, name, utmzone=None, verbose=True, lon0=None, lat0=None, ellps='WGS84'):

        # base class ini
        super(timeseries, self).__init__(name, 
                                         utmzone=utmzone,
                                         lon0=lon0, lat0=lat0,
                                         ellps=ellps)

        # Set things
        self.name = name
        self.dtype = 'timeseries'
 
        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize Time Series {}".format(self.name))
        self.verbose = verbose
    
        # All done
        return

    def initialize(self, time=None, start=None, end=None, increment=1):
        '''
        Initialize the time series.

        Kwargs:
            * time          : list of datetime instances
            * start         : datetime instance of the first period
            * end           : datetime instance of the ending period
            * increment     : increment of time between periods

        Returns:
            * None
        '''

        # check start and end
        if (start.__class__ is float) or (start.__class__ is int) :
            st = dt.datetime(start, 1, 1)
        if (start.__class__ is list):
            if len(start) == 1:
                st = dt.datetime(start[0], 1, 1)
            elif len(start) == 2:
                st = dt.datetime(start[0], start[1], 1)
            elif len(start) == 3:
                st = dt.datetime(start[0], start[1], start[2])
            elif len(start) == 4:
                st = dt.datetime(start[0], start[1], start[2], 
                                 start[3])
            elif len(start) == 5:
                st = dt.datetime(start[0], start[1], start[2], 
                                 start[3], start[4])
            elif len(start) == 6:
                st = dt.datetime(start[0], start[1], start[2], 
                                 start[3], start[4], start[5])
        if start.__class__ is dt.datetime:
            st = start

        if (end.__class__ is float) or (end.__class__ is int) :
            ed = dt.datetime(np.int(end), 1, 1)
        if (end.__class__ is list):
            if len(end) == 1:
                ed = dt.datetime(end[0], 1, 1)
            elif len(end) == 2:
                ed = dt.datetime(end[0], end[1], 1)
            elif len(end) == 3:
                ed = dt.datetime(end[0], end[1], end[2])
            elif len(end) == 4:
                ed = dt.datetime(end[0], end[1], end[2], end[3])
            elif len(end) == 5:
                ed = dt.datetime(end[0], end[1], end[2], end[3], end[4])
            elif len(end) == 6:
                ed = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5])
        if end.__class__ is dt.datetime:
            ed = end

        # Initialize a time vector
        if end is not None:
            delta = ed - st
            delta_sec = np.int(np.floor(delta.days * 24 * 60 * 60 + delta.seconds))
            time_step = np.int(np.floor(increment * 24 * 60 * 60))
            self.time = [st + dt.timedelta(0, t) \
                    for t in range(0, delta_sec, time_step)]
        if time is not None:
            self.time = time

        # Values and errors
        self.value = np.zeros((len(self.time),))
        self.error = np.zeros((len(self.time),))
        self.synth = None        

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

    def readAscii(self, infile, header=0):
        '''
        Reads from an ascii file. Format of the file is

        +------+-------+-----+------+-----+--------+-------+----------------+
        | year | month | day | hour | min | second | value | err (optional) |
        +------+-------+-----+------+-----+--------+-------+----------------+

        Args:
            * infile    : Input file (ascii)
        
        Kwargs:
            * header    : length of the file header

        Returns:
            * None
        '''

        # Read file
        fin = open(infile, 'r')
        Lines = fin.readlines()
        fin.close()

        # Initialize things
        time = []
        value = []
        error = []

        # Loop 
        for i in range(header, len(Lines)):
            tmp = Lines[i].split()
            yr = np.int(tmp[0])
            mo = np.int(tmp[1])
            da = np.int(tmp[2])
            hr = np.int(tmp[3])
            mi = np.int(tmp[4])
            sd = np.int(tmp[5])
            time.append(dt.datetime(yr, mo, da, hr, mi, sd))
            value.append(np.float(tmp[6]))
            if len(tmp)>7:
                error.append(np.float(tmp[7]))
            else:
                error.append(0.0)

        # arrays
        self.time = time
        self.value = np.array(value)
        self.error = np.array(error)

        # Sort 
        self.SortInTime()

        # All done
        return

    def checkNaNs(self):
        '''
        Returns the index of NaNs

        Returns:
            * numpy array of integers
        '''

        # All done
        return np.flatnonzero(np.isnan(self.value))

    def removePoints(self, indexes):
        '''
        Removes the points from the time series

        Args:
            * indexes:  Indexes of the poitns to remove

        Returns:
            * None
        '''

        self.value = np.delete(self.value, indexes)
        self.error = np.delete(self.error, indexes)
        self.time = np.delete(np.array(self.time), indexes).tolist()

        # All done
        return

    def SortInTime(self):
        '''
        Sort ascending in time.

        Returns:
            * None
        '''

        # argsort
        u = np.argsort(self.time)

        # Sort
        self.time = [self.time[i] for i in u]
        self.value = self.value[u]
        self.error = self.error[u]

        # All done
        return

    def trimTime(self, start, end=dt.datetime(2100, 1, 1)):
        '''
        Keeps the data between start and end. start and end are 2 datetime.datetime objects.

        Args:
            * start     : datetime.datetime object

        Kwargs:
            * end       : datetime.datetime object

        Returns:
            * None
        '''

        # Assert
        assert type(start) is dt.datetime, 'Starting date must be datetime.datetime instance'
        assert type(end) is dt.datetime, 'Ending date must be datetime.datetime instance'

        # Get indexes
        u1 = np.flatnonzero(np.array(self.time)>=start)
        u2 = np.flatnonzero(np.array(self.time)<=end)
        u = np.intersect1d(u1, u2)

        # Keep'em
        self._keepDates(u)

        # All done
        return

    def adddata(self, time, values=None, std=None):
        '''
        Augments the time series

        Args:
            * time      : list of datetime objects

        Kwargs:
            * values    : list array or None
            * std       : list array or None

        Returns:
            * None
        '''

        # List
        if type(time) is not list:
            time = list(time)

        # Check 
        if values is not None:
            assert len(time)==len(values), 'Values size inconsistent: {}/{}'.format(len(time), len(values))
        else:
            values = np.zeros((len(time),))
        if std is not None:
            assert len(time)==len(values), 'Std size inconsistent: {}/{}'.format(len(time), len(std))
        else:
            std = np.zeros((len(time),))

        # Augment
        self.time += time
        self.values = np.append(self.values, values)
        self.std = np.append(self.std, std)

        # Sort
        self.SortInTime()

        # All done
        return


    def addPointInTime(self, time, value=0.0, std=0.0):
        '''
        Augments the time series by one point.

        Args:
            * time      : datetime.datetime object

        Kwargs:
            * value     : Value of the time series at time {time}
            * std       : Uncertainty at time {time}
        '''

        # Find the index
        u = 0
        t = self.time[u]
        while t<time and u<len(self.time):
            t = self.time[u]
            u += 1

        # insert
        self.time.insert(u, time)
        self.value = np.insert(self.value, u, value)
        self.error = np.insert(self.error, u, std)
        
        # All done
        return

    def computeDoubleDifference(self):
        '''
        Compute the derivative of the TS with a central difference scheme.

        Returns:
            * None. Results is stored in self.derivative
        '''

        # Get arrays
        up = self.value[2:]
        do = self.value[:-2]
        tup = self.time[2:]
        tdo = self.time[:-2]

        # Compute
        self.derivative = np.zeros((len(self.time),))
        timedelta = np.array([(tu-td).total_seconds() for tu,td in zip(tup, tdo)])
        self.derivative[1:-1] = (up - do)/timedelta

        # First and last
        self.derivative[0] = (self.value[1] - self.value[0])/(self.time[1] - self.time[0]).total_seconds()
        self.derivative[-1] = (self.value[-2] - self.value[-1])/(self.time[-2] - self.time[-1]).total_seconds()

        # All Done
        return

    def smoothGlitches(self, biggerThan=999999., smallerThan=-999999., interpNum=5, interpolation='linear'):
        '''
        Removes the glitches and replace them by a value interpolated on interpNum points.

        Kwargs:
            * biggerThan    : Values higher than biggerThan are glitches.
            * smallerThan   : Values smaller than smallerThan are glitches.
            * interpNum     : Number of points to take before and after the glicth to predict its values.
            * interpolation : Interpolation method.

        Returns:
            * None
        '''

        # Find glitches
        u = np.flatnonzero(self.value>biggerThan)
        d = np.flatnonzero(self.value<smallerThan)
        g = np.union1d(u,d).tolist()

        # Loop on glitches
        while len(g)>0:
            
            # Get index
            iG = g.pop()

            # List
            iGs = [iG]

            # Check next ones
            go = False
            if len(g)>0:
                if (iG-g[-1]<interpNum):
                    go = True
            while go:
                iG = g.pop()
                iGs.append(iG)
                go = False
                if len(g)>0:
                    if (iG-g[-1]<interpNum):
                        go = True

            # Sort
            iGs.sort()

            # Make a list of index to use for interpolation
            iMin = max(0, iGs[0]-interpNum)
            iMax = min(iGs[-1]+interpNum+1, self.value.shape[0])
            iIntTmp = range(iMin, iMax)
            iInt = []
            for i in iIntTmp:
                if i not in iGs:
                    iInt.append(i)
            iInt.sort()
            
            # Build the interpolator
            time = np.array([(self.time[t]-self.time[iInt[0]]).total_seconds() for t in iInt])
            value = np.array([self.value[t] for t in iInt])
            interp = sciint.interp1d(np.array(time), self.value[iInt], kind=interpolation)

            # Interpolate
            self.value[iGs] = np.array([interp((self.time[t]-self.time[iInt[0]]).total_seconds()) for t in iGs])

        # All done
        return

    def removeMean(self, start=None, end=None):
        '''
        Removes the mean between start and end.

        Kwargs:
            * start : datetime.datetime object. If None, takes the first point of the time series
            * end   : datetime.datetime object. If None, takes the last point of the time series

        Returns:
            * None. Attribute {value} is directly modified.
        '''

        # Start end
        if start is None:
            start = self.time[0]
        if end is None:
            end = self.time[-1]

        # Get index
        u1 = np.flatnonzero(np.array(self.time)>=start)
        u2 = np.flatnonzero(np.array(self.time)<=end)
        u = np.intersect1d(u1, u2)

        # Get Mean
        mean = np.nanmean(self.value[u])

        # Correct
        self.value -= mean

        # All Done
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
            * None. Model vector is stored in the {m} attribute
        '''

        # Do the fit
        fit = functionfit(function, verbose=self.verbose)
        fit.doFit(self, m0, solver=solver, iteration=iteration, tol=tol)

        # Do the prediction
        fit.predict(self)

        # Save
        self.m = fit.m

        # All done
        return

    def fitTidalConstituents(self, steps=None, linear=False, tZero=dt.datetime(2000, 1, 1), chunks=None, cossin=False, constituents='all'):
        '''
        Fits tidal constituents on the time series.

        Kwargs:
            * steps     : list of datetime instances to add step functions in the estimation process.
            * linear    : estimate a linear trend.
            * tZero     : origin time (datetime instance).
            * chunks    : List [ [start1, end1], [start2, end2]] where the fit is performed.
            * cossin    : Add a cosine+sine term in the procedure.
            * constituents  : list of tidal constituents to include (default is all). For a list, go check tidalfit class

        Returns:
            * None
        '''

        # Initialize a tidalfit
        tf = tidalfit(constituents=constituents, linear=linear, steps=steps, cossin=cossin)

        # Fit the constituents
        tf.doFit(self, tZero=tZero, chunks=chunks)

        # Predict the time series
        if steps is not None:
            sT = True
        else:
            sT = False
        tf.predict(self,constituents=constituents, linear=linear, steps=sT, cossin=cossin)

        # All done
        return

    def getOffset(self, date1, date2, nodate=np.nan, data='data'):
        '''
        Get the offset between date1 and date2. 

        Args:
            * date1       : datetime object
            * date2       : datetime object

        Kwargs:
            * nodate      : Value to be returned in case no value is available
            * data        : can be 'data' or 'std'

        Returns:
            * float
        '''

        # Get the indexes
        u1 = np.flatnonzero(np.array(self.time)==date1)
        u2 = np.flatnonzero(np.array(self.time)==date2)

        # Check
        if len(u1)==0:
            return nodate, nodate, nodate
        if len(u2)==0:
            return nodate, nodate, nodate

        # Select 
        if data in ('data'):
            value = self.value
        elif data in ('std'):
            value = self.error

        # all done
        return value[u2] - value[u1]

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
        fout.write('# Time | value | std \n')

        # Loop over the dates
        for i in range(len(self.time)-1):
            t = self.time[i].isoformat()
            e = self.value[i]
            es = self.std[i]
            fout.write('{} {} {} \n'.format(t, e, es))
            if steplike:
                e = self.value[i+1]
                es = self.std[i+1]
                fout.write('{} {} {} \n'.format(t, e, es))

        t = self.time[i].isoformat()
        e = self.value[i]
        es = self.std[i]
        fout.write('{} {} {} \n'.format(t, e, es))

        # Done 
        fout.close()

        # All done
        return

    def findZeroIntersect(self, data='data'):
        '''
        Returns all the points just before the function crosses 0.

        Kwargs:
            * data      : Can be 'data', 'synth' or 'derivative'.

        Returns:
            * None
        '''

        # Get the good data
        if data is 'data':
            v = self.value
        elif data is 'synth':
            v = self.synth
        elif data is 'derivative':
            v = self.derivative

        # List 
        indexes = []

        # Loop
        for i in xrange(len(v)-1):
            if (v[i]>0. and v[i+1]<0.) or (v[i]<0. and v[i+1]>0.):
                indexes.append(i)

        # All done
        return indexes

    def plot(self, figure=1, styles=['.r'], show=True, data='data', subplot=None):
        '''
        Plots the time series.

        Args:
            * figure  :   Figure id number (default=1)
            * styles  :   List of styles (default=['.r'])
            * show    :   Show to me (default=True)
            * data    :   can be 'data', 'derivative', 'synth' or a list of those
            * subplot :   axes instance to be used for plotting. If None, creates a new one

        Returns:
            * None
        '''

        # Get values
        if type(data) is str:
            data = [data]

        # iterate
        values = []
        for d in data:
            if d in ('data'):
                v = self.value
            elif d in ('derivative'):
                v = self.derivative
            elif d in ('synth'):
                v = self.synth
            elif d in ('res'):
                v = self.value-self.synth
            else:
                print('Unknown component to plot')
                return
            values.append(v)

        # Create a figure
        if (figure=='new') or type(figure) is int:
            fig = plt.figure(figure)
        else:
            fig = figure

        # Create axes
        if subplot is not None:
            ax = subplot
        else:
            ax = fig.add_subplot(111)

        # Plot ts
        for v,style in zip(values, styles):
            u = np.argsort(self.time)
            ax.plot(np.array(self.time)[u], np.array(v)[u], style)

        # show
        if show:
            plt.show()

        # All done
        return

    def reference2timeseries(self, timeseries):
        '''
        Removes to another gps timeseries the difference between self and timeseries

        Args:
            * timeseries        : Another timeseries

        Returns:
            * float
        '''
        
        # Mean 
        difference = 0.
        elements = 0

        # Find the common dates and compute the difference
        for d, date in enumerate(self.time):
            val = timeseries.value[timeseries.time.index(date)]
            assert len(val)<=1, 'Multiple dates for a measurement'
            if len(val)>0:
                diff = self.value[d] - val
                if np.isfinite(diff):
                    difference += self.value[d] - val
                    elements += 1

        # Average the difference
        if elements>0:
            difference /= float(elements)

        # Remove the difference to the values
        timeseries.value += difference

        # All done
        return difference

#PRIVATE EMTHODS

    def _keepDates(self, u):
        '''
        Keeps the dates corresponding to index u.
        '''

        self.time = [self.time[i] for i in u]
        self.value = self.value[u]
        self.error = self.error[u]
        if hasattr(self, 'synth') and self.synth is not None:
            self.synth = self.synth[u]

        # All done
        return

    def _deleteDates(self, u):
        '''
        Remove the dates corresponding to index u.
        '''

        # Delete stuff
        self.time = np.delete(np.array(self.time), u).tolist()
        self.value = np.delete(self.value, u)
        self.error = np.delete(self.error, u)
        if hasattr(self, 'synth'):
            self.synth = np.delete(self.synth, u)

        # All done
        return

#EOF
