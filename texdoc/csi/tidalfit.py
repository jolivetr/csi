
''' 
A class that offers a fit on a time series.

Written by R. Jolivet, June 2014.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import sys


class tidalfit(object):

    def __init__(self, constituents='all', linear=False, steps=None, cossin=False, verbose=True):
        '''
        Initialize a tidalfit object.
        Args:
            * constituents  : List of constituents to use. Can be 'all'.
            * linear        : Include a linear trend (default is False).
            * steps         : List of datetime instances to add step functions in the fit.
            * cossin        : Just add a cos and sin term to estimate
        '''
        
        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize a tidal fit object")
        
        # Tidal Periods in hours (source: wikipedia)
        self.tidePeriodDict = {'M2' : 12.4206012,       # Principal Lunar
                               'S2' : 12.0,             # Principal Solar
                               'N2' : 12.65834751,      # Lunar elliptic
                               'v2' : 12.62600509,      # Larger Lunar evectional 
                               'MU2': 12.8717576,       # Variational
                               '2N2': 12.90537297,      # Lunar elliptical semidiurnal 2nd order
                               'Lambda2': 12.22177348,  # Smaller lunar evectional
                               'T2' : 12.01644934,      # Larger Solar elliptic
                               'R2' : 11.98359564,      # Smaller Solar elliptic
                               'L2' : 12.19162085,      # Smalle lunar elliptic semidiurnal
                               'K2' : 11.96723606,      # Lunisolar
                               'K1' : 23.93447213,      # Lunisolar
                               'O1' : 25.81933871,      # Principal Lunar
                               'OO1': 22.30608083,      # Lunar diurnal
                               'S1' : 24.00000000,      # Solar diurnal
                               'M1' : 24.84120241,      # Smaller lunar elliptic diurnal
                               'J1' : 23.09848146,      # Smaller lunar elliptic diurnal
                               'Rho': 26.72305326,      # Larger lunar evectional diurnal
                               'P1' : 24.06588766,      # Principal Solar
                               'Q1' : 26.868350,        # Elliptic Lunar
                               '2Q1': 28.00621204,      # Larger elliptic diurnal
                               'Mf' : 327.8599387,      # Fortnightly
                               'Msf': 354.3670666,      # Lunisolar synodic fortnightly
                               'Mm' : 661.3111655,      # Monthly
                               'Ssa': 4383.076325,      # Solar semiannual
                               'Sa' : 8766.15265 }      # Solar annual

        # What periods do we want
        if type(constituents) is str:
            if constituents in ('All', 'all', 'ALL'):
                constituents = self.tidePeriodDict.keys()
        self.constituents = constituents
    
        # Constituents
        if cossin:
            self.constituents.append('cossin')
            self.tidePeriodDict['cossin'] = 1.0

        # Store things
        self.steps = steps
        self.linear = linear

        # All done
        return

    def doFit(self, timeseries, tZero=dt.datetime(2001, 1, 1), chunks=None):
        '''
        Performs the fit on the chunks of data specified in chunks.
        Args:
            * timeseries: Timeseries instance.
            * tZero     : Sets the origin time (datetime instance).
            * chunks    : if not None, provide a list: [[start1, end1], [start2, end2], ...[startn, endn]].
                          if None, takes all the data.
        '''

        # Sets tZero
        self.tZero = tZero

        # Build G and data
        self.buildG(timeseries, chunks=chunks, linear=self.linear, steps=self.steps, constituents=self.constituents)

        # Solve
        m, res, rank, s = np.linalg.lstsq(self.G, self.data, rcond=1e-8) 
        self.m = m
        self.res = res
        self.rank = rank
        self.s = s

        # Save Linear and Steps
        self.Offset = m[0]
        iP = 1
        if self.linear:
            self.Linear = m[1]
            iP += 1
        if self.steps is not None:
            self.Steps = []
            for step in self.steps:
                self.Steps.append(m[iP])
                iP += 1

        # Save Constituents
        self.Constituents = {}
        for constituent in self.constituents:
            self.Constituents[constituent] = m[iP:iP+2]
            iP += 2

        # All done
        return

    def buildG(self, timeseries, chunks=None, linear=False, steps=None, constituents='all', derivative=False):
        ''' 
        Builds the G matrix we will invert.
        Args:
            * timeseries    : Timeseries instance.
            * chunks        : List of chunks of dates [[start1, end1], [start2, end2], ...[startn, endn]]. 
            * linear        : True/False.
            * steps         : List of datetime to add steps.
            * constituents  : List of consituents.
        '''
        
        # Get things
        time = timeseries.time
        value = timeseries.value
        tZero = self.tZero
        
        # What periods do we want
        if type(constituents) is str:
            if constituents in ('All', 'all', 'ALL'):
                constituents = self.tidePeriodDict.keys()

        # Check
        if derivative:
            steps = None

        # Parameters
        nTide = 2*len(constituents)
        nStep = 0
        if steps is not None:
            nStep = len(steps)
        nLin = 0
        if linear:
            nLin = 1
        if derivative:
            add = 0
        else:
            add = 1
        self.nParam = add + nTide + nStep + nLin

        # Get the data indexes
        if chunks is None:
            u = range(len(time))
        else:
            u = []
            for chunk in chunks:
                u1 = np.flatnonzero(np.array(time)>=chunk[0])
                u2 = np.flatnonzero(np.array(time)<=chunk[1])
                uu = np.intersect1d(u1, u2)
                u.append(uu)
            u = np.array(u).flatten().tolist()

        # How many data
        nData = len(u)
        self.nData = nData

        # Initialize G
        G = np.zeros((self.nData, self.nParam))

        # Build time and data vectors
        time = time[u]
        Tvec = np.array([(time[i]-tZero).total_seconds()/(60.*60.*24.) for i in range(time.shape[0])])  # In Days
        self.data = value[u]

        # Constant term
        if not derivative:
            G[:,0] = 1.0
            iP = 1
        else:
            iP = 0

        # Linear?
        if linear:
            if not derivative:
                G[:,iP] = Tvec
            else:
                G[:,iP] = 1.0
            iP += 1

        # Steps?
        if steps is not None:
            self.steps = steps
            for step in steps:
                sline = np.zeros((self.data.shape[0],))
                p = np.flatnonzero(np.flatnonzero(time)>=step)
                sline[p] = 1.0
                G[:,iP] = sline
                iP += 1

        # Constituents
        periods = []
        for constituent in constituents:
            period = self.tidePeriodDict[constituent]/24.
            if not derivative:
                G[:,iP] = np.cos(2*np.pi*Tvec/period)
                G[:,iP+1] = np.sin(2*np.pi*Tvec/period)
            else:
                G[:,iP] = -1.0*(2*np.pi/period)*np.sin(2*np.pi*Tvec/period)
                G[:,iP+1] = (2*np.pi/period)*np.cos(2*np.pi*Tvec/period)
            periods.append(period)
            iP += 2
        self.periods = periods

        # Save G
        if derivative:
            self.Gderiv = G
        else:
            self.G = G

        # All done
        return

    def predict(self, timeseries, constituents='all', linear=False, steps=True, cossin=False,
            derivative=False):
        '''
        Given the results of the fit, this routine predicts the time series.
        Args:
            * timeseries    : timeseries instance.
            * constituents  : List of constituents (default: 'all')
            * linear        : Include the linear trend (default: False).
            * steps         : Include the steps (default: False).
            * cossin        : Add a simple cos+sin term
            * derivative    : If True, stores results in timeseries.deriv, 
                              else, stores results in timeseries.synth
        '''

        # If cossin
        if cossin:
            if type(constituents) is list:
                if 'cossin' not in constituents:
                    constituents += ['cossin']
            else:
                constituents = [constituents, 'cossin']

        # Build G
        if steps:
            steps = self.steps
        else:
            steps = None
        self.buildG(timeseries, chunks=None, linear=linear, steps=steps, 
                constituents=constituents, derivative=derivative)

        # Get the model vector
        if derivative:
            m = np.zeros((self.Gderiv.shape[1],))
        else:
            m = np.zeros((self.G.shape[1],))

        # Offsets
        if not derivative:
            m[0] = self.Offset
            iP = 1
        else:
            iP = 0

        # Linear
        if linear:
            m[iP] = self.Linear
            iP += 1

        # Steps
        if derivative:
            steps = None
        if steps is not None:
            for step in self.Steps:
                m[iP] = step
                iP += 1

        # Constituents
        if type(constituents) is str:
            if constituents in ('All', 'all', 'ALL'):
                constituents = self.tidePeriodDict.keys()
        for constituent in constituents:
            m[iP:iP+2] = self.Constituents[constituent]
            iP += 2

        # Predict
        if derivative:
            timeseries.derivative = np.dot(self.Gderiv, m)
        else:
            timeseries.synth = np.dot(self.G, m)

        # All done
        return

    def findPeaksTidalModel(self, timeseries, maximum=True, minimum=True):
        '''
        Returns all the local maximums (if True) and minimums (if True) in the modeled tidal signal.
        '''

        # Assert a few things
        assert hasattr(self, 'Constituents'), 'Need a dictionary of constituents amplitudes'

        # Get the constituents
        constituents = self.constituents

        # build G and its derivative
        self.buildG(timeseries, constituents=constituents, linear=False, steps=None, derivative=True)
        self.buildG(timeseries, constituents=constituents, linear=False, steps=None, derivative=False)

        # Predict the signal and its derivative
        self.predict(timeseries, constituents=constituents, linear=False, steps=False, derivative=True)
        self.predict(timeseries, constituents=constituents, linear=False, steps=False, derivative=False)

        # Find the zero crossing points
        u = np.array(timeseries.findZeroIntersect(data='derivative'))

        # Get the mean 
        mean = np.mean(timeseries.synth)

        # Maximum are those above the mean (resp. Minimum, under)
        M = np.flatnonzero(timeseries.synth[u]>mean)
        m = np.flatnonzero(timeseries.synth[u]<mean)

        # return
        R = []
        if maximum:
            R.append(u[M])
        if minimum:
            R.append(u[m])

        # All done
        return R
        

