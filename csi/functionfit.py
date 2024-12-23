
''' 
A class that offers a fit on a time series.

Written by R. Jolivet, June 2014.
'''

import numpy as np
import datetime as dt
import scipy.optimize as sciopt
import sys

class functionfit(object):
    '''
    A class that fits a fiunction to a time series

    Args:
        * function  : An objective function predicting the data

    Kwargs:
        * verbose   : Talk to me

    Returns:
        * None
    '''

    def __init__(self, function, verbose=True):
        
        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize a function fit object")
        
        # Save the necessary bits
        self.verbose = verbose
        self.function = function

        # All done
        return

    def doFit(self, timeseries, m0, solver='L-BFGS-B', iteration=1000, tol=1e-8):
        '''
        Performs the fit 

        Args:
            * timeseries        : instance of a timeseries class
            * m0                : initial model

        Kwargs:
            * solver            : type of solver from scipy.optimize.minimize
            * iteration         : maximum number of iteration
            * tol               : tolerance of the fit

        Returns:
            * None
        '''
        
        # Create the function to minimize
        def residuals(m, data, function, time, err):
            return np.sqrt(np.sum(1./err * (data-function(m, time))**2))

        # Get stuff
        data = timeseries.value
        time = timeseries.time
        err = timeseries.error

        # Minimize
        res = sciopt.minimize(residuals, m0, 
                              args=(data, self.function, time, err), 
                              method=solver, 
                              options={'disp': self.verbose, 'maxiter': iteration},
                              tol=tol)

        # Save
        self.solution = res
        self.m = res.x

        # All done
        return

    def predict(self, timeseries, set2ts=True):
        '''
        Given the results of the fit, this routine predicts the time series.

        Args:
            * timeseries    : timeseries instance.

        Kwargs:
            * set2ts        : Put the results in timeseries.synth

        Returns:
            * None
        '''

        # Create 
        synth = np.zeros(timeseries.value.shape)

        # Build the synthetics
        synth += self.function(self.m, timeseries.time)

        # All done
        if set2ts:
            timeseries.synth = synth
            return
        else:
            return synth

