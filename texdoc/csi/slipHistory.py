# A class to implement the history of slip
import numpy as np
import datetime as dt

# Local
from .SourceInv import SourceInv
from .timeseries import timeseries

# Class slipHistory
class slipHistory(SourceInv):

    # ----------------------------------------------------------------------
    # Initialize class #
    def __init__(self, fault, direction='sd', utmzone=None, 
                       ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Class to hold the history of slip for a fault.
        So far, this class can handle only one fault. We need to implement the 
        same thing for multiple faults.

        Args:
            * fault         : Instance of a fault 

        Kwargs:
            * direction     : Direction of slip requested 
                              Any combination of 's', 'd' and 't'
            
        '''

        # Base class init
        super(slipHistory,self).__init__('slip history {}'.format(fault.name), 
                                         utmzone=utmzone, 
                                         ellps=ellps, 
                                         lon0=lon0, lat0=lat0)

        # Initialize the object
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing slipHistory class")

        # Talk to me?
        self.verbose = verbose

        # Store the fault
        self.fault = fault
        self.direction = direction

        # How many slip parameters
        if fault.patchType=='triangletent':
            self.nslip = len(fault.tent)
        elif fault.patchType in ('rectangle', 'triangle'):
            self.nslip = len(fault.patch)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Set up the timeseries holders
    def initializeTimeSeries(self, time=None, start=None, end=None, increment=1):
        '''
        Initializes the time series of slip for each patch.

        Kwargs:
            * time          : list of datetime instances
            * start         : datetime instance of the first period
            * end           : datetime instance of the ending period
            * increment     : increment of time between periods
        '''

        # Create the holder of time series
        if 's' in self.direction:
            self.strikeslip = []
        if 'd' in self.direction:
            self.dipslip = []
        if 't' in self.direction:
            self.tensile = []

        # Check fault type
        if self.fault.patchType=='tent':
            patches = self.fault.tent
        elif self.fault.patchType in ('rectangle', 'triangle'):
            patches = self.fault.patch

        # Iterate over the patches
        for ipatch, patch in enumerate(patches):

            # Strike slip
            if 's' in self.direction:
                strikeslip = timeseries('strikeslip {:d}'.format(ipatch), 
                                        utmzone=self.utmzone,
                                        lon0=self.lon0, lat0=self.lat0, 
                                        ellps=self.ellps,
                                        verbose=self.verbose)
                strikeslip.initialize(time=time, start=start, end=end, 
                                      increment=increment)
                self.strikeslip.append(strikeslip)

            # Dip slip
            if 'd' in self.direction:
                dipslip = timeseries('dipslip {:d}'.format(ipatch), 
                                     utmzone=self.utmzone,
                                     lon0=self.lon0, lat0=self.lat0, 
                                     ellps=self.ellps,
                                     verbose=self.verbose)
                dipslip.initialize(time=time, start=start, end=end, 
                                   increment=increment)
                self.dipslip.append(dipslip)

            # Tensile
            if 't' in self.direction:
                tensile = timeseries('tensile {:d}'.format(ipatch), 
                                     utmzone=self.utmzone,
                                     lon0=self.lon0, lat0=self.lat0, 
                                     ellps=self.ellps,
                                     verbose=self.verbose)
                tensile.initialize(time=time, start=start, end=end, 
                                   increment=increment)
                self.tensile.append(tensile)

        # Get the time
        if hasattr(self, 'strikeslip'):
            self.time = self.strikeslip[0].time
        elif hasattr(self, 'dipslip'):
            self.time = self.dipslip[0].time
        elif hasattr(self, 'tensile'):
            self.time = self.tensile[0].time
        
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Convert Time Bayes results into a the slip history object
    def tbayes2history(self, tbayes, model='mean'):
        '''
        Convert the results of a timebayes instance into time series of fault
        slip.

        Args:
            * tbayes        : Instance of timebayes. Need to run walkWithTime

        Kwargs:
            * model         : Which model do we want? 
                              Can be 'mean', 'median'
                              or any integer number giving the index of the model
        '''
        
        # Some lengths 
        npatches = self.fault.slip.shape[0]
        nbf = tbayes.bfDates.size
        ndir = len(self.direction)

        # Some indexes
        if 's' in self.direction:
            iStrikeslip = range(0, npatches)
            imax = npatches
        else:
            imax = 0
        if 'd' in self.direction:
            iDipslip = range(imax, imax+npatches)
            imax += npatches
        else:
            imax += 0
        if 't' in self.direction:
            iTensile = range(imax, imax+npatches)
            imax += npatches
        else:
            imax += 0

        # initialize some holders
        slipKnots = {}
        errorKnots = {}
        if 's' in self.direction:
            slipKnots['strikeslip'] = np.zeros((npatches, nbf))
            errorKnots['strikeslip'] = np.zeros((npatches, nbf))
        if 'd' in self.direction:
            slipKnots['dipslip'] = np.zeros((npatches, nbf))
            errorKnots['dipslip'] = np.zeros((npatches, nbf))
        if 't' in self.direction:
            slipKnots['tensile'] = np.zeros((npatches, nbf))
            errorKnots['tensile'] = np.zeros((npatches, nbf))

        # Compute the basis function from tbayes given the time vector in self 
        tbayes.initBase(tbayes.bfDates, self.time)
        self.Base = tbayes.Base
        
        # Iterate over the slipState vector
        for iknot, knot in enumerate(tbayes.slipState): # This will not work (slipState is a dictionary)
            if 's' in self.direction:
                strikeslip = self._state2model(tbayes.slipState[knot][:,iStrikeslip], 
                                               model)
                error = self._state2model(tbayes.slipState[knot][:,iStrikeslip], model)
                slipKnots['strikeslip'][:,iknot] = strikeslip
                errorKnots['strikeslip'][:,iknot] = error
            if 'd' in self.direction:
                dipslip = self._state2model(tbayes.slipState[knot][:,iDipslip], model)
                error = self._state2model(tbayes.slipState[knot][:,iDipslip], model)
                slipKnots['dipslip'][:,iknot] = dipslip
                errorKnots['dipslip'][:,iknot] = error
            if 't' in self.direction:
                tensile = self._state2model(tbayes.slipState[knot][:,iTensile], model)
                error = self._state2model(tbayes.slipState[knot][:,iTensile], model)
                slipKnots['tensile'][:,iknot] = tensile
                errorKnots['tensile'][:,iknot] = error

        # Multiply the base by the slip model to get the full time series
        if 's' in self.direction:
            strikeSlip = np.dot(self.Base, slipKnots['strikeslip'].T)
            strikeError = np.dot(self.Base, errorKnots['strikeslip'].T)
        if 'd' in self.direction:
            dipSlip = np.dot(self.Base, slipKnots['dipslip'].T)
            dipError = np.dot(self.Base, errorKnots['dipslip'].T)
        if 't' in self.direction:
            tenSlip = np.dot(self.Base, slipKnots['tensile'].T)
            tenError = np.dot(self.Base, errorKnots['tensile'].T)
    
        # Set it up in the time series initialized
        if 's' in self.direction:
            for iss, ss in enumerate(self.strikeslip):
                ss.value[:] = strikeSlip[:,iss]
                ss.error[:] = strikeError[:,iss]
        if 'd' in self.direction:
            for ids, ds in enumerate(self.dipslip):
                ds.value[:] = dipSlip[:,ids]
                ds.error[:] = dipError[:,ids]
        if 't' in self.direction:
            for its, ts in enumerate(self.tensile):
                ts.value[:] = tenSlip[:,its]
                ts.error[:] = tenError[:,its]

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Get slip along time
    def slipattime(self, time):
        '''
        Gets the slip at a given time

        Args:
            * time          : Instance of datetime or list or array of 
                              instances of datetime

        Returns:
            * slip          : Dictionary of 'strikeslip', 'dipslip' and 
                              'tensile'

        '''

        # Make sure about something
        if type(time) is list:
            time = np.array(time)
        elif type(time) is dt.datetime:
            time = np.array([time])
        assert type(time)==type(np.zeros((1,))),\
                'Time should be a datetime, a list or an array of datetime'
        assert time.size, 'Time size is {}'.format(time.size)

        # Some infos
        nslip = self.fault.slip.shape[0]

        # All time vectors should be consistent
        itime = np.array([np.flatnonzero(self.time==ti) for ti in time])

        # Create the output
        timeseries = {}

        # Get the strike slip
        if 's' in self.direction:
            strikeslip = np.zeros((nslip, itime.size))
            for iss, ss in enumerate(self.strikeslip):
                strikeslip[:,iss] = ss.value[itime]
            timeseries['strikeslip'] = strikeslip

        # Get the dip slip
        if 'd' in self.direction:
            dipslip = np.zeros((nslip, itime.size))
            for ids, ds in enumerate(self.dipslip):
                dipslip[:,ids] = ds.value[itime]
            timeseries['dipslip'] = dipslip

        # Get the tensile
        if 't' in self.direction:
            tensile = np.zeros((nslip, itime.size))
            for its, ts in enumerate(self.tensile):
                tensile[:,its] = ts.value[itime]
            timeseries['tensile'] = tensile

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Sets up slip at time t in the fault object
    def slip2fault(self, time):
        '''
        Sets the slip at a time period 'time' into the fault object.

        Args:
            * time          : Instance of datetime

        '''

        # Some infos
        nslip = self.fault.slip.shape[0]

        # All time vectors should be consistent
        itime = np.flatnonzero(self.time==time)
        assert len(itime)==1, 'Time {} is not in the time vector of {}'\
                .format(time, self)

        # Get the strike slip
        if 's' in self.direction:
            strikeslip = np.array([ss.value[itime] for ss in self.strikeslip])
        else:
            strikeslip = np.zeros((nslip,))
        if 'd' in self.direction:
            dipslip = np.array([ds.value[itime] for ds in self.dipslip])
        else:
            dipslip = np.zeros((nslip,))
        if 't' in self.direction:
            tensile = np.array([ts.value[itime] for ts in self.tensile])
        else:
            tensile = np.zeros((nslip,))

        # Set in fault
        fault.slip[:,0] = strikeslip
        fault.slip[:,1] = dipslip
        fault.slip[:,2] = tensile

        # All done
        return
    # ----------------------------------------------------------------------
        
    # ----------------------------------------------------------------------
    # Build time series of data
    def predict(self, datas):
        '''
        For all the data included in data, build time series of displacement.

        Args:
            * datas         : List of data sets.
                              Time series will be initialized given the time
                              vector of slip.
        '''

        # Check or compute GFs
        for data in datas:
            
            # Check
            if data.name in fault.G:
                try:
                    if 's' in self.direction: 
                        a = fault.G[data.name]['strikeslip']*fault.slip[:,0]
                    if 'd' in self.direction:
                        a = fault.G[data.name]['dipslip']*fault.slip[:,1]
                    if 't' in self.direction:
                        a = fault.G[data.name]['tensile']*fault.slip[:,2]
                except:
                    fault.buildGFs(data, vertical=True, 
                                   slipdir=self.direction, method='okada')
            # Else
            else:
                fault.buildGFs(data, vertical=True, 
                               slipdir=self.direction, method='okada')
        
        # Initialize the time series for the data
        for data in datas:
            if hasattr(data, 'time'):
                time = data.time
            else:
                time = self.time
            data.initializeTimeSeries(time=time)

        # Iterate over the data sets to predict
        for data in datas:

            # Get the GFs
            G = fault.G[data.name]

            # Get the time series of slip
            slip = self.slipattime(data.time)

            # Multiply and sum
            displacement = np.zeros((G.shape[0], slip.shape[1]))
            if 's' in self.direction:
                ssdisp = np.dot((G['strikeslip'], slip['strikeslip']))
                displacement += ssdisp
            if 'd' in self.direction:
                dsdisp = np.dot((G['dipslip'], slip['dipslip']))
                displacement += dsdisp
            if 't' in self.direction:
                tsdisp = np.dot((G['tensile'], slip['tensile']))
                displacement += tsdisp

            # Distribute
            self._setintimeseries(data, displacement)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Plot time series of slip for one patch
    def plotPatch(self, patch):
        '''
        Plots the time series of slip for one patch or one tent.

        Args:
            * patch         : Can be a patch or a tent (given the fault 
                              object)
        '''

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Plot slip at time t
    def plotSlip(self, time):
        '''
        Plots the slip at a given time

        Args:
            * time          : instance of datetime
        '''

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Given a table (nsamples, npatches), returns the desired model type
    def _state2model(self, state, model):
        '''
        Returns a vector corresponding to the desired model.

        Args:
            * state         : 2D table of the samples produced 
                              size is (nSamples, nPatches)
            * model         : Model desired
                              options: 
                                'mean' or 'average' 
                                'median'
                                any index smaller than nsamples
        '''

        # This is only about cases
        if model in ('mean', 'average'):
            return np.mean(state, axis=0)
        elif model in ('median'):
            return np.median(state, axis=0)
        else:
            assert type(model)==int, 'Model type unknown'
            return state[model,:]
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Given a table (ndata, ntime) sets them in the time series
    def _setintimeseries(self, data, displacement):
        '''
        Sets the displacement with time in the time series objects.

        Args:
            * data          : Instance of gps or insartimeseries
            * displacement  : Array of size (ndata,ntimes)
        '''

        # Check stuff
        assert displacement.shape[1]==data.time.size,\
                'Displacement of wrong shape --> {} for {} time steps'.\
                format(displacement.shape, data.time.size)

        # Case gps
        if data.dtype=='gps':

            # Check data size
            assert np.prod(data.vel_enu.shape)==displacement.shape[0],\
                   'Displacement of wrong shape --> {} for {} data'.\
                   format(displacement.shape, np.prod(data.vel_enu.shape))

            # Number of stations
            nstation = data.vel_enu.shape[0]

            # Iterate over the timeseries
            for istation, station in enumerate(data.timeseries):

                # East
                data.timeseries[station].east.value = displacement[istation,:]

                # North
                data.timeseries[station].north.value = displacement[istation+nstation,:]

                # Up
                data.timeseries[station].up.value = displacement[istation+2*nstation,:]
        
        # Case InSAR timeseries
        elif data.dtype=='insartimeseries':

            # Check data size
            assert data.vel.size==displacement.shape[0],\
                   'Displacement of wrong shape --> {} for {} data'.\
                   format(displacement.shape, data.vel.size)

            # Set in timeseries
            for isar, sar in enumerate(data.timeseries):

                # Set values
                sar.vel[:] = displacement[:,isar]

        # All done
        return
    # ----------------------------------------------------------------------

#EOF
