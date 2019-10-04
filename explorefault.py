'''
A class that searches for the best fault to fit some geodetic data.
This class is made for a simple planar fault geometry.
It is close to what R. Grandin has implemented but with a MCMC approach
Grandin's approach will be coded in another class.

Author:
R. Jolivet 2017
'''

# Externals
import sys, os, copy
try:
    import h5py
except:
    print('No hdf5 capabilities detected')
import numpy as np
import matplotlib.pyplot as plt

# PyMC
try:
    import pymc
except:
    pass 

# Personals
from .SourceInv import SourceInv
from .planarfault import planarfault

# Class explorefault
class explorefault(SourceInv):

    '''
    Creates an object that will solve for the best fault details. The fault has only one patch and is embedded in an elastic medium.

    Args:
        * name          : Name of the object

    Kwargs:
        * utmzone       : UTM zone number
        * ellps         : Ellipsoid
        * lon0/lat0     : Refernece of the zone
        * verbose       : Talk to me

    Returns:
        * None
    '''

    def __init__(self, name, utmzone=None, 
                 ellps='WGS84', lon0=None, lat0=None, 
                 verbose=True):

        # Initialize the fault
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing fault exploration {}".format(name))
        self.verbose = verbose

        # Base class init
        super(explorefault,self).__init__(name, utmzone=utmzone, 
                                          ellps=ellps, 
                                          lon0=lon0, lat0=lat0)

        # Keys to look for
        self.keys = ['lon', 'lat', 'depth', 'dip', 
                     'width', 'length', 'strike', 
                     'strikeslip', 'dipslip']

        # Initialize the fault object
        self.fault = planarfault('mcmc fault', utmzone=self.utmzone, 
                                               lon0=self.lon0, 
                                               lat0=self.lat0,
                                               ellps=self.ellps, 
                                               verbose=False)

        # All done
        return

    def setPriors(self, bounds, datas=None, initialSample=None):
        '''
        Initializes the prior likelihood functions.

        Args:
            * bounds        : Bounds is a dictionary that holds the following keys. 
                   - 'lon'        : Longitude (tuple or float)
                   - 'lat'        : Latitude (tuple or float)
                   - 'depth'      : Depth in km of the top of the fault (tuple or float)
                   - 'dip'        : Dip in degree (tuple or float)
                   - 'width'      : Along-dip size in km (tuple or float)
                   - 'length'     : Along-strike length in km (tuple or float)
                   - 'strike'     : Azimuth of the strike (tuple or float)
                   - 'strikeslip' : Strike Slip (tuple or float)
                   - 'dipslip'    : Dip slip (tuple or float)

                              One bound should be a list with the name of a pymc distribution as first element. The following elements will be passed on to the function.
                              example:  bounds[0] = ('Normal', 0., 2.) will give a Normal distribution centered on 0. with a 2. standard deviation.

        Kwargs:
            * datas         : Data sets that will be used. This is in case bounds has tuples or floats for reference of an InSAR data set

            * initialSample : An array the size of the list of bounds default is None and will be randomly set from the prior PDFs

        Returns:
            * None
        '''

        # Make a list of priors
        if not hasattr(self, 'Priors'):
            self.Priors = []

        # Check initialSample
        if initialSample is None:
            initialSample = {}
        else:
            assert len(initialSample)==len(bounds), \
                'Inconsistent size for initialSample: {}'.format(len(initialSample))
        initSampleVec = []

        # What do we sample?
        self.sampledKeys = {}
        isample = 0

        # Iterate over the keys
        for key in self.keys:

            # Check the key has been provided
            assert key in bounds, '{} not defined in the input dictionary'

            # Get the values
            bound = bounds[key]

            # Get the function type
            assert type(bound[0]) is str, 'First element of a bound must be a string'
            function = getattr(pymc, bound[0])

            # Get arguments and create the prior
            args = [key] + bound[1:]
            pm = function(*args)

            # Initial Sample
            if len(initialSample)<len(bounds):
                initialSample[key] = pm.rand()

            # Save it
            if bound[0] is not 'Degenerate':
                self.Priors.append(pm)
                initSampleVec.append(initialSample[key])
                self.sampledKeys[key] = isample
                isample += 1

        # Create a prior for the data set reference term
        # Works only for InSAR data yet
        if datas is not None:

            # Check 
            if type(datas) is not list:
                datas = [datas]
                
            # Iterate over the data
            for data in datas:
                
                # Get it
                assert data.name in bounds, \
                    'No bounds provided for prior for data {}'.format(data.name)
                bound = bounds[data.name]
                key = '{}'.format(data.name)

                # Get function 
                function = getattr(pymc, bound[0])

                # Create prior
                args = [key] + bound[1:]
                pm = function(*args)

                # Initial Sample
                if len(initialSample)<len(bounds):
                    initialSample[key] = pm.rand()

                # Store it
                if bound[0] is not 'Degenerate':
                    self.Priors.append(pm)
                    initSampleVec.append(initialSample[key])
                    self.sampledKeys[key] = isample
                    isample += 1
                    self.keys.append(key)
                data.refnumber = len(self.Priors)-1

        # Save initial sample
        self.initSampleVec = initSampleVec
        self.initialSample = initialSample

        # All done
        return

    def setLikelihood(self, datas, vertical=True):
        '''
        Builds the data likelihood object from the list of geodetic data in datas.

        Args:   
            * datas         : csi geodetic data object (gps or insar) or list of csi geodetic objects. TODO: Add other types of data (opticorr)

        Kwargs:
            * vertical      : Use the verticals for GPS?

        Returns:
            * None
        '''

        # Build the prediction method
        # Initialize the object
        if type(datas) is not list:
            self.datas = [datas]

        # List of likelihoods
        self.Likelihoods = []

        # Create a likelihood function for each of the data set
        for data in self.datas:

            # Get the data type
            if data.dtype=='gps':
                # Get data
                if vertical:
                    dobs = data.vel_enu.flatten()
                else:
                    dobs = data.vel_enu[:,:-1].flatten()
            elif data.dtype=='insar':
                # Get data
                dobs = data.vel

            # Make sure Cd exists
            assert hasattr(data, 'Cd'), \
                    'No data covariance for data set {}'.format(data.name)
            Cd = data.Cd

            # Save the likelihood function
            self.Likelihoods.append([data, dobs, Cd, vertical])

        # All done 
        return

    # Define a function
    def Predict(self, theta, data, vertical=True):
        '''
        Calculates a prediction of the measurement from the theta vector

        Args:
            * theta     : model parameters [lon, lat, depth, dip, width, length, strike, strikeslip, dipslip]
            * data      : Data to test upon

        Kwargs:
            * vertical  : True/False

        Returns:
            * None
        '''

        # Take the values in theta and distribute
        lon = self._getFromTheta(theta, 'lon')
        lat = self._getFromTheta(theta, 'lat') 
        depth = self._getFromTheta(theta, 'depth')
        dip = self._getFromTheta(theta, 'dip')
        width = self._getFromTheta(theta, 'width')
        length = self._getFromTheta(theta, 'length')
        strike = self._getFromTheta(theta, 'strike')
        strikeslip = self._getFromTheta(theta, 'strikeslip')
        dipslip = self._getFromTheta(theta, 'dipslip')
        if hasattr(data, 'refnumber'):
            reference = theta[data.refnumber]
        else:
            reference = 0.

        # Get the fault
        fault = self.fault

        # Build a planar fault
        fault.buildPatches(lon, lat, depth, strike, dip, 
                           length, width, 1, 1, verbose=False)

        # Build the green's functions
        fault.buildGFs(data, vertical=vertical, slipdir='sd', verbose=False)

        # Set slip 
        fault.slip[:,0] = strikeslip
        fault.slip[:,1] = dipslip

        # Build the synthetics
        data.buildsynth(fault)

        # check data type 
        if data.dtype=='gps':
            if vertical: 
                return data.synth.flatten()
            else:
                return data.synth[:,:-1].flatten()
        elif data.dtype=='insar':
            return data.synth.flatten()+reference

        # All done
        return

    def walk(self, niter=10000, nburn=5000, method='AdaptiveMetropolis'):
        '''
        March the MCMC.

        Kwargs:
            * niter             : Number of steps to walk
            * nburn             : Numbero of steps to burn
            * method            : One of the stepmethods of PyMC2

        Returns:
            * None
        '''

        # Define the stochastic function
        @pymc.stochastic
        def prior(value=self.initSampleVec):
            prob = 0.
            for prior, val in zip(self.Priors, value):
                prior.set_value(val)
                prob += prior.logp
            return prob

        # Create the deterministics
        likelihood = []
        for like in self.Likelihoods:
            
            # Get what I need
            data, dobs, Cd, vertical = like 

            # Create the forward method
            @pymc.deterministic(plot=False)
            def forward(theta=prior):
                return self.Predict(theta, data, vertical=vertical)

            # Build likelihood function
            likelihood.append(pymc.MvNormalCov('Data Likelihood: {}'.format(data.name), 
                                               mu=forward, 
                                               C=Cd, 
                                               value=dobs, 
                                               observed=True))
        
        # List of pdf to sample
        pdfs = [prior] + likelihood

        # Create a sampler
        sampler = pymc.MCMC(pdfs)

        # Make sure step method is what is asked for
        sampler.use_step_method(getattr(pymc, method), prior)

        # Sample
        sampler.sample(iter=niter, burn=nburn)

        # Save the sampler
        self.sampler = sampler
        self.nsamples = niter - nburn

        # All done
        return

    def returnModel(self, model='mean'):
        '''
        Returns a fault corresponding to the desired model.

        Kwargs:
            * model             : Can be 'mean', 'median',  'rand', an integer or a dictionary with the appropriate keys

        Returns:
            * fault instance
        '''

        # Create a dictionary
        specs = {}

        # Iterate over the keys
        for key in self.sampledKeys:
            
            # Get index
            ikey = self.sampledKeys[key]
        
            # Get it 
            if model=='mean':
                value = self.sampler.trace('prior')[:][:,ikey].mean()
            elif model=='median':
                value = self.sampler.trace('prior')[:][:,ikey].median()
            elif model=='std':                     
                value = self.sampler.trace('prior')[:][:,ikey].std()
            else: 
                if type(model) is int:
                    assert type(model) is int, 'Model type unknown: {}'.format(model)
                    value = self.sampler.trace('prior')[model,ikey]
                elif type(model) is dict:
                    value = model[key]

            # Set it
            specs[key] = value

        # Iterate over the others
        for key in self.keys:
            if key not in specs:
                specs[key] = self.initialSample[key]

        # Create a fault
        fault = planarfault('{} model'.format(model), 
                            utmzone=self.utmzone, 
                            lon0=self.lon0, 
                            lat0=self.lat0,
                            ellps=self.ellps, 
                            verbose=False)
        fault.buildPatches(specs['lon'], specs['lat'], 
                           specs['depth'], specs['strike'],
                           specs['dip'], specs['length'],
                           specs['width'], 1, 1, verbose=False)
        
        # Set slip values
        fault.slip[:,0] = specs['strikeslip']
        fault.slip[:,1] = specs['dipslip']

        # Save the desired model 
        self.model = specs

        # All done
        return fault
    
    def plot(self, model='mean', show=True):
        '''
        Plots the PDFs and the desired model predictions and residuals.

        Kwargs:
            * model     : 'mean', 'median' or 'rand'
            * show      : True/False

        Returns:
            * None
        '''

        # Plot the pymc stuff
        for iprior, prior in enumerate(self.Priors):
            trace = self.sampler.trace('prior')[:][:,iprior]
            fig = plt.figure()
            plt.subplot2grid((1,4), (0,0), colspan=3)
            plt.plot([0, len(trace)], [trace.mean(), trace.mean()], 
                     '--', linewidth=2)
            plt.plot(trace, 'o-')
            plt.title(prior.__name__)
            plt.subplot2grid((1,4), (0,3), colspan=1)
            plt.hist(trace, orientation='horizontal')
            #plt.savefig('{}.png'.format(prior[0]))

        # Get the model
        fault = self.returnModel(model=model)

        # Build predictions
        for data in self.datas:

            # Build the green's functions
            fault.buildGFs(data, slipdir='sd', verbose=False)

            # Build the synthetics
            data.buildsynth(fault)

            # Check ref
            if '{}'.format(data.name) in self.keys:
                data.synth += self.model['{}'.format(data.name)]

            # Plot the data and synthetics
            cmin = np.min(data.vel)
            cmax = np.max(data.vel)
            data.plot(data='data',  show=False, norm=[cmin, cmax])
            data.plot(data='synth', show=False, norm=[cmin, cmax])
        
        # Plot
        if show:
            plt.show()

        # All done
        return

    def save2h5(self, filename):
        '''
        Save the results to a h5 file.

        Args:
            * filename          : Name of the input file

        Returns:
            * None
        '''

        # Open an h5file
        fout = h5py.File(filename, 'w')

        # Create the data sets for the keys
        for key in self.sampledKeys:
            ikey = self.sampledKeys[key]
            fout.create_dataset(key, data=self.sampler.trace('prior')[:][:,ikey])

        # Close file
        fout.close()

        # All done
        return

    def _getFromTheta(self, theta, string):
        '''
        Returns the value from the set of sampled and unsampled pdfs
        '''

        # Try to get the value
        if string in self.sampledKeys:
            return theta[self.sampledKeys[string]]
        else:
            return self.initialSample[string]

#EOF
