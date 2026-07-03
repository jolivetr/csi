# Class to sample the posterior when adding data from a prior set of samples

# Import stuff
import numpy as np
try:
    import pymc
except:
    pass

class resample(object):
    '''
    Class to sample the posterior probability function of 2 interpolating 
    variables given a 1D set of data.
    '''

    def __init__(self, data, sigma, time, 
                       samples, fixedSamples, bounds, 
                       fpred, comm, niter=1000):
        '''
        Initialization of an instance.

        Args:
            * data          : Set of data 
            * sigma         : standard dev of data covariance
            * time          : Time of each data
            * Samples       : Set of samples to be taken as initial state
            * fixedSample   : Sample set of the previous interpolating functions
            * bounds        : Prior pdf bounds (priors are uniform)
                         ex : bounds = (0., 10.)
            * fpred         : Prediction function
            * comm          : MPI communicator
            * niter         : number of iterations per chain
        '''

        # Save the communicator
        self.comm = comm
        self.me = comm.Get_rank()

        # save the samples
        self.samples = samples
        self.fixedsamples = fixedSamples

        # save the bounds
        self.bounds = bounds

        # save the data and time
        self.data = data
        self.sigma = sigma
        self.time = time

        # save the prediction function
        self.fpred = fpred

        # save the iteration number
        self.niter = niter

        # All done 
        return

    def walkOneChain(self, startingPoints, fixedPoints):
        '''
        Do a metropolis walk starting from a sample.

        Args:
            * startingPoints   : Sample to start from (ex: [0.2342, 1.345, 0.34])
            * fixedPoints      : Value of the preceding basis functions
        '''

        # Create the priors
        Priors = []
        for sp, b in zip(startingPoints, range(len(startingPoints))):
            Priors.append(pymc.Uniform('Alpha {}'.format(b), 
                                       self.bounds[0], self.bounds[1], 
                                       value=sp))

        # Data prediction function
        @pymc.deterministic(plot=False)
        def forward(theta=Priors):
            return self.fpred(fixedPoints + theta)

        # Create a multivariate normal likelihood (the pdf is gaussian so far, so 
        # a diagonal covariance matrix will do it)
        likelihood = pymc.Normal('Data likelihood', mu=forward, 
                                 tau=1./self.sigma**2, 
                                 value=self.data, observed=True)
        
        # Create a sampler
        sampler = pymc.MCMC(Priors+[likelihood])
        for p in Priors:
            sampler.use_step_method(pymc.Metropolis, p)

        # Sample
        sampler.sample(iter=self.niter, burn=self.niter-1, progress_bar=False)

        # All done -- return the last sample
        return [sampler.trace('Alpha {}'.format(b))[-1] for b in range(len(Priors))]

    def sample(self):
        '''
        Samples the posterior PDF, starting from the samples in self.samples
        '''

        # Get counters
        nbasis = self.samples.shape[1]
        nsamples = self.samples.shape[0]

        # New list of samples
        nextSample = [[] for b in range(nbasis)]

        # Iterate over the samples
        for i in range(nsamples):
            starting = self.samples[i,:].tolist()
            fixed = self.fixedsamples[i, :].tolist()
            newsamples = self.walkOneChain(starting, fixed)
            for i in range(nbasis):
                nextSample[i].append(newsamples[i])

        # All done
        return np.array(nextSample)

#EOF
