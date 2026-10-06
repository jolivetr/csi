# Main class for the interpolation problem

# imports
import numpy as np
import sys, gc, os
import copy
from scipy.misc import factorial
from .resample import resample

def nCk(n,k):
    '''Combinatorial function.'''
    c = factorial(n)/(factorial(n-k)*factorial(k)*1.0)
    return c

def bspline(n,dtk,t):
    '''Uniform b-splines.
       n    -  Order
       dtk  -  Spacing
       t    -  Time vector'''
    x = (t/dtk) + n + 1
    b = np.zeros(len(x))
    for k in range(n+2):
        m = x-k-(n+1)/2
        up = np.power(m,n)
        b = b+((-1)**k)*nCk(n+1,k)*up*(m>=0)
    b = b/(1.0*factorial(n))
    return b

class timebayes(object):

    def __init__(self, data, time, sigma, dt, bounds, nsamples=1000, nbasis=2):
        '''
        Initialization of a timebayes instance.
        This class solves the interpolation problem 
        in a Bayesian framework iteratively.

        Args:
            * data      : vector of data to fit
            * time      : time vector
            * sigma     : noise (1 value, std)
            * dt        : delta-time between the interpolating functions
            * bounds    : bounds of the uniform prior PDF
            * nbasis    : degree of the basis function (e.g., 1: linear (triangular), 
            * nsamples  : number of samples
            * nbasis    : spline order - 1
        '''

        # Create the mpi framework
        import mpi4py
        from mpi4py import MPI

        # Store
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.me = MPI.COMM_WORLD.Get_rank()

        # Init
        self.data = self.comm.bcast(data.astype(float), root=0)
        self.time = self.comm.bcast(time.astype(float), root=0)
        self.sigma = self.comm.bcast(sigma, root=0)
        self.dt = self.comm.bcast(dt, root=0)
        self.bounds = self.comm.bcast(bounds, root=0)
        self.nbasis = self.comm.bcast(nbasis, root=0)        
        self.nsamples = self.comm.bcast(nsamples, root=0)
        self.nbasis = self.comm.bcast(nbasis, root=0)

        # All done
        return

    def finalize(self):
        '''
        Kill MPI workers
        '''
        self.MPI.Finalize()
        # All done
        return
        
    def generateInitialSample(self):
        '''
        Generate the initial set of samples for the first step, 
        given a uniform prior (just samples the prior).
        '''

        return np.random.rand(self.nsamples)*(self.bounds[1]-self.bounds[0]) + self.bounds[0]


    def initializePredFunction(self,bfTimes,times):
        '''
        Initialize the prediction function.

        Args:
             * bfTimes: basis function knots
             * times: observation times
        '''

        # Build the basis functions
        Base = np.zeros((times.size,len(bfTimes)))
        for i in range(len(bfTimes)):
            # Time relative to basis function knot
            dtime = times - bfTimes[i]
            # Get the value of basis function
            Base[:,i] = bspline(self.nbasis, self.dt, dtime)

        # Do stuff
        def predict(alphas):
            '''
            Linear interpolation using uniform b-splines
            Args:
                - alphas: coefficient of each basis function
            '''            
            # All done
            return Base.dot(alphas)

        # Save the prediction function
        self.fpred = predict

        # All done
        return

    def initializeBasisFunctionMap(self):
        '''
        Initialize a map linking data time steps to the index of the first basis
        function for which we invert a coefficient (e.g., for a triangle observation_time/triangle_dt)
        '''
        
        # Map index of basis functions
        self.bfMap = np.array(list(map(int,self.time/self.dt)))
        # Knot vector (i.e., discrete time knots)
        Nfirst = int((self.nbasis-1)/2)     # Number of basis functions starting before zeros
        self.knots = self.dt * (np.arange(self.bfMap.max()+self.nbasis)-Nfirst) # Knot vector (To be checked)

        # All done
        return


    def oneTimeStep(self, step):
        '''
        Does the posterior sampling for a new time step.

        Args:
            * step      : index of the observation time step.
        '''

        import time as pouet

        # Identify the first basis function for which we invert a coefficient
        # Get the first basis function index 
        bfIndex  = self.bfMap[step]
        # Get the corresponding knot
        bi = self.knots[bfIndex]  
        # Start time of the basis function (with respect to bi)
        Tstart = - np.ceil(0.5*(self.nbasis+1))*self.dt 
        # Get the start time of the first basis function
        bstart = bi + Tstart                            
        
        # Select the corresponding data 
        di = np.where(np.logical_and(self.time>=bstart,self.time<=self.time[step]))
        data = self.data[di]
        time = self.time[di]
        
        # Identify the samples to update/create and the fixed ones
        nfixed   = self.nbasis     # Number of fixed coefficients 
        nsampled = self.nbasis + 1 # Number of sampled coefficients
        fixed   = np.zeros((self.nsamples,nfixed))
        samples = np.zeros((self.nsamples,nsampled)) # Initial sample matrix (to be communicated to the sampler)
        if self.me==0:
            if step == 0:
                assert bfIndex == 0, 'Error in the basis function map'
                for i in range(nsampled):
                    samples[:,i] = self.generateInitialSample()
            else:                
                # Fixed coefficients
                for i in range(nfixed):
                    if bfIndex-nfixed+i >= 0:
                        fixed[:,i] = self.samples[bfIndex-nfixed+i]
                # Initial coefficient matrix
                for i in range(nsampled):
                    if bfIndex > self.bfMap[step-1] and i==nsampled-1:
                        samples[:,i] = self.generateInitialSample()
                    else:
                        samples[:,i] = self.samples[bfIndex+i]
 
        # Create a prediction function
        bfTimes = np.arange(bfIndex-nfixed,bfIndex+nsampled)*self.dt + self.knots[0] # Basis function knots
        self.initializePredFunction(bfTimes,time) # Initialize prediction function
        
        # Split
        splitSamples = _split_seq(samples, self.comm.Get_size())
        splitFixed = _split_seq(fixed, self.comm.Get_size())
        splitIndex = _split_seq(range(samples.shape[0]), self.comm.Get_size())

        # Send to each worker
        if self.me==0:
            for worker in range(self.comm.Get_size()):
                self.comm.Send(splitSamples[worker], dest=worker, tag=2*worker)
                self.comm.Send(splitFixed[worker], dest=worker, tag=2*worker+1)

        # Create holders of the right size
        subsamples = np.zeros((splitSamples[self.me].shape[0],nsampled))
        subfixed = np.zeros((splitFixed[self.me].shape[0],nfixed))

        # Wait for everybody
        self.comm.Barrier()

        # Receive
        self.comm.Recv(subsamples, source=0, tag=2*self.me)
        self.comm.Recv(subfixed, source=0, tag=2*self.me+1)

        # Walk the chains in each worker
        sampler = resample(data, self.sigma, time, 
                           subsamples, subfixed, self.bounds, 
                           self.fpred, self.comm, niter=self.chainlength)
        subsamples = sampler.sample()

        # Send to master
        self.comm.Send(subsamples, dest=0, tag=2*self.me)

        # Update/Append samples
        if self.me==0:
            newsamples = np.zeros(samples.shape)
            newsubsamples = np.zeros(subsamples.shape)
            for worker in range(self.comm.Get_size()):
                self.comm.Recv(newsubsamples, source=worker, tag=2*worker)
                newsamples[splitIndex[worker],:] = newsubsamples.T
        else:
            newsamples = None
        
        # Clean up 
        del samples
        del splitSamples, splitFixed, splitIndex

        # All done
        return newsamples, bfIndex

    def walkWithTime(self, chainLength=1000):
        '''
        Advance throught time.
        Args:
            chainLength: MCMC chain length
        '''

        # Model sample set
        self.samples = []

        # Create map of data time <-> basis function knot 
        self.initializeBasisFunctionMap()

        # MCMC chain length 
        self.chainlength = chainLength

        # Iterate over the data
        for step in range(self.data.size):

            # Print stuff
            if self.me==0:
                sys.stdout.write('\r Time Step {} / {}'.format(step+1,self.data.size))
                sys.stdout.flush()

            # Walk one step
            newsamples,bfIndex = self.oneTimeStep(step)

            # Update sample set
            if self.me == 0:
                for c in range(newsamples.shape[1]):
                    alpha = newsamples[:,c].tolist()
                    if bfIndex+c>len(self.samples)-1:
                        self.samples.append(alpha)
                    else:
                        self.samples[bfIndex+c] = copy.deepcopy(alpha)

            # Collect the garbage
            gc.collect()
        
        if self.me == 0:
            print(' All done')

        # All done
        return
            
    def plot(self, savefig=None):
        '''
        Plot data and results if there is some
        '''
        if not self.me:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            cNorm  = colors.Normalize(vmin=0, vmax=self.nsamples)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('jet'))
            # Basis Function center times
            bfTimes = np.arange(len(self.samples))*self.dt
            # Initialize prediction function
            self.initializePredFunction(bfTimes,self.time)
            # Get stochastic predictions
            preds = []
            k = 0
            plt.figure(figsize=(15,8))
            for amplitudes in np.array(self.samples).T:
                k += 1
                plt.plot(self.time, 
                         self.fpred(amplitudes), 
                         '-', color=scalarMap.to_rgba(k))
            # Plot them all
            plt.plot(self.time,self.data,'ko-')

            # Save?
            if savefig is not None:
                plt.savefig(savefig)

            plt.show()
        # All done
        return
                
        
        
# List splitter
def _split_seq(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq
