'''
A class that allows to compute, fit and display the
empirical covariances in a function.

Written by R. Jolivet, July 2014.
'''

# Externals
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
import sys, os
import copy

# Personals
from .insar import insar
from .opticorr import opticorr

# Some Usefull functions    
def costFunction(m, t, function, data, weights, fun):
    sil, sig, lam = m
    return np.sum(np.sqrt(((data-function(t, 
        sig, lam, covfn=fun, constant=sil))*weights)**2))

def exp_fn(t,sil,sig,lam):
    return sil - (sig**2)*np.exp(-t/lam)

def gauss_fn(t, sil, sig, lam):
    return sil - (sig**2)*np.exp(-(t**2)/(2*(lam)**2))

def covariance(t,sig,lam,covfn='exp',constant=0.):
    if covfn in ('exp'):
        return (sig**2)*np.exp(-t/lam)+constant
    elif covfn in ('gauss'):
        return (sig**2)*np.exp(-(t**2)/(2*(lam)**2))+constant

def ramp_fn(t,a,b,c):
    v = np.array([a,b,c])
    return np.dot(t, v)

# Main class
class imagecovariance(object):
    '''
    A class that allows image covariance determination.

    Args:
        * name      : Name of the object.
        * image     : InSAR or Opticorr data set

    Kwargs:
        * verbose   : Talk to me

    Returns:
        * None
    '''

    def __init__(self, name, image, verbose=True):

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize InSAR covariance tools {}".format(name))

        # Save it
        self.verbose = verbose

        # Set the name
        self.name = name
        self.datatype = image.dtype

        # Set the transformation
        self.utmzone = image.utmzone
        self.putm = image.putm
        self.ll2xy = image.ll2xy
        self.xy2ll = image.xy2ll

        # Save the image
        self.image = image

        # Iterate and save the datasets to consider
        self.datasets = {}
        if self.datatype is 'insar':
            dname = '{}'.format(self.name)
            self.datasets[dname] = {'x': self.image.x,
                                    'y': self.image.y,
                                    'lon': self.image.lon,
                                    'lat': self.image.lat,
                                    'data': self.image.vel}
        elif self.datatype is 'opticorr':
            dname = '{} East'.format(self.name)
            self.datasets[dname] = {'x': self.image.x,
                                    'y': self.image.y,
                                    'lon': self.image.lon,
                                    'lat': self.image.lat,
                                    'data': self.image.east}
            dname = '{} North'.format(self.name)
            self.datasets[dname] = {'x': self.image.x,
                                    'y': self.image.y,
                                    'lon': self.image.lon,
                                    'lat': self.image.lat,
                                    'data': self.image.north}
        else:
            print('Data type unknown or not recognized by imagecovariance type...')
            sys.exit()

        # All done
        return

    def maskOut(self, box):
        '''
        Picks out some points in order to mask them before computing the covariance.

        Args:
            * box   : List of min and max lon and lat coordinates. Can be a list of lists to specify multiple regions. example: [[ -120, -119, 34, 35], [-122, -121.7, 34.2, 34.3]]

        Returns:
            * None
        '''

        # Check how many zones do we have to remove
        self.maskedZones = []
        if type(box[0]) in (int, float):
            self.maskedZones.append(box)
        else:
            for b in box:
                self.maskedZones.append(b)

        # Iterate over the data sets
        for dname in self.datasets:
            if self.verbose:
                print('Masking data set {}'.format(dname))
            # Iterate over the boxes
            for box in self.maskedZones:
                if self.verbose:
                    print('     Mask: {} <= Lon <= {} || {} <= Lat <= {}'.format(box[0], box[1], box[2], box[3]))
                # Get lon lat
                lon = self.datasets[dname]['lon']
                lat = self.datasets[dname]['lat']
                # Find out the points
                ii = np.flatnonzero(np.logical_and(lon>=box[0], lon<=box[1]))
                jj = np.flatnonzero(np.logical_and(lat>=box[2], lat<=box[3]))
                # intersection
                uu = np.intersect1d(ii,jj)
                # Take them out
                self.datasets[dname]['x'] = np.delete(self.datasets[dname]['x'], uu)
                self.datasets[dname]['y'] = np.delete(self.datasets[dname]['y'], uu)
                self.datasets[dname]['lon'] = np.delete(self.datasets[dname]['lon'], uu)
                self.datasets[dname]['lat'] = np.delete(self.datasets[dname]['lat'], uu)
                self.datasets[dname]['data'] = np.delete(self.datasets[dname]['data'], uu)

        # All done
        return

    def maskIn(self, box):
        '''
        Select Boxes on which to compute the covariance.

        Args:
            * box: List of min and max lon and lat coordinates. Can be a list of lists to specify multiple regions. ex: [[ -120, -119, 34, 35], [-122, -121.7, 34.2, 34.3]]

        Returns:
            * None
        '''

        # Check how many zones do we have to keep
        self.selectedZones = []
        if type(box[0]) in (int, float):
            self.selectedZones.append(box)
        else:
            for b in box:
                self.selectedZones.append(b)

        # Iterate over the data sets
        for dname in self.datasets:
            if self.verbose:
                print('Dealing with data set {}'.format(dname))
            # Create a new data set
            self.datasets['New One'] = {'x': np.empty(0), 
                                        'y': np.empty(0),
                                        'lon': np.empty(0), 
                                        'lat': np.empty(0), 
                                        'data': np.empty(0)}
            # Iterate over the boxes
            for box in self.selectedZones:
                if self.verbose:
                    print('     Zone of Interest: {} <= Lon <= {} || {} <= Lat <= {}'.format(box[0], 
                        box[1], box[2], box[3]))
                # Get lon lat
                lon = self.datasets[dname]['lon']
                lat = self.datasets[dname]['lat']
                # Find out the points
                ii = np.flatnonzero(np.logical_and(lon>=box[0], lon<=box[1]))
                jj = np.flatnonzero(np.logical_and(lat>=box[2], lat<=box[3]))
                # intersection
                uu = np.intersect1d(ii,jj)
                # Take them in
                x = self.datasets[dname]['x'][uu]
                y = self.datasets[dname]['y'][uu]
                lon = self.datasets[dname]['lon'][uu]
                lat = self.datasets[dname]['lat'][uu]
                data = self.datasets[dname]['data'][uu]
                # Put them in the new data set
                self.datasets['New One']['x'] = np.hstack((self.datasets['New One']['x'], x))
                self.datasets['New One']['y'] = np.hstack((self.datasets['New One']['y'], y))
                self.datasets['New One']['lon'] = np.hstack((self.datasets['New One']['lon'], lon))
                self.datasets['New One']['lat'] = np.hstack((self.datasets['New One']['lat'], lat))
                self.datasets['New One']['data'] = np.hstack((self.datasets['New One']['data'], data))

            # Replace the data set by the New One
            self.datasets[dname] = copy.deepcopy(self.datasets['New One'])
            del self.datasets['New One']

        # All done
        return

    def empiricalSemivariograms(self, frac=0.4, every=1., distmax=50., rampEst=True):
        '''
        Computes the empirical Semivariogram as a function of distance.

        Kwargs:
            * frac      : Size of the fraction of the dataset to take (0 to 1) frac can be an integer, then it is going to be the number of pixels used to compute the covariance
            * distmax   : Truncate the covariance function.
            * every     : Binning of the covariance function.
            * rampEst   : Estimates a ramp before computing the semivariograms

        Returns:
            * None
        '''

        # Iterate over the datasets
        for dname in self.datasets:

            # print
            if self.verbose:
                print('Computing 1-D empirical semivariogram function for data set {}'.format(dname))

            # Get data set
            data = self.datasets[dname]

            # Get values
            x = data['x']
            y = data['y']
            d = data['data']

            # How many samples do we use
            if type(frac) is int:
                Nsamp = frac
                if Nsamp>d.shape[0]:
                    Nsamp = d.shape[0]
            else:
                Nsamp = np.int(np.floor(frac*x.size))
            if self.verbose: 
                print('Selecting {} random samples to estimate the covariance function'.format(Nsamp))

            # Create a vector
            regular = np.vstack((x.squeeze(),y.squeeze(),d.squeeze())).T
            
            # Take a random permutation
            randomized = np.random.permutation(regular)

            # Take the first frac of it
            x = randomized[:Nsamp,0]
            y = randomized[:Nsamp,1]
            d = randomized[:Nsamp,2]

            # Remove a ramp
            if rampEst:
                G = np.zeros((Nsamp,6))
                G[:,4] = x*x
                G[:,5] = y*y
                G[:,3] = x*y
                G[:,0] = x
                G[:,1] = y
                G[:,2] = 1.
                pars = np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),d) 
                a, b, c, w, u, v = pars
                d = d - (a*x + b*y + c + w*x*y + u*x*x + v*y*y)
                if self.verbose:
                    print('Estimated Orbital Plane: {}x2 + {}y2 + {}xy + {}x + {}y + {}'.format(u,v,w,a,b,c))  
                # Save it
                data['Ramp'] = [a, b, c, u, v, w]       

            # Build all the permutations
            if self.verbose:
                print('Build the permutations')
            ii, jj = np.meshgrid(range(Nsamp), range(Nsamp))
            ii = ii.flatten()
            jj = jj.flatten()
            uu = np.flatnonzero(ii>jj)
            ii = ii[uu]
            jj = jj[uu]

            # Compute the distances
            dx = x[ii] - x[jj]
            dy = y[ii] - y[jj]
            dis = np.sqrt(dx*dx + dy*dy)

            # Compute the semivariogram
            dv = (d[ii] - d[jj])**2

            # Digitize
            if self.verbose:
                print('Digitize the histogram')
            bins = np.arange(0., distmax, every)
            inds = np.digitize(dis, bins)

            # Average
            distance = []
            semivariogram = []
            std = []
            for i in range(len(bins)-1):
                uu = np.flatnonzero(inds==i)
                if len(uu)>0:
                    distance.append(bins[i] + (bins[i+1] - bins[i])/2.)
                    semivariogram.append(0.5*np.mean(dv[uu]))
                    std.append(np.std(dv[uu]))

            # Store these guys
            data['Distance'] = np.array(distance)
            data['Semivariogram'] = np.array(semivariogram)
            data['Semivariogram Std'] = np.array(std)

        # All done
        return

    def empiricalCovariograms(self, frac=0.4, every=1., distmax=50., rampEst=True):
        '''
        Computes the empirical Covariogram as a function of distance.

        Kwargs:
            * frac      : Size of the fraction of the dataset to take (0 to 1)
                          frac can be an integer, then it is going to be the number of 
                          pixels used to compute the covariance
            * distmax   : Truncate the covariance function.
            * every     : Binning of the covariance function.
            * rampEst   : Estimates a ramp before computing the covariaogram

        Returns:
            * None
        '''

        # Iterate over the datasets
        for dname in self.datasets:

            # print
            if self.verbose:
                print('Computing 1-D empirical semivariogram function for data set {}'.format(dname))

            # Get data set
            data = self.datasets[dname]

            # Get values
            x = data['x']
            y = data['y']
            d = data['data']

            # How many samples do we use
            if type(frac) is int:
                Nsamp = frac
                if Nsamp>d.shape[0]:
                    Nsamp = d.shape[0]
            else:
                Nsamp = np.int(np.floor(frac*x.size))
            if self.verbose: 
                print('Selecting {} random samples to estimate the covariance function'.format(Nsamp))

            # Create a vector
            regular = np.vstack((x.squeeze(),y.squeeze(),d.squeeze())).T
            
            # Take a random permutation
            randomized = np.random.permutation(regular)

            # Take the first frac of it
            x = randomized[:Nsamp,0]
            y = randomized[:Nsamp,1]
            d = randomized[:Nsamp,2]

            # Remove a ramp
            if rampEst:
                G = np.zeros((Nsamp,4))
                G[:,3] = x*y
                G[:,0] = x
                G[:,1] = y
                G[:,2] = 1.
                pars = np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),d) 
                a = pars[0]; b = pars[1]; c = pars[2]; w = pars[3]
                d = d - (a*x + b*y + c + w*x*y)
                if self.verbose:
                    print('Estimated Orbital Plane: {}xy + {}x + {}y + {}'.format(w,a,b,c))  
                # Save it
                data['Ramp'] = [a, b, c, w]       

            # Build all the permutations
            if self.verbose:
                print('Build the permutations')
            ii, jj = np.meshgrid(range(Nsamp), range(Nsamp))
            ii = ii.flatten()
            jj = jj.flatten()
            uu = np.flatnonzero(ii>jj)
            ii = ii[uu]
            jj = jj[uu]

            # Compute the distances
            dx = x[ii] - x[jj]
            dy = y[ii] - y[jj]
            dis = np.sqrt(dx*dx + dy*dy)

            # Compute the semivariogram
            dv = np.abs(d[ii]*d[jj])

            # Digitize
            if self.verbose:
                print('Digitize the histogram')
            bins = np.arange(0., distmax, every)
            inds = np.digitize(dis, bins)

            # Average
            distance = []
            covariogram = []
            std = []
            for i in range(len(bins)-1):
                uu = np.flatnonzero(inds==i)
                if len(uu)>0:
                    distance.append(bins[i] + (bins[i+1] - bins[i])/2.)
                    covariogram.append(0.5*np.mean(dv[uu]))
                    std.append(np.std(dv[uu]))

            # Store these guys
            data['Distance'] = np.array(distance)
            data['Covariogram'] = np.array(covariogram)
            data['Covariogram Std'] = np.array(std)

        # All done
        return

    def computeCovariance(self, function='exp', ComputeCovar=True, frac=0.4, every=1., distmax=50., rampEst=True, prior=None, tol=1e-10):
        '''
        Computes the covariance functions.

        Kwargs:
            * function      : Type of function to fit. Can be 'exp'or 'gauss'.
            * computeCovar  : Recompute the covariogram
            * frac          : Size of the fraction of the dataset to take.
            * distmax       : Truncate the covariance function.
            * every         : Binning of the covariance function.
            * rampEst       : estimate a ramp (default True).
            * prior         : First guess for the covariance estimation [Sill, Sigma, Lambda]
            * tol           : Tolerance for the fit

        Returns:
            * None
        '''

        # Compute the Covariogram
        if ComputeCovar:
            if self.verbose:
                print('Computing covariograms')
            self.empiricalCovariograms(frac=frac, every=every, distmax=distmax, rampEst=rampEst)
        else: # Check that it's already done
            dname = self.datasets.keys()[0]
            assert 'Covariogram' in self.datasets[dname].keys(), 'Need to compute the Covariogram first: {}'.format(dname)

        # Fit the covariograms
        if self.verbose:
            print('Fitting Covariance functions')
        for dname in self.datasets:
 
            # Get the dataset
            data = self.datasets[dname]

            # Find indice of chosen distance
            if distmax < data['Distance'].max():
                idx = np.abs(data['Distance']-distmax).argmin() #indice of closest value to distmax in Distance
            else:
                idx = len(data['Distance'])

            # Get the data
            y = data['Covariogram'][:idx]
            x = data['Distance'][:idx]
            error = data['Covariogram Std'][:idx]
            weights = 1/len(error)

            # Save the type of function
            data['function'] = function

            if prior is None:
                # We need a very starting point
                # Sill ~ np.mean(y at the end)
                # Lambda ~ intersect between first slope and 0 axis
                # Sigma ~ exp(1/2N * (sum(log(y)) + sum(x)/Lambda)
                u = np.flatnonzero(y>0)
                ly = np.log(y[u])
                s0 = np.mean(y[-4:])
                l0 = self._getl0(dname, s0)
                m0 = self._getm0(dname, s0, l0)
                mprior = [s0, m0, l0]
            else:
                mprior = prior

            if self.verbose:
                print('Dataset {}:'.format(dname))
                print('A prior values: Sill | Sigma | Lambda')
                print('                 {:4f} | {:5f} | {:6f}'.format(mprior[0], mprior[1], mprior[2]))

            # Minimize
            res = sp.minimize(costFunction, mprior, 
                    args=(x, covariance, y, weights, function), 
                    method='L-BFGS-B',
                    bounds=[[0., np.inf], [0., np.inf], [0.01, np.inf]], tol=tol, 
                    options={'maxiter': 200, 'disp': True})
            pars = res.x

            # Save parameters
            sill = pars[0]
            sigm = pars[1]
            lamb = pars[2]
            data['Sill'] = sill
            data['Sigma'] = sigm
            data['Lambda'] = lamb
            data['Covariogram'] -= sill

            # Print
            if self.verbose:
                print('Dataset {}:'.format(dname))
                print('     Sill   :  {}'.format(sill))
                print('     Sigma  :  {}'.format(sigm))
                print('     Lambda :  {}'.format(lamb))

            # Compute the covariance function
            data['Semivariogram'] = sill - data['Covariogram']

        # All done
        return

    def buildCovarianceMatrix(self, image, dname, write2file=None):
        '''
        Uses the fitted covariance parameters to build a covariance matrix for the dataset
        image of type insar or opticorr.

        Args:
            * image     : dataset of type opticorr or insar.
            * dname     : Name of the covariance estimator. If image is opticorr, the datasets used are "dname East" and "dname North".

        Kwargs:
            * write2file: Write to a binary file (np.float32).

        Returns:
            * None
        '''

        # Get the data position
        x = image.x
        y = image.y

        # Case 1: InSAR
        if image.dtype is 'insar':

            # Get the Parameters
            assert 'Sigma' in self.datasets[dname].keys(), 'Need to estimate the covariance function first: {}'.format(dname)
            sigma = self.datasets[dname]['Sigma']
            lamb = self.datasets[dname]['Lambda']
            function = self.datasets[dname]['function']

            # Build the covariance
            Cd = self._buildcov(sigma, lamb, function, x, y)

        # Case 2: opticorr
        elif image.dtype is 'opticorr':
            
            # Create the two names
            dnameEast = dname+' East'
            dnameNorth = dname+' North'

            # Get the parameters and Build CdEast
            assert 'Sigma' in self.datasets[dnameEast].keys(), 'Need to estimate the covariance function first: {}'.format(dnameEast)
            sigmaEast = self.datasets[dnameEast]['Sigma']
            lambEast = self.datasets[dnameEast]['Lambda']
            funcEast = self.datasets[dnameEast]['function']
            CdEast = self._buildcov(sigmaEast, lambEast, funcEast, x, y)

            # Get the parameters and Build CdNorth
            assert 'Sigma' in self.datasets[dnameNorth].keys(), 'Need to estimate the covariance function first: {}'.format(dnameNorth)
            sigmaNorth = self.datasets[dnameNorth]['Sigma']
            lambNorth = self.datasets[dnameNorth]['Lambda']
            funcNorth = self.datasets[dnameNorth]['function']
            CdNorth = self._buildcov(sigmaNorth, lambNorth, funcNorth, x, y)

            # Cat matrices
            nd = x.shape[0]
            Cd = np.vstack( (np.hstack((CdEast, np.zeros((nd,nd)))), np.hstack((np.zeros((nd,nd)), CdNorth))) )

        # Write 2 a file?
        if write2file is not None:
            Cd.astype(np.float32).tofile(write2file)

        # All done
        return Cd

    def write2file(self, savedir='./'):
        '''
        Writes the results to a text file.
        '''

        # Iterates over the datasets
        for dname in self.datasets:

            # Get data 
            data = self.datasets[dname]
            print('writing covariance output for {}'.format(dname)) # without this line, output was not written in some cases
            
            # continue if nothing has been done
            if 'Covariogram' not in data.keys():
                print('Nothing to be written for data set {}'.format(dname))
                continue

            # filename
            filename = '{}.cov'.format(dname.replace(' ','_'))
            filename = os.path.join(savedir, filename)

            # Open file
            fout = open(filename, 'w')
        
            # Write stuffs
            fout.write('# Covariance estimated for {}\n'.format(dname))

            # Write fit results
            if 'function' in data.keys():
                fout.write('# Best fit function type {}: \n'.format(data['function']))
                fout.write('#       Sill   : {} \n'.format(data['Sill']))
                fout.write('#       Sigma  : {} \n'.format(data['Sigma']))
                fout.write('#       Lambda : {} \n'.format(data['Lambda']))

            # Write header
            header = '# Distance (km) || Covariogram '
            if 'Semivariogram' in data.keys():
                header = header + '|| Semivariogram'
            header = header + '\n'
            fout.write(header)

            # Write what is in there
            distance = data['Distance']
            covar = data['Covariogram']
            covarstd = data['Covariogram Std']
            if 'Semivariogram' in data.keys():
                semivar = data['Semivariogram']
            for i in range(distance.shape[0]):
                d = distance[i]
                s = covar[i]
                ss = covarstd[i]
                line = '{}     {}     {} '.format(d, s, ss)
                if 'Semivariogram' in data.keys():
                    c = semivar[i]
                    line = line + '    {}'.format(c)
                line = line + '\n'
                fout.write(line)

            # Close file
            fout.close()

        # All done
        return

    def plot(self, data='covariance', plotData=False, figure=1, savefig=False, show=True, savedir='./'):
        '''
        Plots the covariance function.

        Kwargs:
            * data      : Can be covariance or semivariogram or all.
            * plotData  : Also plots the image
            * figure    : Figure number
            * savefig   : True/False
            * show      : True/False
            * savedir   : output directory

        Returns:
            * None
        '''

        # Plot the data?
        if plotData:
            plt.figure(figure+1)
            self.image.plot(figure=figure+1, show=False, drawCoastlines=False)
            #if hasattr(self, 'selectedZones'):
            #    for zone in self.selectedZones:
            #        x = [zone[0], zone[0], zone[1], zone[1], zone[0]]
            #        y = [zone[2], zone[3], zone[3], zone[2], zone[2]]
            #        self.image.fig.carte.plot(x, y, '-b', zorder=20)
            #if hasattr(self, 'maskedZones'):
            #    for zone in self.maskedZones:
            #        x = [zone[0], zone[0], zone[1], zone[1], zone[0]]
            #        y = [zone[2], zone[3], zone[3], zone[2], zone[2]]
            #        self.image.fig.carte.plot(x, y, '-r', zorder=20)
            if savefig:
                figname = 'Data_{}.png'.format(self.name.replace(' ','_'))
                figname = os.path.join(savedir, figname)
                plt.savefig(figname)

        # Create a figure
        fig = plt.figure(figure,figsize=(10,10))
        plt.clf()

        # How many data sets
        nData = len(self.datasets)

        # Iterate
        ii = 1
        for dname in self.datasets:

            # Create an axes
            ax = fig.add_subplot(nData, 1, ii)

            # Set its name
            ax.set_title(dname)

            # Plot Semivariogram
            if data in ('semivariogram', 'semi', 'all', 'semivar'):
                semi = self.datasets[dname]['Semivariogram']
                dist = self.datasets[dname]['Distance']
                ax.plot(dist, semi, '.b', markersize=10)
                if 'function' in self.datasets[dname].keys():
                    sill = self.datasets[dname]['Sill']
                    sigm = self.datasets[dname]['Sigma']
                    lamb = self.datasets[dname]['Lambda']
                    function = self.datasets[dname]['function']
                    fy = sill - covariance(dist, sigm, lamb, covfn=function)
                    ax.plot(dist, fy, '-k')

            # Plot Covariance
            if data in ('covariogram', 'all', 'cov'):
                idx = len(self.datasets[dname]['Covariogram'])
                covar = self.datasets[dname]['Covariogram']
                dist = self.datasets[dname]['Distance'][:idx]
                ax.plot(dist, covar, '.k', markersize=10)
                if 'function' in self.datasets[dname].keys():
                    sill = self.datasets[dname]['Sill']
                    sigm = self.datasets[dname]['Sigma']
                    lamb = self.datasets[dname]['Lambda']
                    function = self.datasets[dname]['function']
                    fy = covariance(dist, sigm, lamb, covfn=function)
                    ax.plot(dist, fy, '-r')

            # Axes
            ax.axis('auto')

            # Increase 
            ii += 1

        # Save?
        if savefig:
            figname = '{}.png'.format(self.name.replace(' ','_'))
            figname = os.path.join(savedir, figname)
            plt.savefig(figname)

        # Show me
        if show:
            plt.show()

        # All done
        return

    def read_from_covfile(self,dname,filename):
        '''
        Read a file that was written by write2file()

        Args:
            * dname          : Name of the covariance estimator.
            * filename       : file written with self.write2file()

        Returns:
            * None
        '''

        import linecache

        tmp = np.loadtxt(filename,comments='#')

        l3 = linecache.getline(filename,3)
        l4 = linecache.getline(filename,4)
        l5 = linecache.getline(filename,5)

        self.datasets[dname]['function'] = 'exp'
        self.datasets[dname]['Sill'] = float(l3.split()[-1])
        self.datasets[dname]['Sigma'] = float(l4.split()[-1])
        self.datasets[dname]['Lambda'] = float(l5.split()[-1])
        self.datasets[dname]['Distance'] = tmp[:,0]
        self.datasets[dname]['Covariogram'] = tmp[:,1]
        self.datasets[dname]['Covariogram Std'] = tmp[:,2]
        
        return



    def _buildcov(self, sigma, lamb, func, x, y):
        '''
        Returns a covariance matrix 

        Args:
            * sigma : Arg #1 of function func
            * lamb  : Arg #2 of function func
            * func  : Function of distance ('exp' or 'gauss')
            * x     : position of data along x-axis
            * y     : position of data along y-axis

        Returns:
            * array
        '''

        # Make a distance map matrix
        X1, X2 = np.meshgrid(x,x)
        Y1, Y2 = np.meshgrid(y,y)
        XX = X2-X1
        YY = Y2-Y1
        D = np.sqrt( XX**2 + YY**2)

        # Compute covariance
        Cd = covariance(D, sigma, lamb, covfn=func)

        # All done
        return Cd

    def _getl0(self, dname, s0):
        '''
        From a value of sill, estimates the intersect between the slope on the first
        points and the 0 level.

        Args:
            * dname : Name of the dataset
            * s0    : Estimate of sill

        Returns:
            * float
        '''

        x = self.datasets[dname]['Distance'][:4]
        y = s0 - self.datasets[dname]['Covariogram'][:4]
        m = np.polyfit(x, y, 1)
        return -m[1]/m[0]

    def _getm0(self, dname, s0, l0):
        '''
        Given a sill value and a characteristic distance, returns a rough estimate of sigma.

        Args:
            * dname : Name of the dataset
            * s0    : Estimate of sill
            * l0    : Characteristic distance

        Returns:
            * float
        '''

        x = self.datasets[dname]['Distance']
        y = s0 - self.datasets[dname]['Covariogram']
        y[y<0.] = 0.
        return np.sqrt(np.mean( y/np.exp(-x/l0) ))

#EOF
