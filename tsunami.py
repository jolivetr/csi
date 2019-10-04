'''
A class that deals with seismic or high-rate GPS data (not finished)

Written by Z. Duputel, April 2013.
'''

# Externals
import os
import sys
import copy
import shutil
import numpy  as np
import pyproj as pp
import matplotlib.pyplot as plt


# Personals
#xfrom WaveMod    import sac
from .SourceInv import SourceInv

class tsunami(SourceInv):

    '''
    A class that handles tsunami data.

    Args:
       * name      : Name of the dataset.

    Kwargs:
       * dtype     : data type 
       * utmzone   : UTM zone  (optional, default=None)
       * lon0      : Longitude of the center of the UTM zone
       * lat0      : Latitude of the center of the UTM zone
       * ellps     : ellipsoid (optional, default='WGS84')

    '''

    def __init__(self,name,dtype='tsunami',utmzone=None,ellps='WGS84',lon0=None, lat0=None):

        super(self.__class__,self).__init__(name,utmzone,ellps=ellps,lon0=lon0,lat0=lat0)

        # Initialize the data set
        self.dtype = dtype

        # Data
        self.d   = []
        self.Cd  = None
        self.sta = None
        self.lat = None
        self.lon = None
        self.t0  = None
        self.G = None

        # All done
        return

    def readFromTxtFile(self,filename,factor=1.0,fileinfo=None):
        '''
        Read d, Cd from files filename.data filename.Cd

        Args:  
            * filename  : prefix of the filenames filename.d and filename.Cd

        Kwargs:
            * factor    : scaling factor
            * fileinfo  : Information about the data (lon, lat and origintime)

        Returns:
            * None
        '''

        self.Cd = np.loadtxt(filename+'.Cd')*factor*factor
        self.d  = np.loadtxt(filename+'.data')*factor
        self.sta = open(filename+'.id').readlines()
        if fileinfo is not None:
            f = open(fileinfo,'rt')
            self.lon = []            
            self.lat = []
            self.t0  = []
            for l in f:
                items = list(map(float,l.strip().split()[1:]))
                self.lon.append(items[0])
                self.lat.append(items[1])
                self.t0.append(items[2])
            f.close()
            assert len(self.t0)==len(self.sta)
        # All done
        return

    def getGF(self,filename,fault,factor=1.0):
        '''
        Read GF from file filename.gf

        Args:
            * filename  : prefix of the file filename.gf

        Kwargs:
            * factor:   scaling factor
        
        Returns:
            * 2d arrays: returns GF_SS and GF_DS
        '''

        GF = np.loadtxt(filename+'.gf')*factor
        n  = GF.shape[1]/2
        assert n == len(fault.slip), 'Incompatible tsunami GF size'
        GF_SS = GF[:,:n]
        GF_DS = GF[:,n:]

        #  All done
        return GF_SS, GF_DS

    def setGFsInFault(self, fault, G, vertical=False):
        '''
        From a dictionary of Green's functions, sets these correctly into the fault 
        object fault for future computation.

        Args:
            * fault     : Instance of Fault
            * G         : Dictionary with 3 entries 'strikeslip', 'dipslip' and 'tensile'. These can be a matrix or None.

        Kwargs:
            * vertical  : Set here for consistency with other data objects, but will always be set to False, whatever you do.

        Returns:
            * None
        '''

        # Get the values
        try: 
            GssLOS = G['strikeslip']
        except:
            GssLOS = None
        try:
            GdsLOS = G['dipslip']
        except:
            GdsLOS = None
        try: 
            GtsLOS = G['tensile']
        except:
            GtsLOS = None
        try:
            GcpLOS = G['coupling']
        except:
            GcpLOS = None

        # set the GFs
        fault.setGFs(self, strikeslip=[GssLOS], dipslip=[GdsLOS], tensile=[GtsLOS],
                    coupling=[GcpLOS], vertical=False)

        # All done
        return


    def buildsynth(self, faults, direction='sd', poly=None):
        '''
        Takes the slip model in each of the faults and builds the synthetic displacement using the Green's functions.

        Args:
            * faults        : list of faults to include.

        Kwargs:
            * direction     : list of directions to use. Can be any combination of 's', 'd' and 't'.
            * poly          : if True, add an offseta in the data

        Returns:   
            * None. Synthetics are stored in the synth attribute
        '''

        Nd = len(self.d)

        # Clean synth
        self.synth = np.zeros(self.d.shape)

        for fault in faults:

            # Get the good part of G
            G = fault.G[self.name]

            if ('s' in direction) and ('strikeslip' in G.keys()):
                Gs = G['strikeslip']
                Ss = fault.slip[:,0]
                self.synth += np.dot(Gs,Ss)
            if ('d' in direction) and ('dipslip' in G.keys()):
                Gd = G['dipslip']
                Sd = fault.slip[:,1]
                self.synth += np.dot(Gd, Sd)

            if poly is not None:
                esti = self.getRampEstimator(fault.poly[self.name])
                sol = fault.polysol[self.name]
                self.shift = esti.dot(sol)
                
                if poly == 'include':
                    self.synth += self.shift

        # All done
        return

    def plot(self, nobs_per_trace, plot_synth=False,alpha=1.,figsize=(13,10),left=0.07,bottom=0.1,
             right=0.99,top=0.9,wspace=0.31,hspace=0.47,scale=100.,ylim=None,yticks=None):
        '''
        Plot tsunami traces

        :Note: We need a description of the options here...

        '''
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(bottom=bottom,top=top,left=left,right=right,wspace=wspace,hspace=hspace)
        nsamp = nobs_per_trace
        nstat = len(self.d)/nobs_per_trace


        for i in range(nstat): 
            data  = self.d[i*nsamp:nsamp*i+nsamp]
            
            if plot_synth == True:
                if len(self.synth.shape)==2:
                    synth = self.synth[i*nsamp:nsamp*i+nsamp,:]
                else:
                    synth = self.synth[i*nsamp:nsamp*i+nsamp]
            plt.subplot(2,np.ceil(nstat/2.),i+1)
            t = np.arange(len(data))
            if self.t0 is not None:
                t += int(self.t0[i])

            plt.plot(t,data*scale,'k',label='data')
                
            if plot_synth == True:            
                plt.plot(t,synth*scale,'r',alpha=alpha,label='predictions')        
                if i>=nstat/2:
                    plt.legend(loc='best')

            #plt.grid()
            plt.title(self.sta[i])
            if not i%np.ceil(nstat/2.):
                plt.ylabel('Water height, cm')
            if i>=nstat/2:
                if self.t0 is not None:
                    plt.xlabel('Time, min')
                else:
                    plt.xlabel('Time since arrival, min')
            if ylim is not None:
                plt.ylim(ylim[0],ylim[1])
            if yticks is not None:
                plt.yticks(yticks)    
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # All done
        return


    def write2file(self, namefile, data='synth'):
        '''
        Write to a text file

        Args:
            * namefile  : Name of the output file

        Kwargs:
            * data      : can be data or synth

        Returns:       
            * None
        '''
        if data == 'synth':
            np.savetxt(namefile, self.synth.T)
        elif data == 'data':
            np.savetxt(namefile, self.d.T)

        # All done
        return


    def getRampEstimator(self,order):
        '''
        Returns the Estimator of a constant offset in the data

        Args:
            * order : 1, estimate just a vertical shift in the data and ,2, estimate a ramp in the data. Order given as argument is in reality order*number_of_station

        Returns:
            * a 2d array
        '''

        nsta = len(self.sta)
        nd = len(self.d)
        obspersta = nd / nsta
        
        order /= nsta

        shift = np.zeros((nd,nsta*order))

        ista = 0
        for i in range(0,order*nsta,order):
            ib = ista * obspersta
            ie = (ista+1) * obspersta
            ista += 1
            
            shift[ib:ie,i] = 1.0

            if order == 1:
                continue
            elif order == 2:
                shift[ib:ie,i+1] = np.arange(0,obspersta)

            

        return shift

#EOF
