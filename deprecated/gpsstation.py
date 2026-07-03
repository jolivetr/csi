''' 
A class that deals with a gps station from the sopac network.

Written by R. Jolivet, April 2013.
'''

import numpy as np
import pyproj as pp
import datetime as dt
import sopac as sopac
import matplotlib.pyplot as plt
import os.path

# Import the class
from .timefnutils import *

class gpsstation:

    '''Class meant to deal with data from a single GPS station.'''

    def __init__(self, station, directory='/Users/jolivetinsar/Documents/ParkfieldCreep/GPS/TimeSeries/Filtered'):
        '''
        Initializes the object.
        Args:
            * station       : name of the station.
            * directory     : where to find the files.
        '''

        # Store things
        self.name = station
        self.directory = directory
        self.scale = 1.

        # Initialize things
        self.model = None

        # check if file exist
        filename = self.directory+'/'+self.name+'CleanFlt.neu'
        if os.path.isfile(filename):
            self.valid = True
        else:
            self.valid = False

        # All done
        return

    def read_sopac_timeseries(self):
        '''
        Reads the time series from the sopac file.
        Files are named STATCleanFlt.enu, where STAT is the station code name.
        '''

        # Get the file name
        fname = self.directory+'/'+self.name+'CleanFlt.neu'

        # Read what is in this file
        [ddec,yr,day,north,east,up,dnor,deas,dup] = self.textread(fname,'F I I F F F F F F')

        # Store the time vector
        self.time = np.array(ddec)

        # build a datetime object
        self.date = []
        for i in range(len(yr)):
            jour = (dt.datetime(yr[i], 1, 1).toordinal()) + day[i]
            self.date.append(dt.datetime.fromordinal(jour))

        # stores the enu 
        self.enu = np.zeros((len(north), 3))
        self.enu[:,0] = np.array(east)
        self.enu[:,1] = np.array(north)
        self.enu[:,2] = np.array(up)

        # stores the errors
        self.err = np.zeros((self.enu.shape))
        self.err[:,0] = np.array(deas)
        self.err[:,1] = np.array(dnor)
        self.err[:,2] = np.array(dup)

        # All done
        return

    def read_sopac_model(self):
        '''
        Reads the sopac model from the sopac file.
        Files are named STATCleanFlt.enu, where STAT is the station code name.
        '''

        # Get the file name 
        fname = self.directory+'/'+self.name+'CleanFlt.neu'

        # Initialize a sopac model
        model = sopac.sopac(fname)

        # Stores the model where it needs to be
        self.model = model

        # Create the synthetic array
        self.synth = None

        # All done
        return

    def spitsopacvelocity(self, components='enu'):
        '''
        Returns the model parameters for the stations.
        Args:
            * components    : any combination of e, n and u.
        '''

        # Check if model has been read
        if self.model is None:
            self.read_sopac_model()

        # east
        velocity = {}
        if 'e' in components:
            velocity['east'] = []
            vel = self.model.east.slope
            for v in vel:
                win = v.win
                amp = v.amp
                err = v.err
                velocity['east'].append([amp, err, win])

        # North
        if 'n' in components:
            velocity['north'] = []
            vel = self.model.north.slope
            for v in vel:
                win = v.win
                amp = v.amp
                err = v.err
                velocity['north'].append([amp, err, win])

        # Up
        if 'u' in components:
            velocity['up'] = []
            vel = self.model.up.slope
            for v in vel:
                win = v.win
                amp = v.amp
                err = v.err
                velocity['up'].append([amp, err, win])

        # All done
        return velocity

    def scaledisp(self, sca):
        ''' 
        Scales the displacement and keep the scaling factor.
        '''

        # Store the scaling factor
        self.scale = sca

        # Scale displacements
        self.enu = self.enu*sca
        self.err = self.err*sca

        # Scale synthetics
        if self.synth is not None:
            self.synth = self.synth*sca

        # All done
        return

    def plot(self, components='enu', figure=42):
        '''
        Plot the time series of a single station.
        Args:
            * components    : Which component to plot. Can be any combination of e, n or u.
            * figure        : Number of the figure.
        '''

        # Get the time
        t = self.time
        p = np.array(self.date)

        # Compute the model
        self.sopacmodel2disp(components=components)

        # open a figure
        fig = plt.figure(figure)

        # how many plots
        nplot = len(components)
        for i in range(nplot):
            ax = fig.add_subplot(nplot, 1, i+1)
            if components[i] is 'e':
                ax.set_title('East')
                d = self.enu[:,0]
                s = self.synth[:,0]
                e = self.err[:,0]
            elif components[i] is 'n':
                ax.set_title('North')
                d = self.enu[:,1]
                s = self.synth[:,1]
                e = self.err[:,1]
            elif components[i] is 'u':
                ax.set_title('Up')
                d = self.enu[:,2]
                s = self.synth[:,2]
                e = self.err[:,2]
            else:
                print ('Unknown component asked, abort....')
                return
            # plot them
            ax.errorbar(p, d, yerr=e)
            ax.plot(p, d, '.k')
            ax.plot(p, s, '-r')

        # show()
        plt.show()

        # All done
        return

    def sopacmodel2disp(self, components='enu'):
        '''
        Computes the displacements using the model in sopac.
        Args:
            * components    : list of the components you want to compute. Can be any combination of e, n, u.
        '''

        # Create the synth if it does not exist
        if self.synth is None:
            self.synth = np.zeros((self.enu.shape))

        # build the list of components to compute
        compo = {}
        if 'e' in components:
            compo['east'] = self.model.east
        if 'n' in components:
            compo['north'] = self.model.north
        if 'u' in components:
            compo['up'] = self.model.up

        # Loop over the asked components
        for c in compo.keys():

            # Get the component
            comp = compo[c]

            # First, compute piece-wise linear contributions separately
            amp = []; win = []
            for frep in comp.slope:
                amp.append(frep.amp)
                win.append(frep.win)
            num = len(win)
            points = np.zeros((num+1,), dtype=float)
            tpts = [win[0][0]]
            for jj in xrange(num):
                disp = amp[jj] * (win[jj][1] - win[jj][0])
                points[jj+1] = points[jj] + disp
                tpts.append(win[jj][1])
            linear = np.interp(self.time, tpts, points)

            # Loop through rest of time functionals
            rep = []; amp = []; win = []
            smodel_all = [comp.decay, comp.offset,
                          comp.annual, comp.semi]
            for model in smodel_all:
                for frep in model:
                    rep.append(frep.rep)
                    if isinstance(frep.amp, np.ndarray):
                        amp.extend(frep.amp)
                    else:
                        amp.append(frep.amp)
                    win.append(frep.win)

            # Construct design matrix and compute displacments
            G = np.asarray(Timefn(rep, self.time)[0], order='C')
            fit = (np.dot(G, amp) + linear)*self.scale

            # Remove constant bias and store this into synth
            if c is 'north':
                fit -= np.mean(fit - self.enu[:,1])
                self.synth[:,1] = fit
            elif c is 'east':
                fit -= np.mean(fit - self.enu[:,0])
                self.synth[:,0] = fit
            elif c is 'up':
                fit -= np.mean(fit - self.enu[:,2])
                self.synth[:,2] = fit

        # All done
        return

    def textread(self, fname, strfmt,delim=['#']):
        '''A generic file list reader. All the possible keywords
        are specified below in defns. Arrays are returned in the same order as 
        specified in the strfmt list. All lines starting with delim are ignored.
    
        'S'  - String
        'I'  - Integer
        'F'  - Float
        'K'  - Skip
    
        Returns:
        
            A tuple of list objects for each of the types requested by the user.

        This is a part of GIAnT 1.0.


        '''
    
        inplist = strfmt.split()
        nval = len(inplist)
   
        ########Initiate empty lists
        for ind, inp in enumerate(inplist):
            if inp.upper() not in ('K', 'F', 'I', 'S'):
                raise ValueError('Undefined data type in textread.')
    
            if inp not in ('K', 'k'):
                sname = 'var{:2d}'.format(ind)
                vars()[sname] = []
   
        fin = open(fname, 'r')
        for line in fin.readlines():
            if line[0] not in delim:
                strs = line.split()
                if len(strs) != nval:
                    raise ValueError('Number of records does not match strfmt')
    
                for ind, val in enumerate(strs):
                    inp = inplist[ind]
                    if inp not in ('K', 'k'):
                        sname = 'var{:2d}'.format(ind)
                        vars()[sname].append(val)

        fin.close()

        for ind, inp in enumerate(inplist):
            if inp not in ('K', 'k'):
                sname = 'var{:2d}'.format(ind)
                vars()[sname] = np.array(vars()[sname])
                if inp in ('F', 'f'):
                    vars()[sname] = vars()[sname].astype(float)
                elif inp in ('I', 'i'):
                    vars()[sname] = vars()[sname].astype(np.int)

        retlist = []
        for ind, inp in enumerate(inplist):
            if inp not in ('K', 'k'):
                sname = 'var{:2d}'.format(ind)
                retlist.append(vars()[sname])

        return retlist

#EOF
