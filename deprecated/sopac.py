import numpy as np

class frep:
    '''Functional representation of the SOPAC model parameters.'''
    def __init__(self):
        '''Setting up any model string.'''
        self.amp = None
        self.err = None
        self.rep = None
        self.win = None

    def setslope(self,slope,err,t0,t1):
        self.amp = float(slope)
        self.err = float(err)
        self.win = [float(t0),float(t1)]
        self.rep = ['LINEAR',[float(t0)]]

    def setdecay(self,amp,err,t0,tau):
        self.amp = -1.0 * float(amp)
        self.err = float(err)
        self.win = [float(t0),np.inf]
        self.rep = ['EXP',[float(t0)],[float(tau)/365.25]] 

    def setoffset(self,amp,err,t0):
        self.amp = float(amp)
        self.err = float(err)
        self.win=[float(t0),np.inf]
        self.rep = ['STEP',[float(t0)]]

    def setannual(self,amp,err,phase):
        ph = float(phase)
        gph = np.array([np.cos(ph),np.sin(ph)])
        self.amp = float(amp)*gph
        self.err = float(err)*np.abs(gph)
        self.win =[-np.inf, np.inf]
        self.rep = ['SEASONAL',[1.0]]
        
    def setsemi(self,amp,err,phase):
        ph = float(phase)
        gph = np.array([np.cos(ph),np.sin(ph)])
        self.amp = float(amp)*gph
        self.err = float(err)*np.abs(gph)
        self.win =[-np.inf, np.inf]
        self.rep = ['SEASONAL',[0.5]]

    def __str__(self):
        return str([self.win, self.amp, self.err, self.rep])

class model:
    '''Putting together a model of the GPS station.'''
    def __init__(self,dlines):
        '''Initiates a model with input lines from header of each station file.'''
        self.slope = []
        self.decay = []
        self.offset = []
        self.annual = []
        self.semi = []

        for line in dlines:
            emptyline = False
            parts = line.split()
             
            if parts[0] == 'slope':
                npart = frep()
                npart.setslope(parts[2], parts[4], parts[6], parts[8])
                self.slope.append(npart)
            elif parts[0] == 'ps':
                if parts[2] == 'postseismic':
                    continue
                npart = frep()
                npart.setdecay(parts[3],parts[5],parts[7],parts[9])
                self.decay.append(npart)
            elif parts[0] == 'offset':
                npart=frep()
                npart.setoffset(parts[2],parts[4],parts[6])
                self.offset.append(npart)
            elif parts[0] == 'annual':
                npart = frep()
                npart.setannual(parts[1], parts[3], parts[6])
                self.annual.append(npart)
            elif parts[0] == 'semi-annual':
                npart = frep()
                npart.setsemi(parts[1],parts[3], parts[6]) 
                self.semi.append(npart)
            else:
                emptyline = True

class sopac:
    '''Class to deal with model description provided in the SOPAC daily solutions. '''
    def __init__(self,fname):
        '''Reads and initiates the model parameters.'''
        dlines = self.read_header(fname)
        nind = np.where([k=='n component' for k in dlines])
        nind = nind[0][0]

        eind = np.where([k=='e component' for k in dlines])
        eind = eind[0][0]

        uind = np.where([k=='u component' for k in dlines])
        uind = uind[0][0]

        self.north = model(dlines[nind:eind])
        self.east = model(dlines[eind:uind])
        self.up = model(dlines[uind:])


    @staticmethod
    def read_header(fname,ch='#'):
        '''Reads in all lines in the text file starting with specified character.

         .. Args:

            * fname           Text file name to read.
            * ch              Starting character or pattern.

        .. Returns:

            * dlines         Lines with specified starting pattern.'''

        nch = len(ch)
        lines = []
        fid = open(fname,'r')

        for line in fid:
            line = line.rstrip()
            if line[0:nch] == ch:
                line = line[nch:].lstrip()
                line= line.translate(None,'#:;*()')
                line = line.lstrip()
                line = line.rstrip()
                if line != '':
                    lines.append(line)

        fid.close()
        return lines


############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
