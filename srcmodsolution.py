'''
A class that deals with vertical faults.

Written by R. Jolivet, April 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
import copy
import sys

# Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok
from .gps import gps as gpsclass
from .SourceInv import SourceInv

class srcmodsolution(object):

    def __init__(self, name, utmzone=None, lon0=None, lat0=None, ellps='WGS84', verbose=True):
        '''
        Args:
            * name          : Name of the fault.
        '''

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing fault {}".format(self.name))
        self.verbose = verbose

        # Base class init
        super(srcmodsolution, self).__init__(name, 
                                             utmzone=utmzone, 
                                             lon0=lon0, lat0=lat0, ellps=ellps)

        # Initialize the fault
        self.name = name

        # Set the reference point in the x,y domain (not implemented)
        self.xref = 0.0
        self.yref = 0.0

        # allocate some things
        self.xf = None
        self.yf = None
        self.xi = None
        self.yi = None
        self.loni = None
        self.lati = None

        # Create a dictionary for the rupture characteristics
        self.rupture = {}

        # Allocate depth and number of patches
        self.top = None             # Depth of the top of the fault
        self.depth = None           # Depth of the bottom of the fault
        self.numz = None

        # Allocate patches
        self.patch = None
        self.slip = None
        self.totalslip = None

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}

        # Create a dictionnary for the polysol
        self.polysol = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # All done
        return

    def duplicateFault(self):
        '''
        Returns a copy of the fault.
        '''

        return copy.deepcopy(self)

    def initializeslip(self, n=None):
        '''
        Re-initializes the fault slip array.
        Args:
            * n     : Number of slip values. If None, it'll take the number of patches.
        '''

        if n is None:
            n = len(self.patch)

        self.slip = np.array(())

        # All done
        return

    def trace(self, Lon, Lat):
        '''
        Set the surface fault trace.

        Args:
            * Lon           : Array/List containing the Lon points.
            * Lat           : Array/List containing the Lat points.
        '''

        # Set lon and lat
        self.lon = np.array(Lon)
        self.lat = np.array(Lat)

        # All done
        return

    def addfaults(self, filename):
        '''
        Add some other faults to plot with the modeled one.

        Args:
            * filename      : Name of the fault file (GMT lon lat format).
        '''

        # Allocate a list
        self.addfaults = []

        # Read the file
        fin = open(filename, 'r')
        A = fin.readline()
        tmpflt=[]
        while len(A.split()) > 0:
            if A.split()[0] is '>':
                if len(tmpflt) > 0:
                    self.addfaults.append(np.array(tmpflt))
                tmpflt = []
            else:
                lon = float(A.split()[0])
                lat = float(A.split()[1])
                tmpflt.append([lon,lat])
            A = fin.readline()
        fin.close()

        # Convert to utm
        self.addfaultsxy = []
        for fault in self.addfaults:
            x,y = self.ll2xy(fault[:,0], fault[:,1])
            self.addfaultsxy.append([x,y])

        # All done
        return

    def file2trace(self, filename):
        '''
        Reads the fault trace directly from a file.
        Format is:
        Lon Lat

        Args:
            * filename      : Name of the fault file.
        '''

        # Open the file
        fin = open(filename, 'r')

        # Read the whole thing
        A = fin.readlines()

        # store these into Lon Lat
        Lon = []
        Lat = []
        for i in range(len(A)):
            Lon.append(np.float(A[i].split()[0]))
            Lat.append(np.float(A[i].split()[1]))

        # Create the trace
        self.trace(Lon, Lat)

        # All done
        return

    def trace2xy(self):
        '''
        Transpose the surface trace of the fault into the UTM reference.
        '''

        # do it
        self.xf, self.yf = self.ll2xy(self.lon, self.lat)

        # All done
        return

    def ll2xy(self, lon, lat):
        '''
        Do the lat lon 2 utm transform
        '''

        # Transpose
        x, y = self.putm(lon, lat)

        # Put it in Km
        x = x/1000.
        y = y/1000.

        # All done
        return x, y

    def xy2ll(self, x, y):
        '''
        Do the utm to lat lon transform
        '''

        # Transpose and return
        return self.putm(x*1000., y*1000., inverse=True)

    def readSLP(self, filename):
        '''
        Reads the slip distribution and the geometry from a .slp file (SRCMOD format).
        Args:
            * filename  : Input file name.
        '''

        # Read all the file
        fin = open(filename, 'r')
        Text = fin.readlines()
        fin.close()

        # Set up the line counter
        l = 0

        # Loop over the lines
        while l!=len(Text):

            # Get the line string
            Lstr = Text[l].split()
            Lful = Text[l]

            # Two cases, metadata (%) or slip/rakes
            if Lstr[0] is '%':

                if len(Lstr) > 2:

                    if Lstr[1] == 'Event':
                        self.article = Lful.split('[')[-1][:-2]
                        self.date = Lful.split('[')[0].split().pop()
                        self.event = Lful.split('[')[0].split()[3:-1]
                        l += 1
                    elif Lstr[1] == 'EventTAG':
                        self.eventtag = Lstr.pop()
                        l += 1
                    elif Lstr[1] == 'Loc':
                        self.rupture['LAT'] = np.float(Lstr[5])
                        self.rupture['LON'] = np.float(Lstr[8])
                        self.rupture['DEP'] = np.float(Lstr[11])
                        l += 1
                    elif Lstr[1] == 'Size':
                        self.rupture['LEN'] = np.float(Lstr[5])
                        self.rupture['WID'] = np.float(Lstr[9])
                        self.rupture['Mw'] = np.float(Lstr[13])
                        self.rupture['Mo'] = np.float(Lstr[16])
                        l += 1
                    elif Lstr[1] == 'Mech':
                        self.rupture['Strike'] = np.float(Lstr[5])
                        self.rupture['Dip'] = np.float(Lstr[8])
                        self.rupture['Rake'] = np.float(Lstr[11])
                        self.rupture['Htop'] = np.float(Lstr[14])
                        l += 1
                    elif Lstr[1] == 'Rupt':
                        self.rupture['HypocenterXZ'] = (np.float(Lstr[5]), np.float(Lstr[9]))
                        self.rupture['Average Rupture Time'] = np.float(Lstr[13])
                        self.rupture['Average Rupture Velocity'] = np.float(Lstr[17])
                        l += 1
                    elif Lstr[1] in ('Invs'):
                        if Lstr[3] in ('inDx'):
                            self.rupture['Patch Length'] = np.float(Lstr[5])
                            self.rupture['Patch Width'] = np.float(Lstr[9])
                            self.rupture['Fmin'] = np.float(Lstr[13])
                            self.rupture['Fmax'] = np.float(Lstr[17])
                            l += 1
                        elif Lstr[3] in ('Nx'):
                            self.rupture['Nx'] = np.int(Lstr[5])
                            self.rupture['Nz'] = np.int(Lstr[8])
                            l += 1
                    else:
                        if (Lstr[2] in ('TOTAL')) and (Lstr[3] in ('SLIP')):
                            # There is Nx lines to read
                            Slip = np.array([ [np.float(Text[i].split()[j]) for j in range(len(Text[i].split()))] for i in range(l+1,l+self.rupture['Nz']+1)])
                            self.rupture['Slip'] = Slip
                            # update l
                            l += self.rupture['Nz']+1
                        elif (Lstr[2] in ('LOCAL')) and (Lstr[3] in ('RAKE')):
                            # There is Nx lines to read
                            Rake = np.array([Text[i].split() for i in range(l+1,l+self.rupture['Nz']+1)])
                            self.rupture['Rake'] = Rake
                            # update l
                            l += self.rupture['Nz']+1
                        else:
                            l += 1

                else:

                    # update l and go to the next line
                    l += 1

            else:

                # update l and go to the next line
                l += 1

        # all done
        return

    def readFSP(self, filename):
        '''
        Reads the fault geometry from a FSP file that has only one segment (will implement the multisegment
        later 'cause it is a pain in the ***)
        '''

        # Read all the file
        fin = open(filename, 'r')
        Text = fin.readlines()
        fin.close()

        # Set the line counter
        l = 0

        # Loop over the lines
        while l!=len(Text):

            # get the line
            Lstr = Text[l].split()
            Lful = Text[l]

            if len(Lstr)>2:
                if Lstr[1] == 'Event':
                    self.article = Lful.split('[')[-1][:-2]
                    self.date = Lful.split('[')[0].split().pop()
                    self.event = Lful.split('[')[0].split()[3:-1]
                    l += 1
                elif Lstr[1] == 'EventTAG:':
                    self.eventtag = Lstr.pop()
                    l += 1
                elif Lstr[1] in ('Loc'):
                    self.rupture['LAT'] = np.float(Lstr[5])
                    self.rupture['LON'] = np.float(Lstr[8])
                    self.rupture['DEP'] = np.float(Lstr[11])
                    l += 1
                elif Lstr[1] in ('Size'):
                    self.rupture['LEN'] = np.float(Lstr[5])
                    self.rupture['WID'] = np.float(Lstr[9])
                    self.rupture['Mw'] = np.float(Lstr[13])
                    self.rupture['Mo'] = np.float(Lstr[16])
                    l += 1
                elif Lstr[1] in ('Mech'):
                    self.rupture['Strike'] = np.float(Lstr[5])
                    self.rupture['Dip'] = np.float(Lstr[8])
                    self.rupture['Rake'] = np.float(Lstr[11])
                    self.rupture['Htop'] = np.float(Lstr[14])
                    l += 1
                elif Lstr[1] in ('Rupt'):
                    self.rupture['HypocenterXZ'] = (np.float(Lstr[5]), np.float(Lstr[9]))
                    self.rupture['Average Rupture Time'] = np.float(Lstr[13])
                    self.rupture['Average Rupture Velocity'] = np.float(Lstr[17])
                    l += 1
                elif Lstr[1] in ('Invs'):
                    if Lstr[3] == 'Dx':
                        self.rupture['Dx'] = np.float(Lstr[5])
                        self.rupture['Dz'] = np.float(Lstr[9])
                        l += 1
                    elif Lstr[3] == 'Nx':
                        self.rupture['Nx'] = np.int(Lstr[5])
                        self.rupture['Nz'] = np.int(Lstr[8])
                        l += 1
                    else:
                        # nothing to do, update l
                        l += 1
                elif (Lstr[1] in ('SOURCE')):
                    if (Lstr[2] in ('MODEL')):
                        # Get the number of patches
                        self.rupture['Npatch'] = np.int(Text[l+1].split()[3])
                        # Read the patches
                        patch = np.array([ [np.float(Text[i].split()[j]) for j in range(len(Text[i].split()))]  for i in range(l+9,l+9+self.rupture['Npatch'])])
                        self.rupture['Latitude Patches'] = patch[:,0]
                        self.rupture['Longitude Patches'] = patch[:,1]
                        self.rupture['Slip Patches'] = patch[:,5]
                        self.rupture['Top Depth Patches'] = patch[:,4]
                        if patch.shape[1]>6:
                            self.rupture['Rake Patches'] = patch[:,6]
                        else:
                            self.rupture['Rake Patches'] = np.ones((patch[:,0].shape))*self.rupture['Rake']
                        # update l
                        l += 8+self.rupture['Npatch']
                    else:
                        # nothing to do, update l
                        l += 1
                else:
                    # nothing to do, update l
                    l += 1
            else:

                # nothing to do, update l
                l += 1

        # Set depth
        self.depth = 2*self.rupture['DEP']

        # all done
        return

    def readInput(self, filename):
        '''
        Reads both the fsp and slp files.
        '''

        # Read the FSP
        self.readFSP(filename+'.fsp')

        # Read the SLP
        self.readSLP(filename+'.slp')

        # All done
        return

    def BuildPatches(self):
        '''
        Build patches from the FSP/SLP description
        '''

        # Get some informations
        dip = np.pi/2. + self.rupture['Dip']
        strike = np.pi - self.rupture['Strike']
        dx = self.rupture['Dx']
        dz = self.rupture['Dz']
        npatch = self.rupture['Npatch']

        # Create patch list
        self.patch = []
        self.patchll = []
        self.slip = []

        # Create the patches by iterating over the rupture['patch']
        for i in range(npatch):

            # Get center top
            lon = self.rupture['Longitude Patches'][i]
            lat = self.rupture['Latitude Patches'][i]
            depth = self.rupture['Top Depth Patches'][i]

            # Set in km
            xc, yc = self.ll2xy(lon,lat)

            # Build the 1st corner
            x1 = xc - np.sin(strike*np.pi/180.)*dx/2.
            y1 = yc + np.cos(strike*np.pi/180.)*dx/2.
            z1 = depth
            lon1, lat1 = self.xy2ll(x1, y1)

            # Build the 2nd corner
            x2 = xc + np.sin(strike*np.pi/180.)*dx/2.
            y2 = yc - np.cos(strike*np.pi/180.)*dx/2.
            z2 = depth
            lon2, lat2 = self.xy2ll(x2, y2)

            # Build the 3rd corner
            x3 = x2 + dz/np.tan(dip*np.pi/180.)*np.cos(strike*np.pi/180.)
            y3 = y2 + dz/np.tan(dip*np.pi/180.)*np.sin(strike*np.pi/180.)
            z3 = depth+dz
            lon3, lat3 = self.xy2ll(x3, y3)

            # Build the 4th corner
            x4 = x1 + dz/np.tan(dip*np.pi/180.)*np.cos(strike*np.pi/180.)
            y4 = y1 + dz/np.tan(dip*np.pi/180.)*np.sin(strike*np.pi/180.)
            z4 = depth+dz
            lon4, lat4 = self.xy2ll(x4, y4)

            # fill p
            p = np.zeros((4,3))
            p[0,:] = [x1, y1, z1]
            p[1,:] = [x2, y2, z2]
            p[2,:] = [x3, y3, z3]
            p[3,:] = [x4, y4, z4]
            self.patch.append(p)

            # Fill pll
            pll = np.zeros((4,3))
            pll[0,:] = [lon1, lat1, z1]
            pll[1,:] = [lon2, lat2, z2]
            pll[2,:] = [lon3, lat3, z3]
            pll[3,:] = [lon4, lat4, z4]
            self.patchll.append(pll)

            # Slip
            rake = self.rupture['Rake Patches'][i]
            slip = self.rupture['Slip Patches'][i]
            ss = slip*np.sin(rake*np.pi/180.)
            ds = slip*np.cos(rake*np.pi/180.)
            self.slip.append([ss, ds, 0.0])

        # Translate slip into an array
        self.slip = np.array(self.slip)

        # Build trace
        self.Patch2Trace()

        # All done
        return

    def Flat23D(self, x, y):
        '''
        From a line in the fault plane referential, returns the 3d x, y, z coordinates of that line.
        '''

        # The first patch is the upper left corner
        x1, y1, z1, w, l, s, d = self.getpatchgeometry(self.patch[0])

        # Get some values
        length = self.rupture['Dx']
        width = self.rupture['Dz']
        Tlen = self.rupture['LEN']
        dip = (self.rupture['Dip'])*np.pi/180.
        strike = (self.rupture['Strike'])*np.pi/180.

        # Loop
        X = []; Y = []; Z = []
        for i in range(len(x)):

            # on the flat plane
            xf = x[i]*length # distance along strike to the origin point
            yf = y[i]*width # distance along dip to the origin point

            # In the 3d coordinate system
            xc = x1 + xf*np.sin(strike) + yf*np.cos(dip)*np.cos(strike+np.pi/2.)
            yc = y1 + xf*np.cos(strike) + yf*np.cos(dip)*np.sin(strike+np.pi/2.)
            zc = z1 + yf*np.sin(dip)

            # Append to lists
            X.append(xc)
            Y.append(yc)
            Z.append(zc)

        # Arrays
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        # all done
        return X, Y, Z

    def computeSlipContour(self, value=2.):
        '''
        Computes the slip contour for one value only.
        '''

        # Stores the value somewhere
        self.contourvalue = value

        # Get the flat plane
        Flat = self.rupture['Slip']

        # Extract the contours
        C = plt.contour(Flat, [value])
        #plt.imshow(Flat)
        #plt.colorbar(orientation='horizontal', shrink=0.6)
        #plt.show()
        S = C.collections[0].get_segments()

        # Map onto the fault plane
        Contour = []
        for s in S:
            x = [s[i][0] for i in range(len(s))]
            y = [s[i][1] for i in range(len(s))]
            X, Y, Z = self.Flat23D(x,y)
            Contour.append([X, Y, Z])

        # Self contour
        self.contour = Contour

        # make it lon/lat
        self.contourll = []
        for c in self.contour:
            lon, lat = self.xy2ll(c[0],c[1])
            self.contourll.append([lon,lat,c[2]])

        # All done
        return

    def writeSlipContour2File(self, filename):
        '''
        Writes the slip contour to a file.
        '''

        # Print stuff
        print('Write slip contour for value {} to file {}'.format(self.contourvalue, filename))

        # open the file
        fout = open(filename, 'w')

        # Loop on the contours
        for c in self.contourll:
            # Write segment separator
            fout.write('> \n')
            # Write segment
            for i in range(c[0].shape[0]):
                fout.write('{} {} {} \n'.format(c[0][i], c[1][i], c[2][i]))

        # close the file
        fout.close()

        # all done
        return

    def Patch2Trace(self):
        '''
        Builds a fake trace from the patches.
        @todo plot only the shallowest fault trace
        '''

        # Create a depth list
        dlist = []

        # Loop
        for p in self.patch:
            x, y, z, width, length, strike, dip = self.getpatchgeometry(p)
            dlist.append(z)

        # array
        dlist = np.array(dlist)

        # mindepth
        md = dlist.min()

        # Loop
        lon = []
        lat = []
        for p in self.patch:
            x, y, z, width, length, strike, dip = self.getpatchgeometry(p)
            lo, la = self.xy2ll(x, y)
            lon.append(lo)
            lat.append(la)
        lat = np.array(lat)
        lon = np.array(lon)

        # set trace
        self.trace(lon, lat)

        # All done
        return



    def mergePatches(self, p1, p2):
        '''
        Merges 2 patches that have common corners.
        Args:
            * p1    : index of the patch #1.
            * p2    : index of the patch #2.
        '''

        print('Merging patches {} and {} into patch {}'.format(p1,p2,p1))

        # Get the patches
        patch1 = self.patch[p1]
        patch2 = self.patch[p2]
        patch1ll = self.patchll[p1]
        patch2ll = self.patchll[p2]

        # Create the newpatches
        newpatch = np.zeros((4,3))
        newpatchll = np.zeros((4,3))

        # determine which corners are in common, needs at least two
        if ((patch1[0]==patch2[1]).all() and (patch1[3]==patch2[2]).all()):     # patch2 is above patch1
            newpatch[0,:] = patch2[0,:]; newpatchll[0,:] = patch2ll[0,:]
            newpatch[1,:] = patch1[1,:]; newpatchll[1,:] = patch1ll[1,:]
            newpatch[2,:] = patch1[2,:]; newpatchll[2,:] = patch1ll[2,:]
            newpatch[3,:] = patch2[3,:]; newpatchll[3,:] = patch2ll[3,:]
        elif ((patch1[3]==patch2[0]).all() and (patch1[2]==patch2[1]).all()):   # patch2 is on the right of patch1
            newpatch[0,:] = patch1[0,:]; newpatchll[0,:] = patch1ll[0,:]
            newpatch[1,:] = patch1[1,:]; newpatchll[1,:] = patch1ll[1,:]
            newpatch[2,:] = patch2[2,:]; newpatchll[2,:] = patch2ll[2,:]
            newpatch[3,:] = patch2[3,:]; newpatchll[3,:] = patch2ll[3,:]
        elif ((patch1[1]==patch2[0]).all() and (patch1[2]==patch2[3]).all()):   # patch2 is under patch1
            newpatch[0,:] = patch1[0,:]; newpatchll[0,:] = patch1ll[0,:]
            newpatch[1,:] = patch2[1,:]; newpatchll[1,:] = patch2ll[1,:]
            newpatch[2,:] = patch2[2,:]; newpatchll[2,:] = patch2ll[2,:]
            newpatch[3,:] = patch1[3,:]; newpatchll[3,:] = patch1ll[3,:]
        elif ((patch1[0]==patch2[3]).all() and (patch1[1]==patch2[2]).all()):   # patch2 is on the left of patch1
            newpatch[0,:] = patch2[0,:]; newpatchll[0,:] = patch2ll[0,:]
            newpatch[1,:] = patch2[1,:]; newpatchll[1,:] = patch2ll[1,:]
            newpatch[2,:] = patch1[2,:]; newpatchll[2,:] = patch1ll[2,:]
            newpatch[3,:] = patch1[3,:]; newpatchll[3,:] = patch1ll[3,:]
        else:
            print('Patches do not have common corners...')
            return

        # Replace the patch 1 by the new patch
        self.patch[p1] = newpatch
        self.patchll[p1] = newpatchll

        # Delete the patch 2
        self.deletepatch(p2)

        # All done
        return

    def writePatches2File(self, filename, add_slip=None, scale=1.0):
        '''
        Writes the patch corners in a file that can be used in psxyz.
        Args:
            * filename      : Name of the file.
            * add_slip      : Put the slip as a value for the color. Can be None, strikeslip, dipslip, total.
            * scale         : Multiply the slip value by a factor.
        '''

        # Write something
        print('Writing geometry to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # Loop over the patches
        for p in range(len(self.patchll)):

            # Select the string for the color
            string = '  '
            if add_slip is not None:
                if add_slip is 'strikeslip':
                    slp = self.slip[p,0]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'dipslip':
                    slp = self.slip[p,1]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'total':
                    slp = np.sqrt(self.slip[p,0]**2 + self.slip[p,1]**2)*scale
                    string = '-Z{}'.format(slp)

            # Put the parameter number in the file as well if it exists
            parameter = ' '
            if hasattr(self,'index_parameter'):
                i = np.int(self.index_parameter[p,0])
                j = np.int(self.index_parameter[p,1])
                k = np.int(self.index_parameter[p,2])
                parameter = '# {} {} {} '.format(i,j,k)

            # Put the slip value
            slipstring = ' # {} {} {} '.format(self.slip[p,0], self.slip[p,1], self.slip[p,2])

            # Write the string to file
            fout.write('> {} {} {}  \n'.format(string,parameter,slipstring))

            # Write the 4 patch corners (the order is to be GMT friendly)
            p = self.patchll[p]
            pp=p[1]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[0]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[3]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp=p[2]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))

        # Close th file
        fout.close()

        # All done
        return

    def getslip(self, p):
        '''
        Returns the slip vector for a patch.
        '''

        # output index
        io = None

        # Find the index of the patch
        for i in range(len(self.patch)):
            if (self.patch[i]==p).all():
                io = i

        # All done
        return self.slip[io,:]

    def writeSlipDirection2File(self, filename, scale=1.0, factor=1.0, neg_depth=False):
        '''
        Write a psxyz compatible file to draw lines starting from the center of each patch,
        indicating the direction of slip.
        Tensile slip is not used...
        scale can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        '''

        # Copmute the slip direction
        self.computeSlipDirection(scale=scale, factor=factor)

        # Write something
        print('Writing slip direction to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # Loop over the patches
        for p in self.slipdirection:

            # Write the > sign to the file
            fout.write('> \n')

            # Get the center of the patch
            xc, yc, zc = p[0]
            lonc, latc = self.xy2ll(xc, yc)
            if neg_depth:
                zc = -1.0*zc
            fout.write('{} {} {} \n'.format(lonc, latc, zc))

            # Get the end of the vector
            xc, yc, zc = p[1]
            lonc, latc = self.xy2ll(xc, yc)
            if neg_depth:
                zc = -1.0*zc
            fout.write('{} {} {} \n'.format(lonc, latc, zc))

        # Close file
        fout.close()

        # all done
        return

    def computeSlipDirection(self, scale=1.0, factor=1.0):
        '''
        Computes the segment indicating the slip direction.
        scale can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        '''

        # Create the array
        self.slipdirection = []

        # Loop over the patches
        for p in self.patch:

            # Get some geometry
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            # Get the slip vector
            slip = self.getslip(p)
            rake = np.arctan(slip[1]/slip[0])

            # Compute the vector
            x = np.sin(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake)
            y = np.cos(strike)*np.cos(rake) - np.sin(strike)*np.cos(dip)*np.sin(rake)
            z = np.sin(dip)*np.sin(rake)

            # Scale these
            if scale.__class__ is float:
                sca = scale
            elif scale.__class__ is str:
                if scale is 'total':
                    sca = np.sqrt(slip[0]**2 + slip[1]**2 + slip[2]**2)*factor
                elif scale is 'strikeslip':
                    sca = slip[0]*factor
                elif scale is 'dipslip':
                    sca = slip[1]*factor
                elif scale is 'tensile':
                    sca = slip[2]*factor
                else:
                    print('Unknown Slip Direction in computeSlipDirection')
                    sys.exit(1)
            x *= sca
            y *= sca
            z *= sca

            # update point
            xe = xc + x
            ye = yc + y
            ze = zc + z

            # Append
            self.slipdirection.append([[xc, yc, zc],[xe, ye, ze]])

        # All done
        return

    def deletepatch(self, patch):
        '''
        Deletes a patch.
        Args:
            * patch     : index of the patch to remove.
        '''

        # Remove the patch
        del self.patch[patch]
        del self.patchll[patch]
        self.slip = np.delete(self.slip, patch, axis=0)

        # All done
        return

    def deletepatches(self, tutu):
        '''
        Deletes a list of patches.
        '''

        while len(tutu)>0:

            # Get index to delete
            i = tutu.pop()

            # delete it
            self.deletepatch(i)

            # Upgrade list
            for u in range(len(tutu)):
                if tutu[u]>i:
                    tutu[u] -= 1

        # All done
        return

    def addpatch(self, patch, slip=[0, 0, 0]):
        '''
        Adds a patch to the list.
        Args:
            * patch     : Geometry of the patch to add
            * slip      : List of the strike, dip and tensile slip.
        '''

        # append the patch
        self.patch.append(patch)

        # modify the slip
        sh = self.slip.shape
        nl = sh[0] + 1
        nc = 3
        tmp = np.zeros((nl, nc))
        if nl > 1:                      # Case where slip is empty
            tmp[:nl-1,:] = self.slip
        tmp[-1,:] = slip
        self.slip = tmp

        # All done
        return

    def getpatchgeometry(self, patch, center=False):
        '''
        Returns the patch geometry as needed for okada85.
        Args:
            * patch         : index of the wanted patch or patch;
            * center        : if true, returns the coordinates of the center of the patch. if False, returns the UL corner.
        '''

        # Get the patch
        if patch.__class__ is int:
            p = self.patch[patch]
        else:
            p = patch

        # Get the UL corner of the patch
        if center:
            x1 = (p[1,0] + p[2,0])/2.
            x2 = (p[1,1] + p[2,1])/2.
            x3 = (p[1,2] + p[0,2])/2.
        else:
            x1 = p[0,0]
            x2 = p[0,1]
            x3 = p[0,2]

        # Get the patch width (this fault is vertical for now)
        width = p[2,2] - p[0,2]

        # Get the length
        length = np.sqrt( (p[2,0] - p[1,0])**2 + (p[2,1] - p[1,1])**2 )

        # Get the strike
        strike = np.arctan( (p[2,0] - p[1,0])/(p[2,1] - p[1,1]) ) + np.pi

        # Set the dip
        dip = np.pi*90. / 180.

        # All done
        return x1, x2, x3, width, length, strike, dip

    def slip2dis(self, data, patch, slip=None):
        '''
        Computes the surface displacement at the data location using okada.

        Args:
            * data          : data object from gps or insar.
            * patch         : number of the patch that slips
            * slip          : if a number is given, that is the amount of slip along strike
                              if three numbers are given, that is the amount of slip along strike, along dip and opening
                              if None, values from slip are taken
        '''

        # Set the slip values
        if slip is None:
            SLP = [self.slip[patch,0], self.slip[patch,1], self.slip[patch,2]]
        elif slip.__class__ is float:
            SLP = [slip, 0.0, 0.0]
        elif slip.__class__ is list:
            SLP = slip

        # Get patch geometry
        x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(patch)

        # Get data position
        x = data.x
        y = data.y

        # Allocate displacement lists
        ss_dis = []
        ds_dis = []
        op_dis = []

        for i in range(len(x)):

            # Run okada for strike slip
            ss = ok.displacement(x[i], y[i], dip, x1, x2, x3, length, width, strike, 1)
            ss_dis.append(ss*SLP[0])

            # Run okada for dip slip
            ds = ok.displacement(x[i], y[i], dip, x1, x2, x3, length, width, strike, 2)
            ds_dis.append(ds*SLP[1])

            # Run okada for opening
            op = ok.displacement(x[i], y[i], dip, x1, x2, x3, length, width, strike, 3)
            op_dis.append(op*SLP[2])

        # Make arrays
        ss_dis = np.array(ss_dis)
        ds_dis = np.array(ds_dis)
        op_dis = np.array(op_dis)

        # All done
        return ss_dis, ds_dis, op_dis

    def buildGFs(self, data, vertical=True, slipdir='sd'):
        '''
        Builds the Green's function matrix based on the discretized fault.
        Args:
            * data      : data object from gps or insar.
            * vertical  : if True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir   : direction of the slip along the patches. can be any combination of s (strikeslip), d (dipslip) and t (tensile).

        The Green's function matrix is stored in a dictionary. Each entry of the dictionary is named after the corresponding dataset. Each of these entry is a dictionary that contains 'strikeslip', 'dipslip' and/or 'tensile'.
        '''

        print ("Building Green's functions for the data set {} of type {}".format(data.name, data.dtype))

        # Get the number of data
        Nd = data.lon.shape[0]
        if data.dtype is 'insar':
            Ndt = Nd
            data.obs_per_station = 1
        elif data.dtype is 'gps':
            Ndt = data.lon.shape[0]*2
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station = 3
                Ndt += data.lon.shape[0]

        # Get the number of parameters
        Np = len(self.patch)
        Npt = len(self.patch)*len(slipdir)

        # Initializes a space in the dictionary to store the green's function
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]
        if 's' in slipdir:
            G['strikeslip'] = np.zeros((Ndt, Np))
        if 'd' in slipdir:
            G['dipslip'] = np.zeros((Ndt, Np))
        if 't' in slipdir:
            G['tensile'] = np.zeros((Ndt, Np))

        # Initializes the data vector and the data covariance
        if data.dtype is 'insar':
            self.d[data.name] = data.vel
            vertical = True                 # In InSAR, you need to use the vertical, no matter what....
        elif data.dtype is 'gps':
            if vertical:
                self.d[data.name] = data.vel_enu.T.flatten()
            else:
                self.d[data.name] = data.vel_enu[:,0:2].T.flatten()

        # Initialize the slip vector
        SLP = []
        if 's' in slipdir:              # If strike slip is aksed
            SLP.append(1.0)
        else:                           # Else
            SLP.append(0.0)
        if 'd' in slipdir:              # If dip slip is asked
            SLP.append(1.0)
        else:                           # Else
            SLP.append(0.0)
        if 't' in slipdir:              # If tensile is asked
            SLP.append(1.0)
        else:                           # Else
            SLP.append(0.0)

        # import something
        import sys

        # Loop over each patch
        for p in range(len(self.patch)):
            sys.stdout.write('\r Patch: {} / {} '.format(p+1,len(self.patch)))
            sys.stdout.flush()

            # get the surface displacement corresponding to unit slip
            ss, ds, op = self.slip2dis(data, p, slip=SLP)

            # Do we keep the verticals
            if not vertical:
                ss = ss[:,0:2]
                ds = ds[:,0:2]
                op = op[:,0:2]

            # Organize the response
            if data.dtype is 'gps':            # If GPS type, just put them in the good format
                ss = ss.T.flatten()
                ds = ds.T.flatten()
                op = op.T.flatten()
            elif data.dtype is 'insar':        # If InSAR, do the dot product with the los
                ss_los = []
                ds_los = []
                op_los = []
                for i in range(Nd):
                    ss_los.append(np.dot(data.los[i,:], ss[i,:]))
                    ds_los.append(np.dot(data.los[i,:], ds[i,:]))
                    op_los.append(np.dot(data.los[i,:], op[i,:]))
                ss = ss_los
                ds = ds_los
                op = op_los

            # Store these guys in the corresponding G slot
            if 's' in slipdir:
                G['strikeslip'][:,p] = ss
            if 'd' in slipdir:
                G['dipslip'][:,p] = ds
            if 't' in slipdir:
                G['tensile'][:,p] = op

        # Clean the screen
        sys.stdout.write('\n')
        sys.stdout.flush()

        # All done
        return

    def saveGFs(self, dtype='d'):
        '''
        Saves the Green's functions in different files
        Args:
            dtype       : Format of the binary data saved.
        '''

        # Print stuff
        print('Writing Greens functions to file for fault {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # StrikeSlip Component
            if 'strikeslip' in G.keys():
                gss = G['strikeslip'].flatten()
                filename = '{}_{}_SS.gf'.format(self.name, data)
                gss = gss.astype(dtype)
                gss.tofile(filename)

            # DipSlip Component
            if 'dipslip' in G.keys():
                gds = G['dipslip'].flatten()
                filename = '{}_{}_DS.gf'.format(self.name, data)
                gds = gds.astype(dtype)
                gds.tofile(filename)

            # Tensile
            if 'tensile' in G.keys():
                gts = G['tensile'].flatten()
                filename = '{}_{}_TS.gf'.format(self.name, data)
                gts = gts.astype(dtype)
                gts.tofile(filename)

        # All done
        return

    def setGFsFromFile(self, data, strikeslip=None, dipslip=None, tensile=None, vertical=False, dtype='d'):
        '''
        Sets the Green's functions from binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already)
        Args:
            * data          : Data structure from gps or insar.
            * strikeslip    : File containing the Green's functions for strikeslip displacements.
            * dipslip       : File containing the Green's functions for dipslip displacements.
            * tensile       : File containing the Green's functions for tensile displacements.
            * vertical      : Deal with the UP component (gps: default is false, insar, it will be true anyway).
            * dtype         : Type of binary data.
        '''

        print('---------------------------------')
        print('---------------------------------')
        print("Set up Green's functions for fault {} from files {}, {} and {}".format(self.name, strikeslip, dipslip, tensile))

        # Get the number of patches
        Npatchs = len(self.patch)

        # Read the files and reshape the GFs
        if strikeslip is not None:
            Gss = np.fromfile(strikeslip, dtype=dtype)
            ndl = int(Gss.shape[0]/Npatchs)
            Gss = Gss.reshape((ndl, Npatchs))
        if dipslip is not None:
            Gds = np.fromfile(dipslip, dtype=dtype)
            ndl = int(Gds.shape[0]/Npatchs)
            Gds = Gds.reshape((ndl, Npatchs))
        if tensile is not None:
            Gts = np.fromfile(tensile, dtype=dtype)
            ndl = int(Gts.shape[0]/Npatchs)
            Gts = Gts.reshape((ndl, Npatchs))

        # Get the data type
        datatype = data.dtype

        # Cut the Matrices following what data do we have and set the GFs
        if datatype is 'gps':

            # Initialize
            GssE = None; GdsE = None; GtsE = None
            GssN = None; GdsN = None; GtsN = None
            GssU = None; GdsU = None; GtsU = None

            # Get the values
            if strikeslip is not None:
                GssE = Gss[range(0,data.vel_enu.shape[0]),:]
                GssN = Gss[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GssU = Gss[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]
            if dipslip is not None:
                GdsE = Gds[range(0,data.vel_enu.shape[0]),:]
                GdsN = Gds[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GdsU = Gds[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]
            if tensile is not None:
                GtsE = Gts[range(0,data.vel_enu.shape[0]),:]
                GtsN = Gts[range(data.vel_enu.shape[0],data.vel_enu.shape[0]*2),:]
                if vertical:
                    GtsU = Gts[range(data.vel_enu.shape[0]*2,data.vel_enu.shape[0]*3),:]

            # set the GFs
            self.setGFs(data, strikeslip=[GssE, GssN, GssU], dipslip=[GdsE, GdsN, GdsU], tensile=[GtsE, GtsN, GtsU], vertical=vertical)

        elif datatype is 'insar':

            # Initialize
            GssLOS = None; GdsLOS = None; GtsLOS = None

            # Get the values
            if strikeslip is not None:
                GssLOS = Gss
            if dipslip is not None:
                GdsLOS = Gds
            if tensile is not None:
                GtsLOS = Gts

            # set the GFs
            self.setGFs(data, strikeslip=[GssLOS], dipslip=[GdsLOS], tensile=[GtsLOS], vertical=True)

        # all done
        return

    def setGFs(self, data, strikeslip=[None, None, None], dipslip=[None, None, None], tensile=[None, None, None], vertical=False):
        '''
        Stores the Green's functions matrices into the fault structure.
        Args:
            * data          : Data structure from gps or insar.
            * strikeslip    : List of matrices of the Strikeslip Green's functions, ordered E, N, U
            * dipslip       : List of matrices of the dipslip Green's functions, ordered E, N, U
            * tensile       : List of matrices of the tensile Green's functions, ordered E, N, U
            If you provide InSAR GFs, these need to be projected onto the LOS direction already.
        '''

        # Get the number of data per point
        if data.dtype is 'insar':
            data.obs_per_station = 1
        elif data.dtype is 'gps':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station = 3

        # Create the storage for that dataset
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]

        # Initializes the data vector
        if data.dtype is 'insar':
            self.d[data.name] = data.vel
            vertical = True                 # In InSAR, you need to use the vertical, no matter what....
        elif data.dtype is 'gps':
            if vertical:
                self.d[data.name] = data.vel_enu.T.flatten()
            else:
                self.d[data.name] = data.vel_enu[:,0:2].T.flatten()

        # StrikeSlip
        if len(strikeslip) == 3:            # GPS case

            E_ss = strikeslip[0]
            N_ss = strikeslip[1]
            U_ss = strikeslip[2]
            ss = []
            nd = 0
            if (E_ss is not None) and (N_ss is not None):
                d = E_ss.shape[0]
                m = E_ss.shape[1]
                ss.append(E_ss)
                ss.append(N_ss)
                nd += 2
            if (U_ss is not None):
                d = U_ss.shape[0]
                m = U_ss.shape[1]
                ss.append(U_ss)
                nd += 1
            if nd > 0:
                ss = np.array(ss)
                ss = ss.reshape((nd*d, m))
                G['strikeslip'] = ss

        elif len(strikeslip) == 1:          # InSAR case

            LOS_ss = strikeslip[0]
            if LOS_ss is not None:
                G['strikeslip'] = LOS_ss

        # DipSlip
        if len(dipslip) == 3:               # GPS case
            E_ds = dipslip[0]
            N_ds = dipslip[1]
            U_ds = dipslip[2]
            ds = []
            nd = 0
            if (E_ds is not None) and (N_ds is not None):
                d = E_ds.shape[0]
                m = E_ds.shape[1]
                ds.append(E_ds)
                ds.append(N_ds)
                nd += 2
            if (U_ds is not None):
                d = U_ds.shape[0]
                m = U_ds.shape[1]
                ds.append(U_ds)
                nd += 1
            if nd > 0:
                ds = np.array(ss)
                ds = ds.reshape((nd*d, m))
                G['dipslip'] = ds

        elif len(dipslip) == 1:             # InSAR case

            LOS_ds = dipslip[0]
            if LOS_ds is not None:
                G['dipslip'] = LOS_ds

        # StrikeSlip
        if len(tensile) == 3:               # GPS case

            E_ts = tensile[0]
            N_ts = tensile[1]
            U_ts = tensile[2]
            ts = []
            nd = 0
            if (E_ts is not None) and (N_ts is not None):
                d = E_ts.shape[0]
                m = E_ts.shape[1]
                ts.append(E_ts)
                ts.append(N_ts)
                nd += 2
            if (U_ts is not None):
                d = U_ts.shape[0]
                m = U_ts.shape[1]
                ts.append(U_ts)
                nd += 1
            if nd > 0:
                ts = np.array(ts)
                ts = ts.reshape((nd*d, m))
                G['tensile'] = ts

        elif len(tensile) == 1:             # InSAR Case

            LOS_ts = tensile[0]
            if LOS_ts is not None:
                G['dipslip'] = LOS_ds

        # All done
        return

    def assembleGFs(self, datas, polys=0, slipdir='sd'):
        '''
        Assemble the Green's functions that have been built using build GFs.
        This routine spits out the General G and the corresponding data vector d.
        Args:
            * datas         : data sets to use as inputs (from gps and insar).
            * polys         : 0 -> nothing additional is estimated
                              1 -> estimate a constant offset
                              3 -> estimate z = ax + by + c
                              4 -> estimate z = axy + bx + cy + d
                              'full' -> Only for GPS, estimates a rotation, translation and scaling with
                                        respect to the center of the network (Helmert transform).
            * slipdir       : which directions of slip to include. can be any combination of s, d and t.
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print("Assembling G for fault {}".format(self.name))

        # Store the assembled slip directions
        self.slipdir = slipdir

        # Create a dictionary to keep track of the orbital froms
        self.poly = {}

        # Set poly right
        if polys.__class__ is not list:
            for data in datas:
                if polys.__class__ is not str:
                    self.poly[data.name] = polys*data.obs_per_station
                else:
                    if data.dtype is 'gps':
                        self.poly[data.name] = polys
                    else:
                        print('Data type must be gps to implement a Helmert transform')
                        return
        elif polys.__class__ is list:
            for d in range(len(datas)):
                if polys[d].__class__ is not str:
                    self.poly[datas[d].name] = polys[d]*datas[d].obs_per_station
                else:
                    if datas[d].dtype is 'gps':
                        self.poly[datas[d].name] = polys[d]
                    else:
                        print('Data type must be gps to implement a Helmert transform')
                        return

        # Get the number of parameters
        N = len(self.patch)
        Nps = N*len(slipdir)
        Npo = 0
        self.helmert = {}
        for data in datas :
            if self.poly[data.name] is 'full':
                if data.obs_per_station==3:
                    Npo += 7                    # 3D Helmert transform is 7 parameters
                    self.helmert[data.name] = 7
                else:
                    Npo += 4                    # 2D Helmert transform is 4 parameters
                    self.helmert[data.name] = 4
            else:
                Npo += (self.poly[data.name])
        Np = Nps + Npo

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Build the desired slip list
        sliplist = []
        if 's' in slipdir:
            sliplist.append('strikeslip')
        if 'd' in slipdir:
            sliplist.append('dipslip')
        if 't' in slipdir:
            sliplist.append('tensile')

        # Allocate G and d
        G = np.zeros((Nd, Np))

        # Create the list of data names, to keep track of it
        self.datanames = []

        # loop over the datasets
        el = 0
        polstart = Nps
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            print("Dealing with {} of type {}".format(data.name, data.dtype))

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Glocal = np.zeros((Ndlocal, Nps))

            # Fill Glocal
            ec = 0
            for sp in sliplist:
                Glocal[:,ec:ec+N] = self.G[data.name][sp]
                ec += N

            # Put Glocal into the big G
            G[el:el+Ndlocal,0:Nps] = Glocal

            # Build the polynomial function
            if self.poly[data.name].__class__ is not str:
                if self.poly[data.name] > 0:
                    if data.dtype is 'gps':
                        orb = np.zeros((Ndlocal, self.poly[data.name]))
                        nn = Ndlocal/data.obs_per_station
                        orb[:nn, 0] = 1.0
                        orb[nn:2*nn, 1] = 1.0
                        if data.obs_per_station == 3:
                            orb[2*nn:3*nn, 2] = 1.0
                    elif data.dtype is 'insar':
                        orb = np.zeros((Ndlocal, self.poly[data.name]))
                        orb[:] = 1.0
                        if self.poly[data.name] >= 3:
                            orb[:,1] = data.x/np.abs(data.x).max()
                            orb[:,2] = data.y/np.abs(data.y).max()
                        if self.poly[data.name] >= 4:
                            orb[:,3] = (data.x/np.abs(data.x).max())*(data.y/np.abs(data.y).max())

                    # Put it into G for as much observable per station we have
                    polend = polstart + self.poly[data.name]
                    G[el:el+Ndlocal, polstart:polend] = orb
                    polstart += self.poly[data.name]
            else:
                if self.poly[data.name] is 'full':
                    orb = self.getHelmertMatrix(data)
                    if data.obs_per_station==3:
                        nc = 7
                    elif data.obs_per_station==2:
                        nc = 4
                    # Put it into G for as much observable per station we have
                    polend = polstart + nc
                    G[el:el+Ndlocal, polstart:polend] = orb
                    polstart += nc

            # Update el to check where we are
            el = el + Ndlocal

        # Store G in self
        self.Gassembled = G

        # All done
        return

    def getHelmertMatrix(self, data):
        '''
        Returns a Helmert matrix for a gps data set.
        '''

        # Check
        assert (data.dtype is 'gps')

        # Get the number of stations
        ns = data.station.shape[0]

        # Get the data vector size
        nd = self.d[data.name].shape[0]

        # Get the number of helmert transform parameters
        if data.obs_per_station==3:
            nc = 7
        else:
            nc = 4

        # Check something
        assert data.obs_per_station*ns==nd

        # Get the position of the center of the network
        x0 = np.mean(data.x)
        y0 = np.mean(data.y)
        z0 = 0              # We do not deal with the altitude of the stations yet (later)

        # Compute the baselines
        base_x = data.x - x0
        base_y = data.y - y0
        base_z = 0

        # Normalize the baselines
        base_x_max = np.abs(base_x).max(); base_x /= base_x_max
        base_y_max = np.abs(base_y).max(); base_y /= base_y_max

        # Allocate a Helmert base
        H = np.zeros((data.obs_per_station,nc))

        # put the translation in it (that part never changes)
        H[:,:data.obs_per_station] = np.eye(data.obs_per_station)

        # Allocate the full matrix
        Hf = np.zeros((nd,nc))

        # Loop over the stations
        for i in range(ns):

            # Clean the part that changes
            H[:,data.obs_per_station:] = 0.0

            # Put the rotation components and the scale components
            x1, y1, z1 = base_x[i], base_y[i], base_z
            if nc==7:
                H[:,3:6] = np.array([[0.0, -z1, y1],
                                     [z1, 0.0, -x1],
                                     [-y1, x1, 0.0]])
                H[:,7] = np.array([x1, y1, z1])
            else:
                H[:,2] = np.array([y1, -x1])
                H[:,3] = np.array([x1, y1])

            # put the lines where they should be
            Hf[i,:] = H[0]
            Hf[i+ns,:] = H[1]
            if nc==7:
                Hf[i+2*ns,:] = H[2]

        # all done
        return Hf

    def assembled(self, datas):
        '''
        Assembles the data vector corresponding to the stored green's functions.
        Args:
            * datas         : list of the data object involved (from gps and insar).
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Assembling d vector")

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Create a data vector
        d = np.zeros((Nd,))

        # Loop over the datasets
        el = 0
        for data in datas:

                # print
                print("Dealing with data {}".format(data.name))

                # Get the local d
                dlocal = self.d[data.name]
                Ndlocal = dlocal.shape[0]

                # Store it in d
                d[el:el+Ndlocal] = dlocal

                # update el
                el += Ndlocal

        # Store d in self
        self.dassembled = d

        # All done
        return

    def assembleCd(self, datas):
        '''
        Assembles the data covariance matrixes that have been built by each data structure.
        '''

        # Check if the Green's function are ready
        if self.Gassembled is None:
            print("You should assemble the Green's function matrix first")
            return

        # Get the total number of data
        Nd = self.Gassembled.shape[0]
        Cd = np.zeros((Nd, Nd))

        # Loop over the data sets
        st = 0
        for data in datas:
            se = st + self.d[data.name].shape[0]
            Cd[st:se, st:se] = data.Cd
            st += self.d[data.name].shape[0]

        # Store Cd in self
        self.Cd = Cd

        # All done
        return

    def buildCm(self, sigma, lam, lam0=None, extra_params=None, lim=None):
        '''
        Builds a model covariance matrix using the equation described in Radiguet et al 2010.
        Args:
            * sigma         : Amplitude of the correlation.
            * lam           : Characteristic length scale.
            * lam0          : Normalizing distance (if None, lam0=min(distance between patches)).
            * extra_params  : Add some extra values on the diagonal.
            * lim           : Limit distance parameter (see self.distancePatchToPatch)
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Assembling the Cm matrix ")
        print ("Sigma = {}".format(sigma))
        print ("Lambda = {}".format(lam))

        # Need the patch geometry
        if self.patch is None:
            print("You should build the patches and the Green's functions first.")
            return

        # Geth the desired slip directions
        slipdir = self.slipdir

        # Get the patch centers
        self.centers = np.array(self.getcenters())

        # Sets the lambda0 value
        if lam0 is None:
            xd = (np.unique(self.centers[:,0]).max() - np.unique(self.centers[:,0]).min())/(np.unique(self.centers[:,0]).size)
            yd = (np.unique(self.centers[:,1]).max() - np.unique(self.centers[:,1]).min())/(np.unique(self.centers[:,1]).size)
            zd = (np.unique(self.centers[:,2]).max() - np.unique(self.centers[:,2]).min())/(np.unique(self.centers[:,2]).size)
            lam0 = np.sqrt( xd**2 + yd**2 + zd**2 )
        print ("Lambda0 = {}".format(lam0))
        C = (sigma*lam0/lam)**2

        # Creates the principal Cm matrix
        Np = len(self.patch)*len(slipdir)
        if extra_params is not None:
            Np += len(extra_params)
        Cmt = np.zeros((len(self.patch), len(self.patch)))
        Cm = np.zeros((Np, Np))

        # Loop over the patches
        i = 0
        for p1 in self.patch:
            j = 0
            for p2 in self.patch:
                # Compute the distance
                d = self.distancePatchToPatch(p1, p2, distance='center', lim=lim)
                # Compute Cm
                Cmt[i,j] = C * np.exp( -1.0*d/lam)
                Cmt[j,i] = C * np.exp( -1.0*d/lam)
                # Upgrade counter
                j += 1
            # upgrade counter
            i += 1

        # Store that into Cm
        st = 0
        for i in range(len(slipdir)):
            se = st + len(self.patch)
            Cm[st:se, st:se] = Cmt
            st += len(self.patch)

        # Put the extra values
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm

        # All done
        return

    def distancePatchToPatch(self, patch1, patch2, distance='center', lim=None):
        '''
        Measures the distance between two patches.
        Args:
            * patch1    : geometry of the first patch.
            * patch2    : geometry of the second patch.
            * distance  : distance estimation mode
                            center : distance between the centers of the patches.
                            no other method is implemented for now.
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].
        '''

        if distance is 'center':

            # Get the centers
            x1, y1, z1 = self.getcenter(patch1)
            x2, y2, z2 = self.getcenter(patch2)

            # Compute the distance
            dis = np.sqrt( (x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis>lim[0]:
                    dis = lim[1]

        # All done
        return dis

    def writeEDKSsubParams(self, data, edksfilename, amax=None, plot=False):
        '''
        Write the subParam file needed for the interpolation of the green's function in EDKS.
        Francisco's program cuts the patches into small patches, interpolates the kernels to get the GFs at each point source,
        then averages the GFs on the pacth. To decide the size of the minimum patch, it uses St Vernant's principle.
        If amax is specified, the minimum size is fixed.
        Args:
            * data          : Data object from gps or insar.
            * edksfilename  : Name of the file containing the kernels.
            * amax          : Specifies the minimum size of the divided patch. If None, uses St Vernant's principle.
            * plot          : Activates plotting.
        Returns:
            * filename      : Name of the subParams file created.
        '''

        # print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Write the EDKS files for fault {} and data {}".format(self.name, data.name))

        # Write the geometry to the EDKS file
        self.writeEDKSgeometry()

        # Write the data to the EDKS file
        data.writeEDKSdata()

        # Create the variables
        RectanglePropFile = 'edks_{}.END'.format(self.name)
        ReceiverFile = 'edks_{}.idEN'.format(data.name)
        if data.dtype is 'insar':
            useRecvDir = True # True for InSAR, uses LOS information
        else:
            useRecvDir = False # False for GPS, uses ENU displacements
        EDKSunits = 1000.0
        EDKSfilename = '{}'.format(edksfilename)
        prefix = 'edks_{}_{}'.format(self.name, data.name)
        plotGeometry = '{}'.format(plot)

        # Open the EDKSsubParams.py file
        filename = 'EDKSParams_{}_{}.py'.format(self.name, data.name)
        fout = open(filename, 'w')

        # Write in it
        fout.write("# File with the rectangles properties\n")
        fout.write("RectanglesPropFile = '{}'\n".format(RectanglePropFile))
        fout.write("# File with id, E[km], N[km] coordinates of the receivers.\n")
        fout.write("ReceiverFile = '{}'\n".format(ReceiverFile))
        fout.write("# read receiver direction (# not yet implemented)\n")
        fout.write("useRecvDir = {} # True for InSAR, uses LOS information\n".format(useRecvDir))
        fout.write("# Maximum Area to subdivide triangles. If None, uses Saint-Venant's principle.\n")
        if amax is None:
            fout.write("Amax = None # None computes Amax automatically. \n")
        else:
            fout.write("Amax = {} # Minimum size for the patch division.\n".format(amax))

        fout.write("EDKSunits = 1000.0 # to convert from kilometers to meters\n")
        fout.write("EDKSfilename = '{}'\n".format(edksfilename))
        fout.write("prefix = '{}'\n".format(prefix))
        fout.write("plotGeometry = {} # set to False if you are running in a remote Workstation\n".format(plot))

        # Close the file
        fout.close()

        # Build usefull outputs
        parNames = ['useRecvDir', 'Amax', 'EDKSunits', 'EDKSfilename', 'prefix']
        parValues = [ useRecvDir ,  amax ,  EDKSunits ,  EDKSfilename ,  prefix ]
        method_par = dict(zip(parNames, parValues))

        # All done
        return filename, RectanglePropFile, ReceiverFile, method_par

    def writeEDKSgeometry(self, ref=None):
        '''
        This routine spits out 2 files:
        filename.lonlatdepth: Lon center | Lat Center | Depth Center (km) | Strike | Dip | Length (km) | Width (km) | patch ID
        filename.END: Easting (km) | Northing (km) | Depth Center (km) | Strike | Dip | Length (km) | Width (km) | patch ID

        These files are to be used with /home/geomod/dev/edks/MPI_EDKS/calcGreenFunctions_EDKS_subRectangles.py

        Args:
            * ref           : Lon and Lat of the reference point. If None, the patches positions is in the UTM coordinates.
        '''

        # Filename
        filename = 'edks_{}'.format(self.name)

        # Open the output file
        flld = open(filename+'.lonlatdepth','w')
        flld.write('#lon lat Dep[km] strike dip length(km) width(km) ID\n')
        fend = open(filename+'.END','w')
        fend.write('#Easting[km] Northing[km] Dep[km] strike dip length(km) width(km) ID\n')

        # Reference
        if ref is not None:
            refx, refy = self.putm(ref[0], ref[1])
            refx /= 1000.
            refy /= 1000.

        # Loop over the patches
        for p in range(len(self.patch)):
            x, y, z, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            strike = strike*180./np.pi
            dip = dip*180./np.pi
            lon, lat = self.xy2ll(x,y)
            if ref is not None:
                x -= refx
                y -= refy
            flld.write('{} {} {} {} {} {} {} {:5d} \n'.format(lon,lat,z,strike,dip,length,width,p))
            fend.write('{} {} {} {} {} {} {} {:5d} \n'.format(x,y,z,strike,dip,length,width,p))

        # Close the files
        flld.close()
        fend.close()

        # All done
        return

    def getcenters(self):
        '''
        Get the center of the patches.
        '''

        # Get the patches
        patch = self.patch

        # Initialize a list
        center = []

        # loop over the patches
        for p in patch:
            x, y, z = self.getcenter(p)
            center.append([x, y, z])

        # All done
        return center

    def getcenter(self, p):
        '''
        Get the center of one patch.
        Args:
            * p    : Patch geometry.
        '''

        # Get center
        x = (p[1][0] + p[2][0])/2.
        y = (p[1][1] + p[2][1])/2.
        z = (p[1][2] + p[0][2])/2.

        # All done
        return x,y,z

    def surfacesimulation(self, box=None, disk=None, err=None, npoints=None):
        '''
        Takes the slip vector and computes the surface displacement that corresponds on a regular grid.
        Args:
            * box       : Can be a list of [minlon, maxlon, minlat, maxlat].
            * disk      : list of [xcenter, ycenter, radius, n]
        '''

        # create a fake gps object
        self.sim = gpsclass('simulation', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0)

        # Create a lon lat grid
        if (box is None) and (disk is None) :
            lon = np.linspace(self.lon.min(), self.lon.max(), 100)
            lat = np.linspace(self.lat.min(), self.lat.max(), 100)
            lon, lat = np.meshgrid(lon,lat)
            lon = lon.flatten()
            lat = lat.flatten()
        elif (box is not None):
            lon = np.linspace(box[0], box[1], 100)
            lat = np.linspace(box[2], box[3], 100)
            lon, lat = np.meshgrid(lon,lat)
            lon = lon.flatten()
            lat = lat.flatten()
        elif (disk is not None):
            lon = []; lat = []
            xd, yd = self.ll2xy(disk[0], disk[1])
            xmin = xd-disk[2]; xmax = xd+disk[2]; ymin = yd-disk[2]; ymax = yd+disk[2]
            ampx = (xmax-xmin)
            ampy = (ymax-ymin)
            n = 0
            while n<disk[3]:
                x, y = np.random.rand(2)
                x *= ampx; x -= ampx/2.; x += xd
                y *= ampy; y -= ampy/2.; y += yd
                if ((x-xd)**2 + (y-yd)**2) <= (disk[2]**2):
                    lo, la = self.xy2ll(x,y)
                    lon.append(lo); lat.append(la)
                    n += 1
            lon = np.array(lon); lat = np.array(lat)

        if (lon.max()>360.) or (lon.min()<-180.0) or (lat.max()>90.) or (lat.min()<-90):
            self.sim.x = lon
            self.sim.y = lat
        else:
            self.sim.lon = lon
            self.sim.lat = lat
            # put these in x y utm coordinates
            self.sim.ll2xy()

        # Initialize the vel_enu array
        self.sim.vel_enu = np.zeros((lon.size, 3))

        # Create the station name array
        self.sim.station = []
        for i in range(len(self.sim.x)):
            name = '{:04d}'.format(i)
            self.sim.station.append(name)
        self.sim.station = np.array(self.sim.station)

        # Create an error array
        if err is not None:
            self.sim.err_enu = []
            for i in range(len(self.sim.x)):
                x,y,z = np.random.rand(3)
                x *= err
                y *= err
                z *= err
                self.sim.err_enu.append([x,y,z])
            self.sim.err_enu = np.array(self.sim.err_enu)

        # import stuff
        import sys

        # Loop over the patches
        for p in range(len(self.patch)):
            sys.stdout.write('\r Patch {} / {} '.format(p+1,len(self.patch)))
            sys.stdout.flush()
            # Get the surface displacement due to the slip on this patch
            ss, ds, op = self.slip2dis(self.sim, p)
            # Sum these to get the synthetics
            self.sim.vel_enu += ss
            self.sim.vel_enu += ds
            self.sim.vel_enu += op

        sys.stdout.write('\n')
        sys.stdout.flush()

        # All done
        return

    def computetotalslip(self):
        '''
        Computes the total slip.
        '''

        # Computes the total slip
        self.totalslip = np.sqrt(self.slip[:,0]**2 + self.slip[:,1]**2 + self.slip[:,2]**2)

        # All done
        return

    def plot(self, ref='utm', figure=134, add=False, maxdepth=None, show=True):
        '''
        Plot the available elements of the fault.

        Args:
            * ref           : Referential for the plot ('utm' or 'lonlat').
            * figure        : Number of the figure.
        '''

        # Import necessary things
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figure)
        ax = fig.add_subplot(111, projection='3d')

        # Title
        if hasattr(self,'eventtag'):
            ax.set_title(self.eventtag)

        # Set the axes
        if ref is 'utm':
            ax.set_xlabel('Easting (km)')
            ax.set_ylabel('Northing (km)')
        else:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth (km)')

        if add and (ref is 'utm'):
            for fault in self.addfaultsxy:
                ax.plot(fault[:,0], fault[:,1], '-k')
        elif add and (ref is not 'utm'):
            for fault in self.addfaults:
                ax.plot(fault[:,0], fault[:,1], '-k')

        # Plot the surface trace
        if ref is 'utm':
            if self.xf is None:
                self.trace2xy()
            ax.plot(self.xf, self.yf, '-b')
        else:
            ax.plot(self.lon, self.lat, '-b')


        # Compute the total slip
        self.computetotalslip()

        # Plot the patches
        if self.patch is not None:

            # import stuff
            import mpl_toolkits.mplot3d.art3d as art3d
            import matplotlib.colors as colors
            import matplotlib.cm as cmx

            # set z axis
            ax.set_zlim3d([-1.0*(self.depth+5), 0])

            # set color business
            cmap = plt.get_cmap('jet')
            cNorm  = colors.Normalize(vmin=0, vmax=self.totalslip.max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

            for p in range(len(self.patch)):
                ncorners = len(self.patch[0])
                x = []
                y = []
                z = []
                for i in range(ncorners):
                    if ref is 'utm':
                        x.append(self.patch[p][i][0])
                        y.append(self.patch[p][i][1])
                        z.append(-1.0*self.patch[p][i][2])
                    else:
                        x.append(self.patchll[p][i][0])
                        y.append(self.patchll[p][i][1])
                        z.append(-1.0*self.patchll[p][i][2])
                verts = [zip(x, y, z)]
                rect = art3d.Poly3DCollection(verts)
                rect.set_color(scalarMap.to_rgba(self.totalslip[p]))
                rect.set_edgecolors('k')
                ax.add_collection3d(rect)

            # put up a colorbar
            scalarMap.set_array(self.totalslip)
            plt.colorbar(scalarMap)

        # Contours
        if hasattr(self, 'contour'):
            for c in self.contour:
                verts = [zip(c[0], c[1], [-1.0*c[2][i] for i in range(len(c[2]))])]
                rect = art3d.Line3DCollection(verts)
                rect.set_edgecolors('r')
                ax.add_collection3d(rect)

        # Depth
        if maxdepth is not None:
            ax.set_zlim3d([-1.0*maxdepth, 0])

        ax.axis('equal')

        # show
        if show is True:
            plt.show()

        # All done
        return

#EOF
