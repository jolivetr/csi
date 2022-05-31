'''
A parent Pressure class

Written by T. Shreve, May 2019
'''

# Import Externals stuff
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os
from argparse import Namespace

# Personals
from .SourceInv import SourceInv
from .EDKSmp import sum_layered
from .EDKSmp import dropSourcesInPatches as Patches2Sources
from .gps import gps as gpsclass

# class Pressure
class Pressure(SourceInv):
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    # We use a static method here so the pressure class can be the parent class, but the object created has attributes and methods of a specific subclass
    @staticmethod
    def chooseSource(name, x0, y0, z0, ax, ay, az, dip, strike, plunge, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        '''
        Method used to initialize object as CDM or one of its degenerate cases (pCDM, mogi, or yang)

        Args:
            * x0, y0       : Center of pressure source in lat/lon or utm
            * z0           : Depth
            * ax, ay, az   : Semi-axes of the CDM along the x, y and z axes respectively, before applying rotations.
            * dip          : Clockwise around N-S (Y) axis; dip = 90 means vertical source
            * strike       : Clockwise from N; strike = 0 means source is oriented N-S
            * plunge       : Clockwise along E-W (X) axis
        '''
        if None not in {ax,ay,az}:
            c, b, a = np.sort([float(ax),float(ay),float(az)])
            if b == c:
                if a == b == c:
                    from .Mogi import Mogi
                    print('All axes are equal, using Mogi.py')
                    return Mogi(name, x0, y0, z0, ax, ay, az, dip, strike, plunge, utmzone=utmzone, ellps='WGS84', lon0=lon0, lat0=lon0, verbose=True)
                else:
                    from .Yang import Yang
                    print('semi-minor axes are equal, using Yang.py')
                    return Yang(name, x0, y0, z0, ax, ay, az, dip, strike, plunge, utmzone=utmzone, ellps='WGS84', lon0=lon0, lat0=lon0, verbose=True)
            else:
                from .CDM import CDM
                print('No analytical simplifications possible, using finite CDM.py!')
                return CDM(name, x0, y0, z0, ax, ay, az, dip, strike, plunge, utmzone=utmzone, ellps='WGS84', lon0=lon0, lat0=lon0, verbose=True)

        elif None in {ax, ay, az}:
            ### p. 884 of Nikkhoo, et al 2016. states that near field of pCDM and CDM are equivalent only for prolate ellipsoid, but how is "prolate ellipsoid" defined? (when two shortest axes are ~approximately~ the same size?)
            from .pCDM import pCDM
            print('Using pCDM.py.')
            return pCDM(name, x0, y0, z0, ax, ay, az, dip, strike, plunge, utmzone=utmzone, ellps='WGS84', lon0=lon0, lat0=lon0, verbose=True)


    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, x0, y0, z0, ax, ay, az, dip, strike, plunge, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        '''
        Parent class implementing what is common in all pressure objects.

        Args:
            * name          : Name of the pressure source.
            * utmzone       : UTM zone  (optional, default=None)
            * ellps         : ellipsoid (optional, default='WGS84')
        '''

        # Base class init
        super(Pressure,self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)

        # Initialize the pressure source
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing pressure source {}".format(self.name))
        self.verbose = verbose

        self.type = "Pressure"
        self.source = None

        # Allocate fault trace attributes
        self.xf   = None # original non-regularly spaced coordinates (UTM)
        self.yf   = None
        self.lon  = None
        self.lat  = None


        # Allocate depth attributes
        self.depth = None           # Depth of the center of the pressure source

        # Allocate volume/pressure
        self.deltavolume    = None  # If given pressure we can calculate volume and vice versa
        self.ellipshape     = None  # Geometry of pressure source
        self.volume    = None       # Volume of cavity
        self.mu        = 30e9       # Shear modulus
        self.nu        = None       # Poissons ratio

        # Remove files
        self.cleanUp = True

        # Create a dictionnary for the polysol
        self.polysol = {}

        # Create a dictionary for the Green's functions and the data vector
        self.G = {}
        self.d = {}

        # Create structure to store the GFs and the assembled d vector
        self.Gassembled = None
        self.dassembled = None

        # All done
        return


    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Set up whats needed for a null pressure source
    def initializeEmptyPressure(self):
        '''
        Initializes what is required for a pressure source with no volume change

        Returns: None
        '''

        # Initialize
        self.deltavolume = 0
        self.initializepressure()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Returns a copy of the pressure source
    def duplicatePressure(self):
        '''
        Returns a full copy (copy.deepcopy) of the pressure object.

        Return:
            * pressure         : pressure object
        '''

        return copy.deepcopy(self)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Initialize the pressure
    def initializepressure(self, delta="volume", values=None):
        '''
        Re-initializes the volume/pressure change.

        Returns:
           * None
        '''

        # Values
        if values is not None:
            if self.source in {"Mogi","Yang"}:
                self.deltavolume = values
            elif self.source in {"pCDM","CDM"}:
                self.deltapotency = values

        return

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def pressure2xy(self):
        '''
        Transpose the initial pressure source position in lat/lon into the UTM reference.
        UTM coordinates are stored in self.xf and self.yf in km

        Returns:
            * None
        '''

        # do it
        self.xf, self.yf = self.ll2xy(self.lon, self.lat)

        # All done
        return self.xf, self.yf
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def pressure2ll(self):
        '''
        Transpose the initial pressure source position in UTM coordinates into lat/lon.
        Lon/Lat coordinates are stored in self.lon and self.lat in degrees

        Returns:
            * None
        '''

        # do it
        self.lon, self.lat = self.xy2ll(self.xf, self.yf)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def trace(self, x, y, utm=False):
        '''
        Set the initial pressure source position from Lat/Lon or UTM coordinates
        Surface initial pressure source position is stored in self.xf, self.yf (UTM) and
        self.lon, self.lat (Lon/lat)

        Args:
            * Lon           : Array/List containing the Lon points.
            * Lat           : Array/List containing the Lat points.

        Kwargs:
            * utm           : If False, considers x and y are lon/lat
                              If True, considers x and y are utm in km

        Returns:
            * None
        '''

        # Set lon and lat
        if utm:
            self.xf  = np.array(x)/1000.
            self.yf  = np.array(y)/1000.
            # to lat/lon
            self.pressure2ll()
        else:
            self.lon = np.array(x)
            self.lat = np.array(y)
            # utmize
            self.pressure2xy()

        # All done
        return

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def writePressure2File(self, filename, add_volume=None, scale=1.0,
                              stdh5=None, decim=1):
            '''
            Writes the pressure parameters in a file. Trying to make this as generic as possible.
            Args:
                * filename      : Name of the file.
                * add_volume  : Put the volume as a value for the color. Can be None or volume for pCDM.
                * scale         : Multiply the volume change by a factor.

            Returns:
                None
            '''
            # Write something
            if self.verbose:
                print('Writing pressure source to file {}'.format(filename))

            # Open the file
            fout = open(filename, 'w')

            # If an h5 file is specified, open it
            if stdh5 is not None:
                import h5py
                h5fid = h5py.File(stdh5, 'r')
                samples = h5fid['samples'].value[::decim,:]

            # Select the string for the color
            string = '  '
            if add_volume is not None:

                if stdh5 is not None:
                    slp = np.std(samples[:])

                elif add_volume == "volume":
                    if self.source in {"Mogi", "Yang"}:
                        if self.deltapressure is not None:
                            self.pressure2volume()
                            slp = self.deltavolume*scale
                            print("Converting pressure to volume, scaled by", self.deltapressure, self.deltavolume, scale)

                    elif self.source == "pCDM":
                        if None not in {self.DVx, self.DVy, self.DVz}:
                            if self.DVtot is None:
                                self.computeTotalpotency()
                            slp = (self.DVtot)
                            self.ellipshape['ax'] = self.DVx
                            self.ellipshape['ay'] = self.DVy
                            self.ellipshape['az'] = self.DVz

                            print("Total potency scaled by", scale)

                    elif self.source == "CDM":
                        if self.deltaopening is not None:
                            self.opening2potency()
                            slp = self.deltapotency*scale
                            print("Converting from opening to potency, scaled by", scale)

                elif add_volume == "pressure":
                    if self.source in {"Mogi", "Yang"}:
                        if self.deltapressure not in {0.0, None}:
                            slp = self.deltapressure*scale
                    else:
                        raise NameError('Must use flag "volume" (potency) for now for pCDM or CDM.')

                # Make string
                try:
                    string = '-Z{}'.format(slp[0])
                except:
                    string = '-Z0.0'

                # Put the parameter number in the file as well if it exists --what is this???
            parameter = ' '
            if hasattr(self,'index_parameter'):
                i = np.int(self.index_parameter[0])
                j = np.int(self.index_parameter[1])
                k = np.int(self.index_parameter[2])
                parameter = '# {} {} {}'.format(i,j,k)

            # Put the slip value
            try:
                slipstring = ' # {} '.format(slp)
            except:
                slipstring = ' # 0.0 '
            # Write the string to file
            fout.write('> {} {} {}\n'.format(string,parameter,slipstring))

            # Save the shape parameters we created
            fout.write(' Source of type {} [Delta{}] \n # x0 y0 -z0 \n {} {} {} \n # ax ay az  (if pCDM, then DVx, DVy, DVz) \n {} {} {} \n # strike dip plunge \n {} {} {} \n'.format(self.source,add_volume,self.ellipshape['x0'], self.ellipshape['y0'],float(self.ellipshape['z0']),float(self.ellipshape['ax']),float(self.ellipshape['ay']),float(self.ellipshape['az']),float(self.ellipshape['strike']),float(self.ellipshape['dip']),float(self.ellipshape['plunge'])))

            # Close th file
            fout.close()

            # Close h5 file if it is open
            if stdh5 is not None:
                h5fid.close()

            # All done
            return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def readPressureFromFile(self, filename, Cm=None, inputCoordinates='lonlat', donotreadvolume=False):
        '''
        Read the pressure source parameters from a GMT formatted file.
        Args:
            * filename  : Name of the file.

        Returns:
            None
        '''

        # create the lists
        print(filename)
        self.ellipshape = []
        self.Cm   = []


        # open the files
        fin = open(filename, 'r')

        # Assign posterior covariances
        if Cm!=None: # Slip
            self.Cm = np.array(Cm)

        # read all the lines
        A = fin.readlines()


        # Loop over the file
        # Assert it works
        assert A[0].split()[0] == '>', 'Reformat your file...'
        # Get the slip value
        if not donotreadvolume:
            if len(A[0].split())>3:
                deltaVlm = np.array([np.float(A[0].split()[3])])
                print("read from file, volume change is ", deltaVlm)
            else:
                deltaVlm = 0.0
        # get the values
        if inputCoordinates in ('lonlat'):
            lon1, lat1, z1 = A[3].split()
            ax, ay, az = A[5].split()
            strike, dip, plunge = A[7].split()
            # Pass as floating point
            lon1 = float(lon1); lat1 = float(lat1); z1 = float(z1); ax = float(ax); ay = float(ay); az = float(az); dip = float(dip); strike = float(strike); plunge = float(plunge)
            # translate to utm
            x1, y1 = self.ll2xy(lon1, lat1)
            self.lon, self.lat = lon1, lat1
            self.pressure2xy()
        elif inputCoordinates in ('xyz'):
            x1, y1, z1 = A[3].split()
            ax, ay, az = A[5].split()
            strike, dip, plunge = A[7].split()
            # Pass as floating point
            x1 = float(x1); y1 = float(y1); z1 = float(z1); ax = float(ax); ay = float(ay); az = float(az); dip = float(dip); strike = float(strike); plunge = float(plunge)
            # translate to lat and lon
            lon1, lat1 = self.xy2ll(x1, y1)
            self.xf, self.yf = x1, y1
            self.pressure2ll()

        self.ellipshape = {'x0': lon1, 'x0m': x1, 'y0': lat1,'y0m': y1,'z0': z1,'ax': ax,'ay': ay, 'az': az, 'dip': dip,'strike': strike,'plunge': plunge}


        if self.source is None:
            self.source = A[1].split()[3]
        elif self.source != A[1].split()[3]:
            raise ValueError("The object type and the source type are not the same. Reinitialize object with source type {}".format(A[1].split()[3]))

        print("Source of type", self.source)

        if not donotreadvolume:
            self.initializepressure(values=deltaVlm)
        else:
            self.initializepressure()


        # Close the file
        fin.close()

        # depth
        self.depth = [float(z1)]

        # All done
        return


    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def saveGFs(self, dtype='d', outputDir='.',
                      suffix={'pressure':'dP','pressureDVx':'dVx','pressureDVy':'dVy','pressureDVz':'dVz'}):
        '''
        Saves the Green's functions in different files.

        Kwargs:
            * dtype       : Format of the binary data saved
                                'd' for double
                                'f' for float32
            * outputDir   : Directory to save binary data.
            * suffix      : suffix for GFs name (dictionary)

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing Greens functions to file for pressure source {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # Create one file for each slip componenets
            for c in G.keys():
                if G[c] is not None:
                    g = G[c].flatten()
                    n = self.name.replace(' ', '_')
                    d = data.replace(' ', '_')
                    filename = '{}_{}_{}.gf'.format(n, d, suffix[c])
                    g = g.astype(dtype)
                    g.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------

    def setGFsFromFile(self, data, pressure=None, DVx=None, DVy=None, DVz=None,
                                   custom=None, vertical=False, dtype='d'):
        '''
        Sets the Green's functions reading binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already). Basically, it will work better if
        you have computed the GFs using csi...

        Args:
            * data          : Data object

        kwargs:
            * pressure    : File containing the Green's functions for
                              pressure source related displacements.
            * vertical      : Deal with the UP component (gps: default is false,
                              insar: it will be true anyway).
            * dtype         : Type of binary data.
                                    'd' for double/float64
                                    'f' for float32

        Returns:
            * None
        '''

        if self.verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Set up Green's functions for pressure source {}".format(self.name))
            print("and data {} from files: ".format(data.name))
            print("     pressure: {}".format(pressure))




        if self.source == "pCDM":
            # Read the files and reshape the GFs
            Gdvx = None; Gdvy = None; Gdvz = None
            if DVx is not None:
                Gdvx = np.fromfile(DVx, dtype=dtype)
                ndl = int(Gdvx.shape[0])
            if DVy is not None:
                Gdvy = np.fromfile(DVy, dtype=dtype)
                ndl = int(Gdvy.shape[0])
            if DVz is not None:
                Gdvz = np.fromfile(DVz, dtype=dtype)
                ndl = int(Gdvz.shape[0])
            # Create the big dictionary
            G = {'pressureDVx': Gdvx, 'pressureDVy': Gdvy, 'pressureDVz': Gdvz }
        else:
            # Read the files and reshape the GFs
            Gdp = None
            if pressure is not None:
                Gdp = np.fromfile(pressure, dtype=dtype)
                ndl = int(Gdp.shape[0])
            G = {'pressure': Gdp}

        # The dataset sets the Green's functions itself
        data.setGFsInFault(self, G, vertical=vertical)

        # If custom
        if custom is not None:
            self.setCustomGFs(data, custom)

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def saveData(self, dtype='d', outputDir='.'):
        '''
        Saves the Data in binary files.

        Kwargs:
            * dtype       : Format of the binary data saved
                                'd' for double
                                'f' for float32
            * outputDir   : Directory to save binary data

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing Greens functions to file for pressure source {}'.format(self.name))

        # Loop over the data names in self.d
        for data in self.d.keys():

            # Get data
            D = self.d[data]

            # Write data file
            filename = '{}_{}.data'.format(self.name, data)
            D.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildGFs(self, data, vertical=True,
                 method='volume', slipdir=None, verbose=True):
        '''
        Builds the Green's function matrix based on the pressure source.

        The Green's function matrix is stored in a dictionary.
        Each entry of the dictionary is named after the corresponding dataset.
        Each of these entry is a dictionary that contains 'volume' or 'pressure'

        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for
                              the vertical displacements in a gps object.
            * method        : Can be "volume". Converted to pressure for the case of Mogi and Yang before the calculation
            * verbose       : Writes stuff to the screen (overwrites self.verbose)


        Returns:
            * None

        '''


        # Print
        if verbose:
            print('Greens functions computation method: {}'.format(method))

        # Data type check
        if data.dtype == 'insar':
            if not vertical:
                if verbose:
                    print('---------------------------------')
                    print('---------------------------------')
                    print(' WARNING WARNING WARNING WARNING ')
                    print('  You specified vertical=False   ')
                    print(' As this is quite dangerous, we  ')
                    print(' switched it directly to True... ')
                    print(' SAR data are very sensitive to  ')
                    print('     vertical displacements.     ')
                    print(' WARNING WARNING WARNING WARNING ')
                    print('---------------------------------')
                    print('---------------------------------')
                vertical = True

        # Compute the Green's functions
        if method in ('volume'):
            G = self.homogeneousGFs(data, vertical=vertical, verbose=verbose)
        elif method in ('empty'):
            G = self.emptyGFs(data, vertical=vertical, verbose=verbose)
        else:
            assert False, 'Not implemented: method must be volume'

        # Separate the Green's functions for each type of data set
        data.setGFsInFault(self, G, vertical=vertical)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def homogeneousGFs(self, data, volumedir='xyz', vertical=True, donotreadpressure=True, verbose=True):
        '''
        Builds the Green's functions for a homogeneous half-space.

        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for
                              the vertical displacements in a gps object.
            --> Needs to be implemented:
            * volumedir     : For pCDM, want to solve for volume change in which direction?

            * verbose       : Writes stuff to the screen (overwrites self.verbose)
        Returns:
            * G             : Dictionary of the built Green's functions
        '''

        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building pressure source Green's functions for the data set ")
            print("{} of type {} in a homogeneous half-space".format(data.name,
                                                                     data.dtype))
        # Initialize the slip vector
        if donotreadpressure:
            VLM = []
            VLM.append(1.0)

        # Create the dictionary

        if self.source == "pCDM":
            G = {'pressureDVx':[], 'pressureDVy':[], 'pressureDVz':[]}

            # Create the matrices to hold the whole thing
            Gdvx = np.zeros((3, len(data.x)))
            Gdvy = np.zeros((3, len(data.x)))
            Gdvz = np.zeros((3, len(data.x)))

            dvx, dvy, dvz = self.pressure2dis(data, delta="volume", volume=VLM)

            # Store them
            Gdvx[:,:] = dvx.T/self.scale
            Gdvy[:,:] = dvy.T/self.scale
            Gdvz[:,:] = dvz.T/self.scale

            Gdp = [Gdvx, Gdvy, Gdvz]

            print("Using pCDM")
        else:
            G = {'pressure':[]}

            # Create the matrices to hold the whole thing
            Gdp = np.zeros((3, len(data.x), 1))

            dp = self.pressure2dis(data, delta="volume", volume=VLM)
            # Store them
            Gdp[:,:,0] = dp.T/self.mu

        # Build the dictionary
        G = self._buildGFsdict(data, Gdp, vertical=vertical)
        if verbose:
            print(' ')


        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def emptyGFs(self, data, vertical=True, slipdir='sd', verbose=True):
        ''' 
        Build zero GFs.

        Args:
            * data          : Data object (gps, insar, optical, ...)

        Kwargs:
            * vertical      : If True, will produce green's functions for the vertical displacements in a gps object.
            * slipdir       : Direction of slip along the patches. Can be any combination of s (strikeslip), d (dipslip), t (tensile) and c (coupling)
            * verbose       : Writes stuff to the screen (overwrites self.verbose)

        Returns:
            * G             : Dictionnary of GFs
        '''

        if self.source == "pCDM":

            raise NotImplementedError

        else:

            # Create the matrices to hold the whole thing
            if data.dtype in ('insar', 'surfaceslip'):
                x = len(data.vel)
            elif data.dtype in ('opticor'):
                x = data.vel.shape[0] * 2
            elif data.dtype in ('gps'):
                if vertical:
                    x = data.vel_enu.shape[0]*3
                else:
                    x = data.vel_enu.shape[0]*2
        
            G = {'pressure': np.zeros((x,1))}

        # All done
        return G
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setGFs(self, data, deltapressure=[None, None, None], GDVx=[None, None, None],GDVy=[None, None, None],GDVz=[None, None, None],
                           vertical=False, synthetic=False):
        '''
        Stores the input Green's functions matrices into the pressure source structure.

        These GFs are organized in a dictionary structure in self.G
        Entries of self.G are the data set names (data.name).
            Entries of self.G[data.name] are 'deltapressure'.

        If you provide GPS GFs, those are organised with E, N and U in lines

        If you provide Optical GFs, those are organised with E and N in lines

        If you provide InSAR GFs, these need to be projected onto the
        LOS direction already.

        Args:
            * data          : Data structure

        Kwargs:
            * deltapressure    : List of matrices of the pressure source
                              Green's functions

        Returns:
            * None
        '''

        # Get the number of data per point
        if data.dtype in ('insar', 'tsunami', 'surfaceslip'):
            data.obs_per_station = 1
        elif data.dtype in ('gps', 'multigps'):
            data.obs_per_station = 0
            # Check components
            if not np.isnan(data.vel_enu[:,0]).any():
                data.obs_per_station += 1
            if not np.isnan(data.vel_enu[:,1]).any():
                data.obs_per_station += 1
            if vertical:
                if np.isnan(data.vel_enu[:,2]).any():
                    raise ValueError('Vertical can only be true if all stations have vertical components')
                data.obs_per_station += 1
        elif data.dtype == 'opticorr':
            data.obs_per_station = 2
            if vertical:
                data.obs_per_station += 1

        # Create the storage for that dataset
        if data.name not in self.G.keys():
            self.G[data.name] = {}
        G = self.G[data.name]

        # Initializes the data vector
        if not synthetic:
            if data.dtype in ('insar', 'surfaceslip'):
                self.d[data.name] = data.vel
                vertical = True # Always true for InSAR
            elif data.dtype == 'tsunami':
                self.d[data.name] = data.d
                vertical = True
            elif data.dtype in ('gps', 'multigps'):
                if vertical:
                    self.d[data.name] = data.vel_enu.T.flatten()
                else:
                    self.d[data.name] = data.vel_enu[:,0:2].T.flatten()
                self.d[data.name]=self.d[data.name][np.isfinite(self.d[data.name])]
            elif data.dtype == 'opticorr':
                self.d[data.name] = np.hstack((data.east.T.flatten(),
                                               data.north.T.flatten()))
                if vertical:
                    self.d[data.name] = np.hstack((self.d[data.name],
                                                   np.zeros_like(data.east.T.ravel())))

        # Pressure
        if len(deltapressure) == 3:            # GPS case

            E_dp = deltapressure[0]
            N_dp = deltapressure[1]
            U_dp = deltapressure[2]
            dp = []
            nd = 0
            if (E_dp is not None) and (N_dp is not None):
                d = E_dp.shape[0]
                m = E_dp.shape[1]
                dp.append(E_dp)
                dp.append(N_dp)
                nd += 2
            if (U_dp is not None):
                d = U_dp.shape[0]
                m = U_dp.shape[1]
                dp.append(U_dp)
                nd += 1
            if nd > 0:
                dp = np.array(dp)
                dp = dp.reshape((nd*d, m))
                G['pressure'] = dp

        elif len(deltapressure) == 1:          # InSAR/Tsunami case
            Green_dp = deltapressure[0]
            if Green_dp is not None:
                G['pressure'] = Green_dp

        # pCDM case DVx
        if len(GDVx) == 3:            # GPS case

            E_dp = deltapressure[0]
            N_dp = deltapressure[1]
            U_dp = deltapressure[2]
            dp = []
            nd = 0
            if (E_dp is not None) and (N_dp is not None):
                d = E_dp.shape[0]
                m = E_dp.shape[1]
                dp.append(E_dp)
                dp.append(N_dp)
                nd += 2
            if (U_dp is not None):
                d = U_dp.shape[0]
                m = U_dp.shape[1]
                dp.append(U_dp)
                nd += 1
            if nd > 0:
                dp = np.array(dp)
                dp = dp.reshape((nd*d, m))
                G['pressureDVx'] = dp

        elif len(GDVx) == 1:          # InSAR/Tsunami case
            Green_dvx = GDVx[0]
            if Green_dvx is not None:
                G['pressureDVx'] = Green_dvx

        # pCDM case DVy
        if len(GDVy) == 3:            # GPS case

            E_dp = deltapressure[0]
            N_dp = deltapressure[1]
            U_dp = deltapressure[2]
            dp = []
            nd = 0
            if (E_dp is not None) and (N_dp is not None):
                d = E_dp.shape[0]
                m = E_dp.shape[1]
                dp.append(E_dp)
                dp.append(N_dp)
                nd += 2
            if (U_dp is not None):
                d = U_dp.shape[0]
                m = U_dp.shape[1]
                dp.append(U_dp)
                nd += 1
            if nd > 0:
                dp = np.array(dp)
                dp = dp.reshape((nd*d, m))
                G['pressureDVy'] = dp

        elif len(GDVy) == 1:          # InSAR/Tsunami case
            Green_dvy = GDVy[0]
            if Green_dvy is not None:
                G['pressureDVy'] = Green_dvy

        # pCDM case DVz
        if len(GDVz) == 3:            # GPS case

            E_dp = deltapressure[0]
            N_dp = deltapressure[1]
            U_dp = deltapressure[2]
            dp = []
            nd = 0
            if (E_dp is not None) and (N_dp is not None):
                d = E_dp.shape[0]
                m = E_dp.shape[1]
                dp.append(E_dp)
                dp.append(N_dp)
                nd += 2
            if (U_dp is not None):
                d = U_dp.shape[0]
                m = U_dp.shape[1]
                dp.append(U_dp)
                nd += 1
            if nd > 0:
                dp = np.array(dp)
                dp = dp.reshape((nd*d, m))
                G['pressureDVz'] = dp

        elif len(GDVz) == 1:          # InSAR/Tsunami case
            Green_dvz = GDVz[0]
            if Green_dvz is not None:
                G['pressureDVz'] = Green_dvz

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    #

    def assembled(self, datas, verbose=True):
        '''
        Assembles a data vector for inversion using the list datas
        Assembled vector is stored in self.dassembled

        Args:
            * datas         : list of data objects

        Returns:
            * None
        '''

        # Check
        if type(datas) is not list:
            datas = [datas]

        if verbose:
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
                if verbose:
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
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleGFs(self, datas, polys=None, verbose=True,
                                 custom=False, computeNormFact=True):
        '''
        Assemble the Green's functions corresponding to the data in datas.
        This method allows to specify which transformation is going
        to be estimated in the data sets, through the polys argument.

        Assembled Green's function matrix is stored in self.Gassembled

        Args:
            * datas             : list of data sets. If only one data set is
                                  used, can be a data instance only.

        Kwargs:
            * polys             : None -> nothing additional is estimated

                 For InSAR, Optical, GPS:
                       1 -> estimate a constant offset
                       3 -> estimate z = ax + by + c
                       4 -> estimate z = axy + bx + cy + d

                 For GPS only:
                       'full'                -> Estimates a rotation,
                                                translation and scaling
                                                (Helmert transform).
                       'strain'              -> Estimates the full strain
                                                tensor (Rotation + Translation
                                                + Internal strain)
                       'strainnorotation'    -> Estimates the strain tensor and a
                                                translation
                       'strainonly'          -> Estimates the strain tensor
                       'strainnotranslation' -> Estimates the strain tensor and a
                                                rotation
                       'translation'         -> Estimates the translation
                       'translationrotation  -> Estimates the translation and a
                                                rotation

            * custom            : If True, gets the additional Green's function
                                  from the dictionary self.G[data.name]['custom']

            * computeNormFact   : bool
                if True, compute new OrbNormalizingFactor
                if False, uses parameters in self.OrbNormalizingFactor

            * verbose           : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check
        if type(datas) is not list:
            datas = [datas]

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print("Assembling G for pressure source {}".format(self.name))

        # Create a dictionary to keep track of the orbital froms
        self.poly = {}

        # Set poly right
        if polys.__class__ is not list:
            for data in datas:
                if (polys.__class__ is not str) and (polys is not None):
                    self.poly[data.name] = polys*data.obs_per_station
                else:
                    self.poly[data.name] = polys
        elif polys.__class__ is list:
            for data, poly in zip(datas, polys):
                print(poly.__class__,data.name)
                print((poly.__class__ is not str))
                if (poly.__class__ is not str) and (poly is not None) and (poly.__class__ is not list):
                    self.poly[data.name] = poly*data.obs_per_station
                    print(data.obs_per_station)
                else:
                    self.poly[data.name] = poly

        # Create the transformation holder
        if not hasattr(self, 'helmert'):
            self.helmert = {}
        if not hasattr(self, 'strain'):
            self.strain = {}
        if not hasattr(self, 'transformation'):
            self.transformation = {}

        Npo = 0
        #Solve for just the deltaPressure parameter
        if self.source == "pCDM":
            Nps = 3
        else:
            Nps = 1
        for data in datas :
            transformation = self.poly[data.name]
            if type(transformation) in (str, list):
                tmpNpo = data.getNumberOfTransformParameters(self.poly[data.name])
                Npo += tmpNpo
                if type(transformation) is str:
                    if transformation in ('full'):
                        self.helmert[data.name] = tmpNpo
                    elif transformation in ('strain', 'strainonly',
                                            'strainnorotation', 'strainnotranslation',
                                            'translation', 'translationrotation'):
                        self.strain[data.name] = tmpNpo
                else:
                    self.transformation[data.name] = tmpNpo
            elif transformation is not None:
                Npo += transformation
        Np = Nps + Npo

        # Save extra Parameters
        self.TransformationParameters = Npo

        # Custom?
        if custom:
            Npc = 0
            for data in datas:
                if 'custom' in self.G[data.name].keys():
                    Npc += self.G[data.name]['custom'].shape[1]
            Np += Npc
            self.NumberCustom = Npc
        else:
            Npc = 0

        # Get the number of data
        Nd = 0
        for data in datas:
            Nd += self.d[data.name].shape[0]

        # Allocate G and d
        G = np.zeros((Nd, Np))

        # Create the list of data names, to keep track of it
        self.datanames = []

        # loop over the datasets
        el = 0
        custstart = Nps # custom indices
        polstart = Nps + Npc # poly indices
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            if verbose:
                print("Dealing with {} of type {}".format(data.name, data.dtype))

            # Elastic Green's functions

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
            Glocal = np.zeros((Ndlocal, Nps))

            # Fill Glocal --- difference between Glocal and big G?
            ec = 0

            if self.source == "pCDM":
                Glocal[:,0] = self.G[data.name]['pressureDVx'].squeeze() #???
                Glocal[:,1] = self.G[data.name]['pressureDVy'].squeeze() #???
                Glocal[:,2] = self.G[data.name]['pressureDVz'].squeeze() #???
            else:
                print(Glocal.shape, self.G[data.name]['pressure'].shape)
                Glocal[:,0] = self.G[data.name]['pressure'].squeeze() #???

            #ec += Nclocal

            # Put Glocal into the big G
            G[el:el+Ndlocal,0:Nps] = Glocal
            # Custom
            if custom:
                # Check if data has custom GFs
                if 'custom' in self.G[data.name].keys():
                    nc = self.G[data.name]['custom'].shape[1] # Nb of custom param
                    custend = custstart + nc
                    G[el:el+Ndlocal,custstart:custend] = self.G[data.name]['custom']
                    custstart += nc

            # Polynomes and strain
            if self.poly[data.name] is not None:

                # Build the polynomial function
                if data.dtype in ('gps', 'multigps'):
                    orb = data.getTransformEstimator(self.poly[data.name])
                elif data.dtype in ('insar', 'opticorr'):
                    orb = data.getPolyEstimator(self.poly[data.name],computeNormFact=computeNormFact)
                elif data.dtype == 'tsunami':
                    orb = data.getRampEstimator(self.poly[data.name])

                # Number of columns
                nc = orb.shape[1]

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
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def assembleCd(self, datas, add_prediction=None, verbose=False):
        '''
        Assembles the data covariance matrices that have been built for each
        data structure.

        Args:
            * datas         : List of data instances or one data instance

        Kwargs:
            * add_prediction: Precentage of displacement to add to the Cd
                              diagonal to simulate a Cp (dirty version of
                              a prediction error covariance, see Duputel et
                              al 2013, GJI).
            * verbose       : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''

        # Check if the Green's function are ready
        assert self.Gassembled is not None, \
                "You should assemble the Green's function matrix first"

        # Check
        if type(datas) is not list:
            datas = [datas]

        # Get the total number of data
        Nd = self.Gassembled.shape[0]
        Cd = np.zeros((Nd, Nd))

        # Loop over the data sets
        st = 0
        for data in datas:
            # Fill in Cd
            if verbose:
                print("{0:s}: data vector shape {1:s}".format(data.name, self.d[data.name].shape))
            se = st + self.d[data.name].shape[0]
            Cd[st:se, st:se] = data.Cd
            # Add some Cp if asked
            if add_prediction is not None:
                Cd[st:se, st:se] += np.diag((self.d[data.name]*add_prediction/100.)**2)
            st += self.d[data.name].shape[0]

        # Store Cd in self
        self.Cd = Cd

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildCm(self, sigma, extra_params=None, user_Cm=None, verbose=True):
        '''
        Builds a dummy model covariance matrix using user-defined value.

        Model covariance is stored in self.Cm.

        Kwargs:
            * extra_params  : a list of extra parameters.
            * user_Cm       : user-defined value for the covariance matrix
            * verbose       : Talk to me (overwrites self.verrbose)

        Returns:
            * None
        '''

        # print
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling the Cm matrix ")


        # Creates the principal Cm matrix
        if self.source == "pCDM":
            Nps = 3
        else:
            Nps = 1
        if extra_params is not None:
            Npe = len(extra_params)
        else:
            Npe = 0

        # Check
        Cm = np.eye(Nps+Npe, Nps+Npe)
        Cm[:Nps,:Nps] *= sigma

        # Put the extra values
        st = 0
        if user_Cm is not None:
            for i in range(Nps):
                Cm[st, st] = user_Cm
                st += 1
        if extra_params is not None:
            for i in range(len(extra_params)):
                Cm[st+i, st+i] = extra_params[i]

        # Store Cm into self
        self.Cm = Cm

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def _buildGFsdict(self, data, Gdp, vertical=True):
        '''
        Some ordering of the Gfs to make the computation routines simpler.

        Args:
            * data          : instance of data
            * Gdp           : Pressure greens functions

        Kwargs:
            *vertical       : If true, assumes verticals are used for the GPS case

        Returns:
            * G             : Dictionary of GFs
        '''

        # Verticals?
        Ncomp = 3
        if not vertical:
            Ncomp = 2
            Gdp = Gdp[:2,:,:]
        
        # Size 
        Nparm = Gdp.shape[2]
        Npoints = Gdp.shape[1]

        # Get some size info
        if self.source == "pCDM":
            Npoints = Gdp[0].shape[1]
        else:
            Npoints = Gdp.shape[1]
        Ndata = Ncomp*Npoints

        # Check format
        if data.dtype in ['gps', 'opticorr', 'multigps']:
            if self.source == "pCDM":
                Gdvx = Gdp[0].reshape((Ndata, Nparm))
                Gdvy = Gdp[1].reshape((Ndata, Nparm))
                Gdvz = Gdp[2].reshape((Ndata, Nparm))
            else:
                # Flat arrays with e, then n, then u (optional)
                Gdp = Gdp.reshape((Ndata, Nparm))
        elif data.dtype in ('insar', 'insartimeseries'):
            if self.source == "pCDM":
                # If InSAR, do the dot product with the los
                Gdvx_los = []
                Gdvy_los = []
                Gdvz_los = []
                for i in range(Npoints):
                        Gdvx_los.append(np.dot(data.los[i,:], Gdp[0][:,i]))
                        Gdvy_los.append(np.dot(data.los[i,:], Gdp[1][:,i]))
                        Gdvz_los.append(np.dot(data.los[i,:], Gdp[2][:,i]))

                Gdvx, Gdvy, Gdvz = [np.array(greens).reshape(Npoints) for greens in [Gdvx_los,Gdvy_los,Gdvz_los]]
            else:
                # If InSAR, do the dot product with the los
                Gdp_los = []
                for i in range(Npoints):
                        Gdp_los.append(np.dot(data.los[i,:], Gdp[:,i]))
                Gdp = np.array(Gdp_los).reshape((Npoints))


        # Create the dictionary
        if self.source == "pCDM":
            G = {'pressureDVx':[], 'pressureDVy':[], 'pressureDVz':[]}
            # Reshape the Green's functions
            G['pressureDVx'] = Gdvx
            G['pressureDVy'] = Gdvy
            G['pressureDVz'] = Gdvz
        else:
            G = {'pressure':[]}
            # Reshape the Green's functions
            G['pressure'] = Gdp

        # All done
        return G

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------


    def surfacesimulation(self, box=None, disk=None, err=None, lonlat=None,
                          volume=None):
        '''
        Takes the slip vector and computes the surface displacement that corresponds on a regular grid.

        Args:
            * box       : Can be a list of [minlon, maxlon, minlat, maxlat, n].

        Kwargs:
            * disk      : list of [xcenter, ycenter, radius, n]
            * err       :
            * lonlat    : Arrays of lat and lon. [lon, lat]
            * volume   : Provide values of volume from which to calculate displacement
        '''

        # create a fake gps object
        self.sim = gpsclass('simulation', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0)

        # Create a lon lat grid
        if lonlat is None:
            if (box is None) and (disk is None) :
                n = box[-1]
                lon = np.linspace(self.lon.min(), self.lon.max(), n)
                lat = np.linspace(self.lat.min(), self.lat.max(), n)
                lon, lat = np.meshgrid(lon,lat)
                lon = lon.flatten()
                lat = lat.flatten()
            elif (box is not None):
                n = box[-1]
                lon = np.linspace(box[0], box[1], n)
                lat = np.linspace(box[2], box[3], n)
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
        else:
            lon = np.array(lonlat[0])
            lat = np.array(lonlat[1])

        # Clean it
        if (lon.max()>360.) or (lon.min()<-180.0) or (lat.max()>90.) or (lat.min()<-90):
            self.sim.x = lon
            self.sim.y = lat
        else:
            self.sim.lon = lon
            self.sim.lat = lat
            # put these in x y utm coordinates
            self.sim.lonlat2xy()

        # Initialize the vel_enu array
        self.sim.vel_enu = np.zeros((lon.size, 3))

        # Create the station name array
        self.sim.station = []
        for i in range(len(self.sim.x)):
            name = '{:04d}'.format(i)
            self.sim.station.append(name)
        self.sim.station = np.array(self.sim.station)

        # Create an error array
        self.sim.err_enu = np.zeros(self.sim.vel_enu.shape)
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

        # Load the pressure values is provided
        if volume is not None:
            self.deltavolume = volume

        sys.stdout.write('\r Calculating Greens functions for source of type {}'.format(self.source))
        sys.stdout.flush()
        # Get the surface displacement due to the slip on this patch
        if self.source == "pCDM":
            u1,u2,u3 = self.pressure2dis(self.sim)
            self.sim.vel_enu += u1
            self.sim.vel_enu += u2
            self.sim.vel_enu += u3
        else:
            u = self.pressure2dis(self.sim)
            # Sum these to get the synthetics
            self.sim.vel_enu += u

        sys.stdout.write('\n')
        sys.stdout.flush()

        # All done
        return
