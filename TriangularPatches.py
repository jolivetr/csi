'''
A parent class that deals with triangular patches fault

Written by Bryan Riel, Z. Duputel and R. Jolivet November 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from . import triangularDisp as tdisp
from scipy.linalg import block_diag
import copy
import sys
import os

# Personals
from .Fault import Fault
from .geodeticplot import geodeticplot as geoplot
from .gps import gps as gpsclass

class TriangularPatches(Fault):
    '''
    Classes implementing a fault made of triangular patches. Inherits from Fault

    Args:
        * name      : Name of the fault.

    Kwargs:
        * utmzone   : UTM zone  (optional, default=None)
        * lon0      : Longitude of the center of the UTM zone
        * lat0      : Latitude of the center of the UTM zone
        * ellps     : ellipsoid (optional, default='WGS84')
        * verbose   : Speak to me (default=True)
    '''

    # ----------------------------------------------------------------------
    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True, lon0=None, lat0=None):

        # Base class init
        super(TriangularPatches,self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=verbose)

        # Specify the type of patch
        self.patchType = 'triangle'

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setdepth(self):
        '''
        Set depth patch attributes

        Returns:
            * None
        '''

        # Set depth
        self.depth = np.max([v[2] for v in self.Vertices])
        self.z_patches = np.linspace(self.depth, 0.0, 5)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def computeArea(self):
        '''
        Computes the area of all triangles.

        Returns:
            * None
        '''

        # Area
        self.area = []

        # Loop over patches
        for patch in self.patch:
            self.area.append(self.patchArea(patch))

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def patchArea(self, patch):
        '''
        Returns the area of one patch.

        Args:   
            * patch : one item of the patch list.

        Returns:
            * Area  : float
        '''

        # Get vertices of patch
        p1, p2, p3 = list(patch)

        # Compute side lengths
        a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
        b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)
        c = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)

        # Compute area using numerically stable Heron's formula
        c,b,a = np.sort([a, b, c])
        area = 0.25 * np.sqrt((a + (b + c)) * (c - (a - b))
                       * (c + (a - b)) * (a + (b - c)))

        # All Done
        return area
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def splitPatch(self, patch):
        '''
        Splits a patch into 4 patches, based on the mid-point of each side.

        Args:
            * patch : item of the patch list.

        Returns:
            * t1, t2, t3, t4    : 4 patches
        '''

        # Get corners
        p1, p2, p3 = list(patch)
        if type(p1) is not list:
            p1 = p1.tolist()
        if type(p2) is not list:
            p2 = p2.tolist()
        if type(p3) is not list:
            p3 = p3.tolist()

        # Compute mid-points
        p12 = [p1[0] + (p2[0]-p1[0])/2., 
               p1[1] + (p2[1]-p1[1])/2.,
               p1[2] + (p2[2]-p1[2])/2.]
        p23 = [p2[0] + (p3[0]-p2[0])/2.,
               p2[1] + (p3[1]-p2[1])/2.,
               p2[2] + (p3[2]-p2[2])/2.]
        p31 = [p3[0] + (p1[0]-p3[0])/2.,
               p3[1] + (p1[1]-p3[1])/2.,
               p3[2] + (p1[2]-p3[2])/2.]

        # make 4 triangles
        t1 = [p1, p12, p31]
        t2 = [p12, p2, p23]
        t3 = [p31, p23, p3]
        t4 = [p31, p12, p23]

        # All done
        return t1, t2, t3, t4
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def selectPatches(self,minlon,maxlon,minlat,maxlat,mindep,maxdep):
        '''
        Removes patches that are outside of a 3D box.

        Args:   
            * minlon        : west longitude
            * maxlon        : east longitude
            * minlat        : south latitude
            * maxlat        : north latitude
            * mindep        : Minimum depth
            * maxdep        : Maximum depth

        Returns:
            * None
        '''

        xmin,ymin = self.ll2xy(minlon,minlat)
        xmax,ymax = self.ll2xy(maxlon,maxlat)

        for p in range(len(self.patch)-1,-1,-1):
            x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(p)
            if x1<xmin or x1>xmax or x2<ymin or x2>ymax or x3<mindep or x3>maxdep:
                self.deletepatch(p)

        for i in range(len(self.xf)-1,-1,-1):
            x1 = self.xf[i]
            x2 = self.yf[i]
            if x1<xmin or x1>xmax or x2<ymin or x2>ymax:
                self.xf = np.delete(self.xf,i)
                self.yf = np.delete(self.yf,i)

        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def vertices2ll(self):
        '''
        Converts all the vertices into lonlat coordinates.

        Returns:    
            * None
        '''

        # Create a list
        vertices_ll = []

        # iterate
        for vertex in self.Vertices:
            lon, lat = self.xy2ll(vertex[0], vertex[1])
            vertices_ll.append([lon, lat, vertex[2]])

        # Save
        self.Vertices_ll = np.array(vertices_ll)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setVerticesFromPatches(self):
        '''
        Takes the patches and constructs a list of Vertices and Faces

        Returns:
            * None
        '''

        # Get patches
        patches = self.patch

        # Create lists
        vertices = []
        faces = []

        # Iterate to build vertices
        for patch in patches:
            for vertex in patch.tolist():
                if vertex not in vertices:
                    vertices.append(vertex)

        # Iterate to build Faces
        for patch in patches:
            face = []
            for vertex in patch.tolist():
                uu = np.flatnonzero(np.array([vertex==v for v in vertices]))[0]
                face.append(uu)
            faces.append(face)

        # Set them
        self.Vertices = np.array(vertices)
        self.Faces = np.array(faces)

        # 2 lon lat
        self.vertices2ll()

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    def patches2triangles(self, fault, numberOfTriangles=4):
        '''
        Takes a fault with rectangular patches and splits them into triangles to 
        initialize self.

        Args:
            * fault             : instance of rectangular patches.

        Kwargs:
            * numberOfTriangles : Split each patch in 2 or 4 (default) triangle

        Returns:
            * None
        '''

        # Initialize the lists of patches
        self.patch = []
        self.patchll = []

        # Initialize vertices and faces
        vertices = []
        faces = []

        # Each patch is being splitted in 2 or 4 triangles
        for patch in fault.patch:

            # Add vertices
            for vertex in patch.tolist():
                if vertex not in vertices:
                    vertices.append(vertex)

            # Find the vertices in the list
            i0 = np.flatnonzero(np.array([patch[0].tolist()==v for v in vertices]))[0]
            i1 = np.flatnonzero(np.array([patch[1].tolist()==v for v in vertices]))[0]
            i2 = np.flatnonzero(np.array([patch[2].tolist()==v for v in vertices]))[0]
            i3 = np.flatnonzero(np.array([patch[3].tolist()==v for v in vertices]))[0]

            if numberOfTriangles==4:
                
                # Get the center
                center = np.array(fault.getpatchgeometry(patch, center=True)[:3])
                vertices.append(list(center))
                ic = np.flatnonzero(np.array([center.tolist()==v for v in vertices]))[0]

                # Split in 4
                t1 = np.array([patch[0], patch[1], center])
                t2 = np.array([patch[1], patch[2], center])
                t3 = np.array([patch[2], patch[3], center])
                t4 = np.array([patch[3], patch[0], center])

                # faces
                fs = ([i0, i1, ic],
                      [i1, i2, ic],
                      [i2, i3, ic],
                      [i3, i0, ic])
        
                # patches
                ps = [t1, t2, t3, t4]

            elif numberOfTriangles==2:
        
                # Split in 2
                t1 = np.array([patch[0], patch[1], patch[2]])
                t2 = np.array([patch[2], patch[3], patch[0]])

                # faces
                fs = ([i0, i1, i2], [i2, i3, i0])

                # patches
                ps = (t1, t2)
            
            else:
                assert False, 'numberOfTriangles should be 2 or 4'

            for f,p in zip(fs, ps):
                faces.append(f)
                self.patch.append(p)

        # Save
        self.Vertices = np.array(vertices)
        self.Faces = np.array(faces)

        # Convert
        self.vertices2ll()
        self.patch2ll()

        # Initialize slip
        self.initializeslip()

        # Set fault trace
        self.xf = fault.xf
        self.yf = fault.yf
        self.lon = fault.lon
        self.lat = fault.lat
        if hasattr(fault, 'xi'):
            self.xi = fault.xi
            self.yi = fault.yi
            self.loni = fault.loni
            self.lati = fault.lati

        # Set depth
        self.setdepth()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def readPatchesFromFile(self, filename, readpatchindex=True, 
                            donotreadslip=False, gmtslip=True,
                            inputCoordinates='lonlat'):
        '''
        Reads patches from a GMT formatted file.

        Args:
            * filename          : Name of the file

        Kwargs:
            * inputCoordinates  : Default is 'lonlat'. Can be 'utm'
            * readpatchindex    : Default True.
            * donotreadslip     : Default is False. If True, does not read the slip
            * gmtslip           : A -Zxxx is in the header of each patch
            * inputCoordinates  : Default is 'lonlat', can be 'xyz'

        Returns:
            * None
        '''

        # create the lists
        self.patch = []
        self.patchll = []
        if readpatchindex:
            self.index_parameter = []
        if not donotreadslip:
            Slip = []

        # open the files
        fin = open(filename, 'r') 
        
        # read all the lines
        A = fin.readlines()

        # depth
        D = 0.0
        d = 10000.

        # Index
        if gmtslip:
            ipatch = 3
        else:
            ipatch = 2

        # Loop over the file
        i = 0
        while i<len(A):
            
            # Assert it works
            assert A[i].split()[0] is '>', 'Not a patch, reformat your file...'
            # Get the Patch Id
            if readpatchindex:
                self.index_parameter.append([np.int(A[i].split()[ipatch]),np.int(A[i].split()[ipatch+1]),np.int(A[i].split()[ipatch+2])])
            # Get the slip value
            if not donotreadslip:
                if len(A[i].split())>7:
                    slip = np.array([np.float(A[i].split()[ipatch+4]), np.float(A[i].split()[ipatch+5]), np.float(A[i].split()[ipatch+6])])
                else:
                    slip = np.array([0.0, 0.0, 0.0])
                Slip.append(slip)
            # get the values
            if inputCoordinates in ('lonlat'):
                lon1, lat1, z1 = A[i+1].split()
                lon2, lat2, z2 = A[i+2].split()
                lon3, lat3, z3 = A[i+3].split()
                # Pass as floating point
                lon1 = float(lon1); lat1 = float(lat1); z1 = float(z1)
                lon2 = float(lon2); lat2 = float(lat2); z2 = float(z2)
                lon3 = float(lon3); lat3 = float(lat3); z3 = float(z3)
                # translate to utm
                x1, y1 = self.ll2xy(lon1, lat1)
                x2, y2 = self.ll2xy(lon2, lat2)
                x3, y3 = self.ll2xy(lon3, lat3)
            elif inputCoordinates in ('xyz'):
                x1, y1, z1 = A[i+1].split()
                x2, y2, z2 = A[i+2].split()
                x3, y3, z3 = A[i+3].split()
                # Pass as floating point
                x1 = float(x1); y1 = float(y1); z1 = float(z1)
                x2 = float(x2); y2 = float(y2); z2 = float(z2)
                x3 = float(x3); y3 = float(y3); z3 = float(z3)
                # translate to utm
                lon1, lat1 = self.xy2ll(x1, y1)
                lon2, lat2 = self.xy2ll(x2, y2)
                lon3, lat3 = self.xy2ll(x3, y3)
            # Depth
            mm = min([float(z1), float(z2), float(z3)])
            if D<mm:
                D=mm
            if d>mm:
                d=mm
            # Set points
            p1 = [x1, y1, z1]; p1ll = [lon1, lat1, z1]
            p2 = [x2, y2, z2]; p2ll = [lon2, lat2, z2]
            p3 = [x3, y3, z3]; p3ll = [lon3, lat3, z3]
            # Store these
            p = [p1, p2, p3]
            pll = [p1ll, p2ll, p3ll]
            p = np.array(p)
            pll = np.array(pll)
            # Store these in the lists
            self.patch.append(p)
            self.patchll.append(pll)
            # increase i
            i += 4

        # Close the file
        fin.close()

        # depth
        self.depth = D
        self.top = d
        self.z_patches = np.linspace(0,D,5)
        self.factor_depth = 1.

        # Patches 2 vertices
        self.setVerticesFromPatches()
        self.numpatch = self.Faces.shape[0]

        # Translate slip to np.array
        if not donotreadslip:
            self.initializeslip(values=np.array(Slip))
        else:
            self.initializeslip()
        if readpatchindex:
            self.index_parameter = np.array(self.index_parameter)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def readGocadPatches(self, filename, neg_depth=False, utm=False, factor_xy=1.0,
                         factor_depth=1.0, verbose=False):
        '''
        Load a triangulated surface from a Gocad formatted file. Vertices 
        must be in geographical coordinates.

        Args:
            * filename:  tsurf file to read

        Kwargs:
            * neg_depth: if true, use negative depth
            * utm: if true, input file is given as utm coordinates (if false -> lon/lat)
            * factor_xy: if utm==True, multiplication factor for x and y
            * factor_depth: multiplication factor for z
            * verbose: Speak to me

        Returns:
            * None
        '''

        # Initialize the lists of patches
        self.patch   = []
        self.patchll = []

        # Factor to correct input negative depths (we want depths to be positive)
        if neg_depth:
            negFactor = -1.0
        else:
            negFactor =  1.0

        # Get the geographic vertices and connectivities from the Gocad file
        with open(filename, 'r') as fid:
            vertices = []
            vids     = []
            faces    = []
            for line in fid:
                if line.startswith('VRTX'):
                    items = line.split()
                    name, vid, x, y, z = items[:5]
                    vids.append(vid)
                    vertices.append([float(x), float(y), negFactor*float(z)])
                elif line.startswith('TRGL'):
                    name, p1, p2, p3 = line.split()
                    faces.append([int(p1), int(p2), int(p3)])
            fid.close()
            vids = np.array(vids,dtype=int)
            i0 = np.min(vids)
            vids = vids - i0
            i    = np.argsort(vids)
            vertices = np.array(vertices, dtype=float)[i,:]
            faces = np.array(faces, dtype=int) - i0

        # Resample vertices to UTM
        if utm:
            vx = vertices[:,0].copy()*factor_xy
            vy = vertices[:,1].copy()*factor_xy
            vertices[:,0],vertices[:,1] = self.xy2ll(vx,vy)
        else:
            vx, vy = self.ll2xy(vertices[:,0], vertices[:,1])
        vz = vertices[:,2]*factor_depth
        self.factor_depth = factor_depth
        self.Vertices = np.column_stack((vx, vy, vz))
        self.Vertices_ll = vertices
        self.Faces = faces
        if verbose:
            print('min/max depth: {} km/ {} km'.format(vz.min(),vz.max()))
            print('min/max lat: {} deg/ {} deg'.format(vertices[:,1].min(),vertices[:,1].max()))
            print('min/max lon: {} deg/ {} deg'.format(vertices[:,0].min(),vertices[:,0].max()))
            print('min/max x: {} km/ {} km'.format(vx.min(),vx.max()))
            print('min/max y: {} km/ {} km'.format(vy.min(),vy.max()))

        # Loop over faces and create a triangular patch consisting of coordinate tuples
        self.numpatch = faces.shape[0]
        for i in range(self.numpatch):
            # Get the indices of the vertices
            v1, v2, v3 = faces[i,:]
            # Get the coordinates
            x1, y1, lon1, lat1, z1 = vx[v1], vy[v1], vertices[v1,0], vertices[v1,1], vz[v1]
            x2, y2, lon2, lat2, z2 = vx[v2], vy[v2], vertices[v2,0], vertices[v2,1], vz[v2]
            x3, y3, lon3, lat3, z3 = vx[v3], vy[v3], vertices[v3,0], vertices[v3,1], vz[v3]
            # Make the coordinate tuples
            p1 = [x1, y1, z1]; pll1 = [lon1, lat1, z1]
            p2 = [x2, y2, z2]; pll2 = [lon2, lat2, z2]
            p3 = [x3, y3, z3]; pll3 = [lon3, lat3, z3]
            # Store the patch
            self.patch.append(np.array([p1, p2, p3]))
            self.patchll.append(np.array([pll1, pll2, pll3]))

        # Update the depth of the bottom of the fault
        if neg_depth:
            self.top   = np.max(vz)
            self.depth = np.min(vz)
        else:
            self.top   = np.min(vz)
            self.depth = np.max(vz)
        self.z_patches = np.linspace(self.depth, 0.0, 5)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writeGocadPatches(self, filename, utm=False):
        '''
        Write a triangulated Gocad surface file.

        Args:
            * filename  : output file name

        Kwargs:
            * utm       : Write in utm coordinates if True

        Returns:
            * None
        '''

        # Get the geographic vertices and connectivities from the Gocad file

        fid = open(filename, 'w')
        if utm:
            vertices = self.Vertices*1.0e3
        else:
            vertices = self.Vertices_ll
        for i in range(vertices.shape[0]):
            v = vertices[i]
            fid.write('VRTX {} {} {} {}\n'.format(i+1,v[0],v[1],v[2]))
        for i in range(self.Faces.shape[0]):
            vid = self.Faces[i,:]+1
            fid.write('TRGL {} {} {}\n'.format(vid[0],vid[1],vid[2]))
        fid.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getStrikes(self):
        '''
        Returns an array of strikes.
        '''

        # all done in one line
        return np.array([self.getpatchgeometry(p)[5] for p in self.patch])
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getDips(self):
        '''
        Returns an array of dips.
        '''

        # all done in one line
        return np.array([self.getpatchgeometry(p)[6] for p in self.patch])
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getDepths(self, center=True):
        '''
        Returns an array of depths.

        Kwargs:
            * center        : If True, returns the center of the patches
        '''

        # All done in one line
        return np.array([self.getpatchgeometry(p, center=True)[2] for p in self.patch]) 

    # ----------------------------------------------------------------------
    def writePatches2File(self, filename, add_slip=None, scale=1.0, stdh5=None, decim=1):
        '''
        Writes the patch corners in a file that can be used in psxyz.

        Args:
            * filename      : Name of the file.

        Kwargs:
            * add_slip      : Put the slip as a value for the color. 
                              Can be None, strikeslip, dipslip, total, coupling
            * scale         : Multiply the slip value by a factor.
            * patch         : Can be 'normal' or 'equiv'
            * stdh5         : Get the standard deviation from a h5 file
            * decim         : Decimate the h5 file

        Returns:
            * None
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch) and add_slip is not None:
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # Write something
        print('Writing geometry to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # If an h5 file is specified, open it
        if stdh5 is not None:
            import h5py
            h5fid = h5py.File(stdh5, 'r')
            samples = h5fid['samples'].value[::decim,:]

        # Loop over the patches
        nPatches = len(self.patch)
        for pIndex in range(nPatches):

            # Select the string for the color
            string = '  '
            if add_slip is not None:
                if add_slip is 'coupling':
                    slp = self.coupling[pIndex]
                    string = '-Z{}'.format(slp)
                if add_slip is 'strikeslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex])
                    else:
                        slp = self.slip[pIndex,0]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'dipslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex+nPatches])
                    else:
                        slp = self.slip[pIndex,1]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'tensile':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex+2*nPatches])
                    else:
                        slp = self.slip[pIndex,2]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip is 'total':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex]**2 + samples[:,pIndex+nPatches]**2)
                    else:
                        slp = np.sqrt(self.slip[pIndex,0]**2 + self.slip[pIndex,1]**2)*scale
                    string = '-Z{}'.format(slp)

            # Put the parameter number in the file as well if it exists
            parameter = ' '
            if hasattr(self,'index_parameter') and add_slip is not None:
                i = np.int(self.index_parameter[pIndex,0])
                j = np.int(self.index_parameter[pIndex,1])
                k = np.int(self.index_parameter[pIndex,2])
                parameter = '# {} {} {} '.format(i,j,k)

            # Put the slip value
            if add_slip is not None:
                if add_slip=='coupling':
                    slipstring = ' # {}'.format(self.coupling[pIndex])
                else:
                    slipstring = ' # {} {} {} '.format(self.slip[pIndex,0],
                                               self.slip[pIndex,1], self.slip[pIndex,2])

            # Write the string to file
            if add_slip is None:
                fout.write('> {} {} \n'.format(string, parameter))
            else:
                fout.write('> {} {} {}  \n'.format(string,parameter,slipstring))

            # Write the 3 patch corners (the order is to be GMT friendly)
            p = self.patchll[pIndex]
            pp = p[0]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp = p[1]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))
            pp = p[2]; fout.write('{} {} {} \n'.format(pp[0], pp[1], pp[2]))

        # Close the file
        fout.close()

        # Close h5 file if it is open
        if stdh5 is not None:
            h5fid.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writeSlipDirection2File(self, filename, scale=1.0, factor=1.0,
                                neg_depth=False, ellipse=False,nsigma=1.):
        '''
        Write a psxyz compatible file to draw lines starting from the center 
        of each patch, indicating the direction of slip. Scale can be a real 
        number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'

        Args:
            * filename      : Name of the output file

        Kwargs:
            * scale         : Scale of the line
            * factor        : Multiply slip by a factor
            * neg_depth     : if True, depth is a negative nmber
            * ellipse       : Write the ellipse
            * nsigma        : Nxsigma for the ellipse

        Returns:
            * None
        '''

        # Copmute the slip direction
        self.computeSlipDirection(scale=scale, factor=factor, ellipse=ellipse,nsigma=nsigma)

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

        if ellipse:
            # Open the file
            fout = open('ellipse_'+filename, 'w')

            # Loop over the patches
            for e in self.ellipse:

                # Get ellipse points
                ex, ey, ez = e[:,0],e[:,1],e[:,2]

                # Depth
                if neg_depth:
                    ez = -1.0 * ez

                # Conversion to geographical coordinates
                lone,late = self.putm(ex*1000.,ey*1000.,inverse=True)

                # Write the > sign to the file
                fout.write('> \n')

                for lon,lat,z in zip(lone,late,ez):
                    fout.write('{} {} {} \n'.format(lon, lat, -1.*z))
            # Close file
            fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getEllipse(self, patch, ellipseCenter=None, Npoints=10, factor=1.0,
                   nsigma=1.):
        '''
        Compute the ellipse error given Cm for a given patch

        Args:
            * patch : Which patch to consider

        Kwargs:
            * center  : center of the ellipse
            * Npoints : number of points on the ellipse
            * factor  : scaling factor
            * nsigma  : will design a nsigma*sigma error ellipse

        Returns:
            * Ellipse   : Array containing the ellipse
        '''

        # Get Cm
        Cm = np.diag(self.Cm[patch,:2])
        Cm[0,1] = Cm[1,0] = self.Cm[patch,2]

        # Get strike and dip
        xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(patch, center=True)
        dip *= np.pi/180.
        strike *= np.pi/180.
        if ellipseCenter!=None:
            xc,yc,zc = ellipseCenter

        # Compute eigenvalues/eigenvectors
        D,V = np.linalg.eig(Cm)
        v1 = V[:,0]
        a = nsigma*np.sqrt(np.abs(D[0]))
        b = nsigma*np.sqrt(np.abs(D[1]))
        phi = np.arctan2(v1[1],v1[0])
        theta = np.linspace(0,2*np.pi,Npoints);

        # The ellipse in x and y coordinates
        Ex = a * np.cos(theta) * factor
        Ey = b * np.sin(theta) * factor

        # Correlation Rotation
        R  = np.array([[np.cos(phi), -np.sin(phi)],
                       [np.sin(phi), np.cos(phi)]])
        RE = np.dot(R,np.array([Ex,Ey]))

        # Strike/Dip rotation
        ME = np.array([RE[0,:], RE[1,:] * np.cos(dip), RE[1,:]*np.sin(dip)])
        R  = np.array([[np.sin(strike), -np.cos(strike), 0.0],
                       [np.cos(strike), np.sin(strike), 0.0],
                       [0.0, 0.0, 1.]])
        RE = np.dot(R,ME).T

        # Translation on Fault
        RE[:,0] += xc
        RE[:,1] += yc
        RE[:,2] += zc

        # All done
        return RE
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def computeSlipDirection(self, scale=1.0, factor=1.0, ellipse=False,nsigma=1.):
        '''
        Computes the segment indicating the slip direction.

        Kwargs:
            * scale     : can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
            * factor    : Multiply by a factor
            * ellipse   : Compute the ellipse
            * nsigma    : How many times sigma for the ellipse

        Returns:
            * None
        '''

        # Create the array
        self.slipdirection = []

        # Check Cm if ellipse
        if ellipse:
            self.ellipse = []
            assert(self.Cm!=None), 'Provide Cm values'

        # Loop over the patches
        if self.N_slip == None:
            self.N_slip = len(self.patch)
        for p in range(self.N_slip):
            # Get some geometry
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            # Get the slip vector
            slip = self.slip[p,:]
            rake = np.arctan2(slip[1],slip[0])

            # Compute the vector
            #x = np.sin(strike)*np.cos(rake) + np.sin(strike)*np.cos(dip)*np.sin(rake)
            #y = np.cos(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake)
            #z = -1.0*np.sin(dip)*np.sin(rake)
            x = (np.sin(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake))
            y = (np.cos(strike)*np.cos(rake) + np.sin(strike)*np.cos(dip)*np.sin(rake))
            z =  1.0*np.sin(dip)*np.sin(rake)

            # Scale these
            if scale.__class__ is float:
                sca = scale
            elif scale.__class__ is str:
                if scale in ('total'):
                    sca = np.sqrt(slip[0]**2 + slip[1]**2 + slip[2]**2)*factor
                elif scale in ('strikeslip'):
                    sca = slip[0]*factor
                elif scale in ('dipslip'):
                    sca = slip[1]*factor
                elif scale in ('tensile'):
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

            # Append ellipse
            if ellipse:
                self.ellipse.append(self.getEllipse(p,ellipseCenter=[xe, ye, ze],factor=factor,nsigma=nsigma))

            # Append slip direction
            self.slipdirection.append([[xc, yc, zc],[xe, ye, ze]])

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def deletepatch(self, patch, checkVertices=True, checkSlip=False):
        '''
        Deletes a patch.

        Args:
            * patch     : index of the patch to remove.

        Kwargs:
            * checkVertices : Make sure vertice array corresponds to patch corners
            * checkSlip     : Check that slip vector corresponds to patch corners

        Returns:
            * None
        '''

        # Save vertices
        vids = copy.deepcopy(self.Faces[patch])

        # Remove the patch
        del self.patch[patch]
        del self.patchll[patch]
        self.Faces = np.delete(self.Faces, patch, axis=0)
        
        # Check if vertices are to be removed
        v2del = []
        if checkVertices:
            for v in vids:
                if v not in self.Faces.flatten().tolist():
                    v2del.append(v)

        # Remove if needed
        if len(v2del)>0:
            self.deletevertices(v2del, checkPatch=False)

        # Clean slip vector
        if self.slip is not None and checkSlip:
            self.slip = np.delete(self.slip, patch, axis=0)
            self.N_slip = len(self.slip)
            if hasattr(self, 'numpatch'):
                self.numpatch -= 1
            else:
                self.numpatch = len(self.patch)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def deletevertices(self, iVertices, checkPatch=True, checkSlip=True):
        '''
        Deletes some vertices. If some patches are composed of these vertices 
        and checkPatch is True, deletes the patches.

        Args:
            * iVertices     : List of vertices to delete.

        Kwargs:
            * checkPatch    : Check and delete if patches are concerned.
            * checkSlip     : Check and delete if slip terms are concerned.

        Returns:
            * None
        ''' 

        # If some patches are concerned
        if checkPatch:
            # Check
            iPatch = []
            for iV in iVertices:
                i, j = np.where(self.Faces==iV)
                if len(i.tolist())>0:
                    iPatch.append(np.unique(i))
            if len(iPatch)>0:
                # Delete
                iPatch = np.unique(np.concatenate(iPatch)).tolist()
                self.deletepatches(iPatch, checkVertices=False, checkSlip=checkSlip)

        # Modify the vertex numbers in Faces
        newFaces = copy.deepcopy(self.Faces)
        for v in iVertices:
            i,j = np.where(self.Faces>v)
            newFaces[i,j] -= 1
        self.Faces = newFaces
         
        # Do the deletion
        self.Vertices = np.delete(self.Vertices, iVertices, axis=0)
        self.Vertices_ll = np.delete(self.Vertices_ll, iVertices, axis=0)
        
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def deletevertex(self, iVertex, checkPatch=True, checkSlip=True):
        '''
        Delete a Vertex. If some patches are composed of this vertex and 
        checkPatch is True, deletes the patches.

        Args:
            * iVertex       : index of the vertex to delete

        Kwargs:
            * checkPatch    : Check and delete if patches are concerned.
            * checkSlip     : Check and delete is slip is concerned.

        Returns:
            * None
        '''

        # Delete only one vertex
        self.deletevertices([iVertex], checkPatch, checkSlip=checkSlip)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def deletepatches(self, tutu, checkVertices=True, checkSlip=True):
        '''
        Deletes a list of patches.

        Args:
            * tutu      : List of indices

        Kwargs:
            * checkVertices : Check and delete if patches are concerned.
            * checkSlip     : Check and delete is slip is concerned.

        Returns:
            * None
        '''

        while len(tutu)>0:

            # Get index to delete
            i = tutu.pop()

            # delete it
            self.deletepatch(i, checkVertices=checkVertices, checkSlip=checkSlip)

            # Upgrade list
            for u in range(len(tutu)):
                if tutu[u]>i:
                    tutu[u] -= 1

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def refineMesh(self):
        '''
        Cuts all the patches in 4, based on the mid-point of each triangle and 
        builds a new fault from that.

        Returns:
            * None
        '''

        # Iterate over the fault patches
        newpatches = []
        for patch in self.patch:
            triangles = self.splitPatch(patch)
            for triangle in triangles:
                newpatches.append(triangle)

        # Delete all the patches 
        del self.patch
        del self.patchll
        del self.Vertices
        del self.Vertices_ll
        del self.Faces
        self.patch = None
        self.N_slip = None

        # Add the new patches
        self.addpatches(newpatches)

        # Update the depth of the bottom of the fault
        self.top   = np.min(self.Vertices[:,2])
        self.depth = np.max(self.Vertices[:,2])
        self.z_patches = np.linspace(self.depth, 0.0, 5)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def addpatches(self, patches):
        '''
        Adds patches to the list.

        Args:
            * patches     : List of patch geometries

        Returns:
            * None
        '''

        # Iterate
        for patch in patches:
            self.addpatch(patch)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def addpatch(self, patch, slip=[0, 0, 0]):
        '''
        Adds a patch to the list.

        Args:
            * patch     : Geometry of the patch to add (km, not lon lat)

        Kwargs:
            * slip      : List of the strike, dip and tensile slip.

        Returns:
            * None
        '''

        # Check if the list of patch exists
        if self.patch is None:
            self.patch = []

        # Check that patch is an array
        if type(patch) is list:
            patch = np.array(patch)
        assert type(patch) is np.ndarray, 'addPatch: Patch has to be a numpy array'

        # append the patch
        self.patch.append(patch)

        # modify the slip
        if self.N_slip!=None and self.N_slip==len(self.patch):
            sh = self.slip.shape
            nl = sh[0] + 1
            nc = 3
            tmp = np.zeros((nl, nc))
            if nl > 1:                      # Case where slip is empty
                tmp[:nl-1,:] = self.slip
            tmp[-1,:] = slip
            self.slip = tmp

        # Create Vertices and Faces if not there
        if not hasattr(self, 'Vertices'):
            self.Vertices = np.array([patch[0], patch[1], patch[2]])
            self.Faces = np.array([[0, 1, 2]])

        # Check if vertices are already there
        vids = []
        for p in patch:
            ii = np.flatnonzero(np.array([(p.tolist()==v).all() for v in self.Vertices]))
            if len(ii)==0:
                self.Vertices = np.insert(self.Vertices, self.Vertices.shape[0],
                        p, axis=0)
                vids.append(self.Vertices.shape[0]-1)
            else:
                vids.append(ii[0])
        self.Faces = np.insert(self.Faces, self.Faces.shape[0], vids, axis=0)

        # Vertices2ll
        self.vertices2ll()
        self.patch2ll()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def replacePatch(self, patch, iPatch):
        '''
        Replaces one patch by the given geometry.

        Args:
            * patch     : Patch geometry.
            * iPatch    : index of the patch to replace.

        Returns:    
            * None
        '''

        # Replace
        if type(patch) is list:
            patch = np.array(patch)
        assert type(patch) is np.ndarray, 'replacePatch: Patch must be an array'
        self.patch[iPatch] = patch

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def pointRotation3D(self, iPatch, iPoint, theta, p_axis1, p_axis2):
        '''
        Rotate a point with an arbitrary axis (fault tip)
        Used in rotatePatch
        
        Args:
            * iPatch: index of the patch to be rotated
            * iPoint: index of the patch corner (point) to be rotated
            * theta : angle of rotation in degrees
            * p_axis1 : first point of axis (ex: one side of a fault)
            * p_axis2 : second point to define the axis (ex: the other side of a fault)
            
        Returns:
            * rotated point
        Reference: 'Rotate A Point About An Arbitrary Axis (3D)' - Paul Bourke 
        '''
        def to_radians(angle):
            return np.divide(np.dot(angle, np.pi), 180.0)
    
        def to_degrees(angle):
            return np.divide(np.dot(angle, 180.0), np.pi)
        
        point = self.patch[iPatch][iPoint]
        
        # Translate so axis is at origin    
        p = point - p_axis1
    
        N = p_axis2 - p_axis1
        Nm = np.sqrt(N[0]**2 + N[1]**2 + N[2]**2)
        
        # Rotation axis unit vector
        n = [N[0]/Nm, N[1]/Nm, N[2]/Nm]
    
        # Matrix common factors     
        c = np.cos(to_radians(theta))
        t = 1 - np.cos(to_radians(theta))
        s = np.sin(to_radians(theta))
        X = n[0]
        Y = n[1]
        Z = n[2]
    
        # Matrix 'M'
        d11 = t*X**2 + c
        d12 = t*X*Y - s*Z
        d13 = t*X*Z + s*Y
        d21 = t*X*Y + s*Z
        d22 = t*Y**2 + c
        d23 = t*Y*Z - s*X
        d31 = t*X*Z - s*Y
        d32 = t*Y*Z + s*X
        d33 = t*Z**2 + c
    
        #            |p.x|
        # Matrix 'M'*|p.y|
        #            |p.z|
        q = np.empty((3))
        q[0] = d11*p[0] + d12*p[1] + d13*p[2]
        q[1] = d21*p[0] + d22*p[1] + d23*p[2]
        q[2]= d31*p[0] + d32*p[1] + d33*p[2]
        
        # Translate axis and rotated point back to original location    
        return np.array(q + p_axis1)
    # ----------------------------------------------------------------------
        
    # ----------------------------------------------------------------------
    def rotatePatch(self, iPatch , theta, p_axis1, p_axis2, verbose=False):
        '''
        Rotate a patch with an arbitrary axis (fault tip)
        Used by fault class uncertainties
        
        Args:
            * iPatch: index of the patch to be rotated
            * theta : angle of rotation in degrees
            * p_axis1 : first point of axis (ex: one side of a fault)
            * p_axis2 : second point to define the axis (ex: the other side of a fault)
            
        Returns:
            * rotated patch
        '''
        if verbose:
            print('Rotating patch {} '.format(iPatch))
        
        # Calculate rotated patch
        rotated_patch = [self.pointRotation3D(iPatch,0, theta, p_axis1, p_axis2),
                         self.pointRotation3D(iPatch,1, theta, p_axis1, p_axis2),
                         self.pointRotation3D(iPatch,2, theta, p_axis1, p_axis2)]
        
        patch = rotated_patch
        
        # Replace
        self.patch[iPatch] = np.array(patch)

        # Build the ll patch
        lon1, lat1 = self.xy2ll(patch[0][0], patch[0][1])
        z1 = patch[0][2]
        lon2, lat2 = self.xy2ll(patch[1][0], patch[1][1])
        z2 = patch[1][2]
        lon3, lat3 = self.xy2ll(patch[2][0], patch[2][1])
        z3 = patch[2][2]

        # append the ll patch
        patchll = [ [lon1, lat1, z1],
                    [lon2, lat2, z2],
                    [lon3, lat3, z3] ]

        # Replace
        self.patchll[iPatch] = np.array(patchll)
        return 
    # ----------------------------------------------------------------------    

    # ----------------------------------------------------------------------
    # Translate a patch
    def translatePatch(self, iPatch , tr_vector):
        '''
        Translate a patch
        Used by class uncertainties
        
        Args:
            * iPatch: index of the patch to be rotated
            * tr_vector: array, translation vector in 3D
            
        Returns:
            * None
        '''        
        # Calculate rotated patch
        tr_p1 = np.array( [ self.patch[iPatch][0][0]+tr_vector[0], 
                          self.patch[iPatch][0][1]+tr_vector[1], 
                          self.patch[iPatch][0][2]+tr_vector[2]])
        tr_p2 = np.array( [self.patch[iPatch][1][0]+tr_vector[0], 
                          self.patch[iPatch][1][1]+tr_vector[1], 
                          self.patch[iPatch][1][2]+tr_vector[2]])
        tr_p3 = np.array( [self.patch[iPatch][2][0]+tr_vector[0], 
                          self.patch[iPatch][2][1]+tr_vector[1], 
                          self.patch[iPatch][2][2]+tr_vector[2]])
        
        tr_patch=[tr_p1, tr_p2, tr_p3]
                                             
        # Replace
        self.patch[iPatch] = tr_patch

        # Build the ll patch
        lon1, lat1 = self.xy2ll(tr_patch[0][0], tr_patch[0][1])
        z1 = tr_patch[0][2]
        lon2, lat2 = self.xy2ll(tr_patch[1][0], tr_patch[1][1])
        z2 = tr_patch[1][2]
        lon3, lat3 = self.xy2ll(tr_patch[2][0], tr_patch[2][1])
        z3 = tr_patch[2][2]

        # append the ll patch
        patchll = [ [lon1, lat1, z1],
                    [lon2, lat2, z2],
                    [lon3, lat3, z3] ]

        # Replace
        self.patchll[iPatch] = np.array(patchll)
        return 
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getpatchgeometry(self, patch, center=False, retNormal=False, checkindex=True):
        '''
        Returns the patch geometry as needed for triangleDisp.

        Args:
            * patch         : index of the wanted patch or patch

        Kwargs:
            * center        : if true, returns the coordinates of the center of the patch. if False, returns the first corner
            * checkindex    : Checks the index of the patch
            * retNormal     : If True gives, also the normal vector to the patch

        Returns:
            * x, y, z, width, length, strike, dip, (normal)
        '''

        # Get the patch
        u = None
        if type(patch) in (int, np.int64, np.int, np.int32):
            u = patch
        else:
            if checkindex:
                u = self.getindex(patch)
        if u is not None:
            patch = self.patch[u]

        # Get the center of the patch
        x1, x2, x3 = self.getcenter(patch)

        # Get the vertices of the patch
        verts = copy.deepcopy(patch)
        p1, p2, p3 = [np.array([lst[1],lst[0],lst[2]]) for lst in verts]

        # Get a dummy width and height
        width = np.linalg.norm(p1 - p2)
        length = np.linalg.norm(p3 - p1)

        # Get the patch normal
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal)
        # Enforce clockwise circulation
        if np.round(normal[2],decimals=1) < 0:
            normal *= -1.0
            p2, p3 = p3, p2

        # If fault is vertical, force normal to be horizontal
        if np.round(normal[2],decimals=1) == 0.: 
            normal[2] = 0.
        # Force strike between 0 and 90 or between 270 and 360
            if normal[1] > 0:
                normal *= -1
                    
        # Get the strike vector and strike angle
        strike = np.arctan2(-normal[0], normal[1]) - np.pi
        if strike<0.:
            strike += 2*np.pi

        # Set the dip vector
        dip = np.arccos(normal[2])

        if retNormal:
            return x1, x2, x3, width, length, strike, dip, normal
        else:
            return x1, x2, x3, width, length, strike, dip
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distanceVertexToVertex(self, vertex1, vertex2, distance='center', lim=None):
        '''
        Measures the distance between two vertexes.

        Args:
            * vertex1   : first patch or its index
            * vertex2   : second patch or its index

        Kwargs:
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].
            * distance  : Useless argument only here for compatibility reasons

        Returns:
            * distance  : float
        '''

        if distance is 'center':

            # Get the centers
            x1, y1, z1 = vertex1
            x2, y2, z2 = vertex2

            # Compute the distance
            dis = np.sqrt((x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis > lim[0]:
                    dis = lim[1]

        else:
            raise NotImplementedError('only distance=center is implemented')

        # All done
        return dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distanceMatrix(self, distance='center', lim=None):
        '''
        Returns a matrix of the distances between patches.

        Kwargs:
            * distance  : distance estimation mode

                 - center : distance between the centers of the patches.
                 - no other method is implemented for now.
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].

        Returns:
            * distances : Array of floats
        '''

        # Assert 
        assert distance is 'center', 'No other method implemented than center'

        # Check
        if self.N_slip==None:
            self.N_slip = self.slip.shape[0]

        # Loop
        Distances = np.zeros((self.N_slip, self.N_slip))
        for i in range(self.N_slip):
            p1 = self.patch[i]
            for j in range(self.N_slip):
                if j == i:
                    continue
                p2 = self.patch[j]
                Distances[i,j] = self.distancePatchToPatch(p1, p2, distance='center', lim=lim)

        # All done
        return Distances
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distancePatchToPatch(self, patch1, patch2, distance='center', lim=None):
        '''
        Measures the distance between two patches.

        Args:
            * patch1    : first patch or its index
            * patch2    : second patch or its index

        Kwargs:
            * distance  : distance estimation mode

                    - center : distance between the centers of the patches.
                    - no other method is implemented for now.
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].

        Returns:
            * distace   : float
        '''

        if distance is 'center':

            # Get the centers
            x1, y1, z1 = self.getcenter(patch1)
            x2, y2, z2 = self.getcenter(patch2)

            # Compute the distance
            dis = np.sqrt((x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis > lim[0]:
                    dis = lim[1]

        else:
            raise NotImplementedError('only distance=center is implemented')

        # All done
        return dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def slip2dis(self, data, patch, slip=None):
        '''
        Computes the surface displacement for a given patch at the data location
        using a homogeneous half-space.

        Args:
            * data          : data object from gps or insar.
            * patch         : number of the patch that slips

        Kwargs:
            * slip          : if a number is given, that is the amount of slip along strike. If three numbers are given, that is the amount of slip along strike, along dip and opening. if None, values from self.slip are taken.

        Returns:
            * ss_dis        : Surface displacements due to strike slip
            * ds_dis        : Surface displacements due to dip slip
            * ts_dis        : Surface displacements due to tensile opening
        '''

        # Set the slip values
        if slip is None:
            SLP = [self.slip[patch,0], self.slip[patch,1], self.slip[patch,2]]
        elif slip.__class__ is float:
            SLP = [slip, 0.0, 0.0]
        elif slip.__class__ is list:
            SLP = slip

        # Get patch vertices
        vertices = list(self.patch[patch])

        # Get data position
        x = data.x
        y = data.y
        z = np.zeros_like(x)

        # Get strike slip displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, SLP[0], 0.0, 0.0)
        ss_dis = np.column_stack((ux, uy, uz))

        # Get dip slip displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, 0.0, SLP[1], 0.0)
        ds_dis = np.column_stack((ux, uy, uz))

        # Get opening displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, 0.0, 0.0, SLP[2])
        op_dis = np.column_stack((ux, uy, uz))

        # All done
        return ss_dis, ds_dis, op_dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildAdjacencyMap(self, verbose=True):
        '''
        For each triangle, find the indices of the adjacent (edgewise) triangles.

        Kwargs:
            * verbose

        Returns:
            * None
        '''
        if verbose:
            print("------------------------------------------")
            print("------------------------------------------")
            print("Building the adjacency map for all patches")

        self.adjacencyMap = []

        # Cache the vertices and faces arrays
        vertices, faces = self.Vertices, self.Faces

        # First find adjacent triangles for all triangles
        npatch = len(self.patch)
        for i in range(npatch):

            if verbose:
                sys.stdout.write('%i / %i\r' % (i, npatch))
                sys.stdout.flush()

            # Indices of Vertices of current patch
            refVertInds = faces[i,:]

            # Find triangles that share an edge
            adjacents = []
            for j in range(npatch):
                if j == i:
                    continue
                sharedVertices = np.intersect1d(refVertInds, faces[j,:])
                numSharedVertices = sharedVertices.size
                if numSharedVertices < 2:
                    continue
                adjacents.append(j)
                if len(adjacents) == 3:
                    break

            self.adjacencyMap.append(adjacents)

        if verbose:
            print('')
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildLaplacian(self, verbose=True, method=None, irregular=False):
        '''
        Build a discrete Laplacian smoothing matrix.

        Kwargs:
            * verbose       : Speak to me
            * method        : Not used, here for consistency purposes
            * irregular     : Not used, here for consistency purposes
        
        Returns:
            * Laplacian     : 2D array
        '''
        
        if self.adjacencyMap is None or len(self.adjacencyMap) != len(self.patch):
            self.buildAdjacencyMap(verbose=verbose)

        if verbose:
            print("------------------------------------------")
            print("------------------------------------------")
            print("Building the Laplacian matrix")

        # Pre-compute patch centers
        centers = self.getcenters()

        # Cache the vertices and faces arrays
        vertices, faces = self.Vertices, self.Faces

        # Allocate array for Laplace operator
        npatch = len(self.patch)
        D = np.zeros((npatch,npatch))

        # Loop over patches
        for i in range(npatch):

            if verbose:
                sys.stdout.write('%i / %i\r' % (i, npatch))
                sys.stdout.flush()

            # Center for current patch
            refCenter = np.array(centers[i])

            # Compute Laplacian using adjacent triangles
            hvals = []
            adjacents = self.adjacencyMap[i]
            for index in adjacents:
                pcenter = np.array(centers[index])
                dist = np.linalg.norm(pcenter - refCenter)
                hvals.append(dist)
            if len(hvals) == 3:
                h12, h13, h14 = hvals
                D[i,adjacents[0]] = -h13*h14
                D[i,adjacents[1]] = -h12*h14
                D[i,adjacents[2]] = -h12*h13
                sumProd = h13*h14 + h12*h14 + h12*h13
            elif len(hvals) == 2:
                h12, h13 = hvals
                # Make a virtual patch
                h14 = max(h12, h13)
                D[i,adjacents[0]] = -h13*h14
                D[i,adjacents[1]] = -h12*h14
                sumProd = h13*h14 + h12*h14 + h12*h13
            D[i,i] = sumProd

        if verbose:
            print('')
        D = D / np.max(np.abs(np.diag(D)))
        return D
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getcenter(self, p):
        '''
        Get the center of one triangular patch.

        Args:
            * p     : Patch geometry.

        Returns:
            * x,y,z : floats 
        '''

        # Get center
        if type(p) is int:
            p1, p2, p3 = self.patch[p]
        else:
            p1, p2, p3 = p

        # Compute the center
        x = (p1[0] + p2[0] + p3[0]) / 3.0
        y = (p1[1] + p2[1] + p3[1]) / 3.0
        z = (p1[2] + p2[2] + p3[2]) / 3.0

        # All done
        return x,y,z
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    def computetotalslip(self):
        '''
        Computes the total slip and stores it self.totalslip
        '''

        # Computes the total slip
        self.totalslip = np.sqrt(self.slip[:,0]**2 + self.slip[:,1]**2 \
                + self.slip[:,2]**2)

        # All done
        return
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    def getcenters(self):
        '''
        Get the center of the patches.

        Returns:
            * centers:  list of triplets
        '''

        # Initialize a list
        center = []

        # loop over the patches
        for p in self.patch:
            x, y, z = self.getcenter(p)
            center.append(np.array([x, y, z]))

        # All done
        return center
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    def surfacesimulation(self, box=None, disk=None, err=None, npoints=None, 
                          lonlat=None, slipVec=None):
        '''
        Takes the slip vector and computes the surface displacement that 
        corresponds on a regular grid.

        Kwargs:
            * box       : A list of [minlon, maxlon, minlat, maxlat].
            * disk      : list of [xcenter, ycenter, radius, n]
            * lonlat    : Arrays of lat and lon. [lon, lat]
            * err       : Compute random errors and scale them by {err}
            * slipVec   : Replace slip by what is in slipVec
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # create a fake gps object
        self.sim = gpsclass('simulation', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0)

        # Create a lon lat grid
        if lonlat is None:
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

        # Load the slip values if provided
        if slipVec is not None:
            nPatches = len(self.patch)
            print(nPatches, slipVec.shape)
            assert slipVec.shape == (nPatches,3), 'mismatch in shape for input slip vector'
            self.slip = slipVec

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
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def cumdistance(self, discretized=False):
        '''
        Computes the distance between the first point of the fault and every other
        point, when you walk along the fault.

        Kwargs:
            * discretized           : if True, use the discretized fault trace

        Returns:
            * cum                   : Array of floats
        '''

        # Get the x and y positions
        if discretized:
            x = self.xi
            y = self.yi
        else:
            x = self.xf
            y = self.yf

        # initialize
        dis = np.zeros((x.shape[0]))

        # Loop
        for i in np.arange(1,x.shape[0]):
            d = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
            dis[i] = dis[i-1] + d

        # all done
        return dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def AverageAlongStrikeOffsets(self, name, insars, filename, 
                                        discretized=True, smooth=None):
        '''
        !Untested in a looong time...!

        If the profiles have the lon lat vectors as the fault,
        This routines averages it and write it to an output file.
        '''

        if discretized:
            lon = self.loni
            lat = self.lati
        else:
            lon = self.lon
            lat = self.lat

        # Check if good
        for sar in insars:
            dlon = sar.AlongStrikeOffsets[name]['lon']
            dlat = sar.AlongStrikeOffsets[name]['lat']
            assert (dlon==lon).all(), '{} dataset rejected'.format(sar.name)
            assert (dlat==lat).all(), '{} dataset rejected'.format(sar.name)

        # Get distance
        x = insars[0].AlongStrikeOffsets[name]['distance']

        # Initialize lists
        D = []; AV = []; AZ = []; LO = []; LA = []

        # Loop on the distance
        for i in range(len(x)):

            # initialize average
            av = 0.0
            ni = 0.0

            # Get values
            for sar in insars:
                o = sar.AlongStrikeOffsets[name]['offset'][i]
                if np.isfinite(o):
                    av += o
                    ni += 1.0

            # if not only nan
            if ni>0:
                d = x[i]
                av /= ni
                az = insars[0].AlongStrikeOffsets[name]['azimuth'][i]
                lo = lon[i]
                la = lat[i]
            else:
                d = np.nan
                av = np.nan
                az = np.nan
                lo = lon[i]
                la = lat[i]

            # Append
            D.append(d)
            AV.append(av)
            AZ.append(az)
            LO.append(lo)
            LA.append(la)


        # smooth?
        if smooth is not None:
            # Arrays
            D = np.array(D); AV = np.array(AV); AZ = np.array(AZ); LO = np.array(LO); LA = np.array(LA)
            # Get the non nans
            u = np.flatnonzero(np.isfinite(AV))
            # Gaussian Smoothing
            dd = np.abs(D[u][:,None] - D[u][None,:])
            dd = np.exp(-0.5*dd*dd/(smooth*smooth))
            norm = np.sum(dd, axis=1)
            dd = dd/norm[:,None]
            AV[u] = np.dot(dd,AV[u])
            # List
            D = D.tolist(); AV = AV.tolist(); AZ = AZ.tolist(); LO = LO.tolist(); LA = LA.tolist()

        # Open file and write header
        fout = open(filename, 'w')
        fout.write('# Distance (km) || Offset || Azimuth (rad) || Lon || Lat \n')

        # Write to file
        for i in range(len(D)):
            d = D[i]; av = AV[i]; az = AZ[i]; lo = LO[i]; la = LA[i]
            fout.write('{} {} {} {} {} \n'.format(d,av,az,lo,la))

        # Close the file
        fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def ExtractAlongStrikeVariationsOnDiscretizedFault(self, depth=0.5, filename=None, discret=0.5):
        '''
        ! Untested in a looong time !
        
        Extracts the Along Strike variations of the slip at a given depth, resampled along the discretized fault trace.

        Kwargs:
            * depth       : Depth at which we extract the along strike variations of slip.
            * discret     : Discretization length
            * filename    : Saves to a file.

        Returns:
            * None
        '''

        # Import things we need
        import scipy.spatial.distance as scidis

        # Dictionary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # Creates the list where we store things
        # [lon, lat, strike-slip, dip-slip, tensile, distance, xi, yi]
        Var = []

        # Open the output file if needed
        if filename is not None:
            fout = open(filename, 'w')
            fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Distance to origin (km) | Position (x,y) (km)\n')

        # Discretize the fault
        if discret is not None:
            self.discretize(every=discret, tol=discret/10., fracstep=discret/12.)
        nd = self.xi.shape[0]

        # Compute the cumulative distance along the fault
        dis = self.cumdistance(discretized=True)

        # Get the patches concerned by the depths asked
        dPatches = []
        sPatches = []
        for p in self.patch:
            # Check depth
            if ((p[0,2]<=depth) and (p[2,2]>=depth)):
                # Get patch
                sPatches.append(self.getslip(p))
                # Put it in dis
                xc, yc = self.getcenter(p)[:2]
                d = scidis.cdist([[xc, yc]], [[self.xi[i], self.yi[i]] for i in range(self.xi.shape[0])])[0]
                imin1 = d.argmin()
                dmin1 = d[imin1]
                d[imin1] = 99999999.
                imin2 = d.argmin()
                dmin2 = d[imin2]
                dtot=dmin1+dmin2
                # Put it along the fault
                xcd = (self.xi[imin1]*dmin1 + self.xi[imin2]*dmin2)/dtot
                ycd = (self.yi[imin1]*dmin1 + self.yi[imin2]*dmin2)/dtot
                # Distance
                if dmin1<dmin2:
                    jm = imin1
                else:
                    jm = imin2
                dPatches.append(dis[jm] + np.sqrt( (xcd-self.xi[jm])**2 + (ycd-self.yi[jm])**2) )

        # Create the interpolator
        ssint = sciint.interp1d(dPatches, [sPatches[i][0] for i in range(len(sPatches))], kind='linear', bounds_error=False)
        dsint = sciint.interp1d(dPatches, [sPatches[i][1] for i in range(len(sPatches))], kind='linear', bounds_error=False)
        tsint = sciint.interp1d(dPatches, [sPatches[i][2] for i in range(len(sPatches))], kind='linear', bounds_error=False)

        # Interpolate
        for i in range(self.xi.shape[0]):
            x = self.xi[i]
            y = self.yi[i]
            lon = self.loni[i]
            lat = self.lati[i]
            d = dis[i]
            ss = ssint(d)
            ds = dsint(d)
            ts = tsint(d)
            Var.append([lon, lat, ss, ds, ts, d, x, y])
            # Write things if asked
            if filename is not None:
                fout.write('{} {} {} {} {} {} {} {} \n'.format(lon, lat, ss, ds, ts, d, x, y))

        # Store it in AlongStrike
        self.AlongStrike['Depth {}'.format(depth)] = np.array(Var)

        # Close fi needed
        if filename is not None:
            fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def ExtractAlongStrikeVariations(self, depth=0.5, origin=None, filename=None, orientation=0.0):
        '''
        Extract the Along Strike Variations of the creep at a given depth

        Kwargs:
            * depth   : Depth at which we extract the along strike variations of slip.
            * origin  : Computes a distance from origin. Give [lon, lat].
            * filename: Saves to a file.
            * orientation: defines the direction of positive distances.

        Returns:
            * None
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # Dictionary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # Creates the List where we will store things
        # For each patch, it will be [lon, lat, strike-slip, dip-slip, tensile, distance]
        Var = []

        # Creates the orientation vector
        Dir = np.array([np.cos(orientation*np.pi/180.), np.sin(orientation*np.pi/180.)])

        # initialize the origin
        x0 = 0
        y0 = 0
        if origin is not None:
            x0, y0 = self.ll2xy(origin[0], origin[1])

        # open the output file
        if filename is not None:
            fout = open(filename, 'w')
            fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Patch Area (km2) | Distance to origin (km) \n')

        # compute area, if not done yet
        if not hasattr(self,'area'):
            self.computeArea()

        # Loop over the patches
        for p in self.patch:

            # Get depth range
            dmin = np.min([p[i,2] for i in range(4)])
            dmax = np.max([p[i,2] for i in range(4)])

            # If good depth, keep it
            if ((depth>=dmin) & (depth<=dmax)):

                # Get index
                io = self.getindex(p)

                # Get the slip and area
                slip = self.slip[io,:]
                area = self.area[io]

                # Get patch center
                xc, yc, zc = self.getcenter(p)
                lonc, latc = self.xy2ll(xc, yc)

                # Computes the horizontal distance
                vec = np.array([x0-xc, y0-yc])
                sign = np.sign( np.dot(Dir,vec) )
                dist = sign * np.sqrt( (xc-x0)**2 + (yc-y0)**2 )

                # Assemble
                o = [lonc, latc, slip[0], slip[1], slip[2], area, dist]

                # write output
                if filename is not None:
                    fout.write('{} {} {} {} {} {} \n'.format(lonc, latc, slip[0], slip[1], slip[2], area, dist))

                # append
                Var.append(o)

        # Close the file
        if filename is not None:
            fout.close()

        # Stores it
        self.AlongStrike['Depth {}'.format(depth)] = np.array(Var)

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def ExtractAlongStrikeAllDepths(self, filename=None, discret=0.5):
        '''
        Extracts the Along Strike Variations of the creep at all depths for 
        the discretized fault trace.

        Kwargs:
            * filename      : Name of the output file
            * discret       : Fault discretization

        Returns:
            * None
        '''

        # Dictionnary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # If filename provided, create it
        if filename is not None:
            fout = open(filename, 'w')

        # Create the list of depths
        depths = np.unique(np.array([[self.patch[i][u,2] for u in range(4)] for i in range(len(self.patch))]).flatten())
        depths = depths[:-1] + (depths[1:] - depths[:-1])/2.

        # Discretize the fault
        self.discretize(every=discret, tol=discret/10., fracstep=discret/12.)

        # For a list of depths, iterate
        for d in depths.tolist():

            # Get the values
            self.ExtractAlongStrikeVariationsOnDiscretizedFault(depth=d, filename=None, discret=None)

            # If filename, write to it
            if filename is not None:
                fout.write('> # Depth = {} \n'.format(d))
                fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Distance to origin (km) | x, y \n')
                Var = self.AlongStrike['Depth {}'.format(d)]
                for i in range(Var.shape[0]):
                    lon = Var[i,0]
                    lat = Var[i,1]
                    ss = Var[i,2]
                    ds = Var[i,3]
                    ts = Var[i,4]
                    dist = Var[i,5]
                    x = Var[i,6]
                    y = Var[i,7]
                    fout.write('{} {} {} {} {} {} {} \n'.format(lon, lat, ss, ds, ts, area, dist, x, y))

        # Close file if done
        if filename is not None:
            fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def plot(self, figure=134, slip='total', equiv=False, show=True, 
             axesscaling=True, norm=None, linewidth=1.0, plot_on_2d=True, 
             drawCoastlines=True, expand=0.2):
        '''
        Plot the available elements of the fault.
        
        Kwargs:
            * figure        : Number of the figure.
            * slip          : What slip to plot
            * equiv         : useless. For consitency between fault objects
            * show          : Show me
            * axesscaling   : Scale the axis
            * Norm          : colorbar min and max values
            * linewidth     : Line width in points
            * plot_on_2d    : Plot on a map as well
            * drawCoastline : Self-explanatory argument...
            * expand        : Expand the map by {expand} degree around the edges
                              of the fault.
        
        Returns:
            * None
        '''

        # Get lons lats
        lon = np.unique(np.array([p[:,0] for p in self.patchll]))
        lon[lon<0.] += 360.
        lat = np.unique(np.array([p[:,1] for p in self.patchll]))
        lonmin = lon.min()-expand
        lonmax = lon.max()+expand
        latmin = lat.min()-expand
        latmax = lat.max()+expand

        # Create a figure
        fig = geoplot(figure=figure, lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax)

        # Draw the coastlines
        if drawCoastlines:
            fig.drawCoastlines(drawLand=False, parallels=5, meridians=5, drawOnFault=True)

        # Draw the fault
        fig.faultpatches(self, slip=slip, norm=norm, colorbar=True, plot_on_2d=plot_on_2d)

        # show
        if show:
            showFig = ['fault']
            if plot_on_2d:
                showFig.append('map')
            fig.show(showFig=showFig)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def plotMayavi(self, neg_depth=True, value_to_plot='total', colormap='jet',
                   reverseSign=False):
        '''
        ! OBSOLETE BUT KEPT HERE TO BE TESTED IN THE FUTURE !
        Plot 3D representation of fault using MayaVi.

        Args:
            * neg_depth     : Flag to specify if patch depths are negative or positive
            * value_to_plot : What to plot on patches
            * colormap      : Colormap for patches
            * reverseSign   : Flag to reverse sign of value_to_plot

        ! OBSOLETE BUT KEPT HERE TO BE TESTED IN THE FUTURE !
        '''
        try:
            from mayavi import mlab
        except ImportError:
            print('mayavi module not installed. skipping plotting...')
            return

        # Sign factor for negative depths
        negFactor = -1.0
        if neg_depth:
            negFactor = 1.0

        # Sign for values
        valueSign = 1.0
        if reverseSign:
            valueSign = -1.0

        # Plot the wireframe
        x, y, z = self.Vertices[:,0], self.Vertices[:,1], self.Vertices[:,2]
        z *= negFactor
        mesh = mlab.triangular_mesh(x, y, z, self.Faces, representation='wireframe',
                                    opacity=0.6, color=(0.0,0.0,0.0))

        # Compute the scalar value to color the patches
        if value_to_plot == 'total':
            self.computetotalslip()
            plotval = self.totalslip
        elif value_to_plot == 'strikeslip':
            plotval = self.slip[:,0]
        elif value_to_plot == 'dipslip':
            plotval = self.slip[:,1]
        elif value_to_plot == 'tensile':
            plotval = self.slip[:,2]
        elif value_to_plot == 'index':
            plotval = np.linspace(0, len(self.patch)-1, len(self.patch))
        else:
            assert False, 'unsupported value_to_plot'

        # Assign the scalar data to a source dataset
        cell_data = mesh.mlab_source.dataset.cell_data
        cell_data.scalars = valueSign * plotval
        cell_data.scalars.name = 'Cell data'
        cell_data.update()

        # Make a new mesh with the scalar data applied to patches
        mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='Cell data')
        surface = mlab.pipeline.surface(mesh2, colormap=colormap)

        mlab.colorbar(surface)
        mlab.show()

        return
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    def mapFault2Fault(self, Map, fault):
        '''
        User provides a Mapping function np.array((len(self.patch), len(fault.patch)))
        and a fault and the slip from the argument
        fault is mapped into self.slip.
        
        Function just does
        self.slip[:,0] = np.dot(Map,fault.slip)
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # Get the number of patches
        nPatches = len(self.patch)
        nPatchesExt = len(fault.patch)

        # Assert the Mapping function is correct
        assert(Map.shape==(nPatches,nPatchesExt)), 'Mapping function has the wrong size...'

        # Map the slip
        self.slip[:,0] = np.dot(Map, fault.slip[:,0])
        self.slip[:,1] = np.dot(Map, fault.slip[:,1])
        self.slip[:,2] = np.dot(Map, fault.slip[:,2])

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def mapUnder2Above(self, deepfault):
        '''
        This routine is very very particular. It only works with 2 vertical faults.
        It Builds the mapping function from one fault to another, when these are vertical.
        These two faults must have the same surface trace. If the deep fault has more than one raw of patches,
        it might go wrong and give some unexpected results.

        Args:
            * deepfault     : Deep section of the fault.
        '''

        # Assert faults are compatible
        assert ( (self.lon==deepfault.lon).all() and (self.lat==deepfault.lat).all()), 'Surface traces are different...'

        # Check that all patches are verticals
        dips = np.array([self.getpatchgeometry(i)[-1]*180./np.pi for i in range(len(self.patch))])
        assert((dips == 90.).all()), 'Not viable for non-vertical patches, fault {}....'.format(self.name)
        deepdips = np.array([deepfault.getpatchgeometry(i)[-1]*180./np.pi for i in range(len(deepfault.patch))])
        assert((deepdips == 90.).all()), 'Not viable for non-vertical patches, fault {}...'.format(deepfault.name)

        # Get the number of patches
        nPatches = len(self.patch)
        nDeepPatches = len(deepfault.patch)

        # Create the map from under to above
        Map = np.zeros((nPatches, nDeepPatches))

        # Discretize the surface trace quite finely
        self.discretize(every=0.5, tol=0.05, fracstep=0.02)

        # Compute the cumulative distance along the fault
        dis = self.cumdistance(discretized=True)

        # Compute the cumulative distance between the beginning of the fault and the corners of the patches
        distance = []
        for p in self.patch:
            D = []
            # for each corner
            for c in p:
                # Get x,y
                x = c[0]
                y = c[1]
                # Get the index of the nearest xi value under x
                i = np.flatnonzero(x>=self.xi)[-1]
                # Get the corresponding distance along the fault
                d = dis[i] + np.sqrt( (x-self.xi[i])**2 + (y-self.yi[i])**2 )
                # Append
                D.append(d)
            # Array unique
            D = np.unique(np.array(D))
            # append
            distance.append(D)

        # Do the same for the deep patches
        deepdistance = []
        for p in deepfault.patch:
            D = []
            for c in p:
                x = c[0]
                y = c[1]
                i = np.flatnonzero(x>=self.xi)
                if len(i)>0:
                    i = i[-1]
                    d = dis[i] + np.sqrt( (x-self.xi[i])**2 + (y-self.yi[i])**2 )
                else:
                    d = 99999999.
                D.append(d)
            D = np.unique(np.array(D))
            deepdistance.append(D)

        # Numpy arrays
        distance = np.array(distance)
        deepdistance = np.array(deepdistance)

        # Loop over the patches to find out which are over which
        for p in range(len(self.patch)):

            # Get the patch distances
            d1 = distance[p,0]
            d2 = distance[p,1]

            # Get the index for the points
            i1 = np.intersect1d(np.flatnonzero((d1>=deepdistance[:,0])), np.flatnonzero((d1<deepdistance[:,1])))[0]
            i2 = np.intersect1d(np.flatnonzero((d2>deepdistance[:,0])), np.flatnonzero((d2<=deepdistance[:,1])))[0]

            # two cases possible:
            if i1==i2:              # The shallow patch is fully inside the deep patch
                Map[p,i1] = 1.0     # All the slip comes from this patch
            else:                   # The shallow patch is on top of several patches
                # two cases again
                if np.abs(i2-i1)==1:       # It covers the boundary between 2 patches
                    delta1 = np.abs(d1-deepdistance[i1][1])
                    delta2 = np.abs(d2-deepdistance[i2][0])
                    total = delta1 + delta2
                    delta1 /= total
                    delta2 /= total
                    Map[p,i1] = delta1
                    Map[p,i2] = delta2
                else:                       # It is larger than the boundary between 2 patches and covers several deep patches
                    delta = []
                    delta.append(np.abs(d1-deepdistance[i1][1]))
                    for i in range(i1+1,i2):
                        delta.append(np.abs(deepdistance[i][1]-deepdistance[i][0]))
                    delta.append(np.abs(d2-deepdistance[i2][0]))
                    delta = np.array(delta)
                    total = np.sum(delta)
                    delta = delta/total
                    for i in range(i1,i2+1):
                        Map[p,i] = delta

        # All done
        return Map
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getSubSourcesFault(self, verbose=True):
        '''
        Returns a TriangularPatches fault object with each triangle
        corresponding to the subsources used for plotting.
    
        Kwargs:
            * verbose       : Talk to me (default: True)

        Returns:
            * fault         : Returns a triangularpatches instance
        '''

        # Import What is needed
        from .EDKSmp import dropSourcesInPatches as Patches2Sources

        # Drop the sources in the patches and get the corresponding fault
        Ids, xs, ys, zs, strike, dip, Areas, fault = Patches2Sources(self, 
                                                verbose=verbose,
                                                returnSplittedPatches=True)
        self.plotSources = [Ids, xs, ys, zs, strike, dip, Areas]

        # Interpolate the slip on each subsource
        fault.initializeslip()
        fault.slip[:,0] = self._getSlipOnSubSources(Ids, xs, ys, zs, self.slip[:,0])
        fault.slip[:,1] = self._getSlipOnSubSources(Ids, xs, ys, zs, self.slip[:,1])
        fault.slip[:,2] = self._getSlipOnSubSources(Ids, xs, ys, zs, self.slip[:,2])

        # All done
        return fault
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def findAsperities(self, function, slip='strikeslip', verbose=True):
        '''
        Finds the number, size and location of asperities that are identified by the 
        given function.

        Args:
            * function          : Function that takes an array the size of the number of patches and returns an array of bolean the same size. Trues are within the asperity.

        Kwargs:
            * slip              : Which slip vector do you want to apply the function to
            * verbose           : Talk to me?

        Returns:
            * Asperities
        '''

        # Assert 
        assert self.patchType is 'triangle', 'Not implemented for Triangular tents'

        # Update the map
        def _checkUpdate(check, iTriangle, modifier, fault):
            # Get the 3 surrounding triangles
            Adjacents = fault.adjacencyMap[iTriangle]
            # Check if they are in the asperity
            modify = [iTriangle]
            for adjacent in Adjacents:
                if check[adjacent]==1.: modify.append(adjacent)
            # Modify the map
            for mod in modify: check[mod] = modifier
            # Return the triangles surrounding
            modify.remove(iTriangle)
            return modify

        # Get the array to test
        if slip is 'strikeslip':
            values = self.slip[:,0]
        elif slip is 'dipslip':
            values = self.slip[:,1]
        elif slip is 'tensile':
            values = self.slip[:,2]
        elif slip is 'coupling':
            values = self.coupling
        else:
            print('findAsperities: Unknown type slip vector...')

        # Get the bolean array
        test = function(values).astype(float)

        # Build the adjacency Map
        if self.adjacencyMap is None:
            self.buildAdjacencyMap(verbose=verbose)

        # Number of the first asperity
        i = 1

        # We iterate until no triangle has been classified in an asperity
        # 0 means the triangle is not in an asperity
        # 1 means the triangle is in an asperity
        # 2 or more means the triangle is in an asperity and has been classified
        while len(np.flatnonzero(test==1.))>0:

            # Pick a triangle inside an asperity
            iTriangle = np.flatnonzero(test==1.)[0]
 
            # This is asperity i
            i += 1

            # Talk to me
            if verbose:
                print('Dealing with asperity #{}'.format(i))

            # Build the list of new triangles to check
            toCheck = _checkUpdate(test, iTriangle, i, self)

            # While toCheck has stuff to check, check them
            nT = 0
            while len(toCheck)>0:
                # Get triangle to check
                iCheck = toCheck.pop()
                if verbose:
                    nT += 1
                    sys.stdout.write('\r Triangles: {}'.format(nT))
                    sys.stdout.flush()
                # Check it
                toCheck += _checkUpdate(test, iCheck, i, self)

        # Normally, all the 1. have been replaced by 2., 3., etc
        
        # Find the unique numbers in test
        Counters = np.unique(test).tolist()
        Counters.remove(0.)
    
        # Get the asperities
        Asperities = []
        for counter in Counters:
            Asperities.append(np.flatnonzero(test==counter))

        # All done
        return Asperities
    # ----------------------------------------------------------------------

#EOF
