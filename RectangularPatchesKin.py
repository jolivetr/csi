'''
A parent class that deals with rectangular patches fault

Written by R. Jolivet, Z. Duputel and Bryan Riel November 2013
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import matplotlib.path as path
import scipy.signal as signal
from   glob import glob
import shutil as sh
import copy
import sys
import os

# Personals
from .RectangularPatches import RectangularPatches
from .stressfield import stressfield
from . import okadafull


class RectangularPatchesKin(RectangularPatches):
    '''
    A class that can handle what is required for a kinematic inversion 
    with AlTar for a fault with rectangular patches.

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
    # Initialize class
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, 
                       lat0=None, verbose=True):
        
        # Parent class init
        super(RectangularPatchesKin,self).__init__(name,
                                                   utmzone=utmzone,
                                                   ellps=ellps, 
                                                   lon0=lon0,
                                                   lat0=lat0,
                                                   verbose=verbose)

        # Specify the type of patch
        self.patchType = 'rectangle'

        # Allocate depth and number of patches
        self.numz = None            # Number of patches along dip        

        # Hypocenter coordinates
        self.hypo_x   = None
        self.hypo_y   = None
        self.hypo_z   = None
        self.hypo_lon = None
        self.hypo_lat = None
        self.hypo_patch_index = None
                
        # Patch objects
        self.patch = None
        self.grid  = None
        self.vr    = None
        self.tr    = None
        self.mu    = None

        # bigG and bigD
        self.bigD_map = None
        self.bigG = None
        self.bigD = None
        
        # Patch index mapping along strike and along dip
        self.fault_map = None

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setHypoXY(self,x,y, UTM=True):
        '''
        Set hypocenter attributes from x,y
        
        East/West UTM/Lon coordinates, depth attributes are set

        Args:
            * x     : east  coordinates 
            * y     : north coordinates
            * UTM   : Set true is x and y are in km, false if x and y are in degrees

        Returns:
            * None
        '''

        # If UTM==False, convert x=lon/y=lat to UTM
        if not UTM:
            self.hypo_x,self.hypo_y = self.ll2xy(x,y)
        else:
            self.hypo_x = x
            self.hypo_y = y

        # Check if within a patch
        hypo_point = np.array([self.hypo_x,self.hypo_y])
        for p in self.patch:
            Reg = []
            for v in p:
                Reg.append([v[0],v[1]])
            Reg = np.array(Reg)
            region = path.Path(Reg,closed=False)
            if region.contains_point(hypo_point):
                x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(p, center=True)
                dx = self.hypo_x-x1
                dy = self.hypo_y-x2            
                dh = dy*np.sin(strike)-dx*np.cos(strike)                
                self.hypo_z = x3 - dh*np.tan(dip)
                self.hypo_patch_index = self.getindex(p)

        # UTM to lat/lon conversion        
        self.hypo_lon,self.hypo_lat = self.xy2ll(self.hypo_x,self.hypo_y)
        
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setHypoOnFault(self,h_strike,h_dip):
        '''
        Set hypocenter attributes from fault coordinates

        Args:
            * h_strike: Along strike distance from the center of the top left patch
            * h_dip:    Along dip distance from the center of the top left patch

        Returns:
            * None
        '''
        
        # Fault map must exists
        assert self.fault_map is not None, 'Fault map must exists'
        
        # Initialize hypocenter values
        self.hypo_x = None
        self.hypo_y = None
        self.hypo_z = None
        self.hypo_lon = None
        self.hypo_lat = None
        self.hypo_patch_index = None
        
        # Check if within a patch
        hypo_point = np.array([h_strike,h_dip])
        x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(0, center=True)
        length = np.round(length,3)
        width  = np.round(width,3)
        for p in range(len(self.patch)):
            # Get patch size
            x1, x2, x3, W, L, strike, dip = self.getpatchgeometry(p, center=True)
            assert np.round(W,3)==width,  'Patch width  must be homogeneous accross fault (%f vs %f)'%(W,width)
            assert np.round(L,3)==length, 'Patch length must be homogeneous accross fault  (%f vs %f)'%(W,length)
            # Get patch onfault coordinates
            nstrike,ndip = self.fault_map[p]
            p_strike = nstrike * L
            p_dip    = ndip    * W
            # Define region
            Reg = [[p_strike-L/2,p_dip-W/2],
                   [p_strike-L/2,p_dip+W/2],
                   [p_strike+L/2,p_dip+W/2],
                   [p_strike+L/2,p_dip-W/2]]
            Reg = np.array(Reg)
            region = path.Path(Reg,closed=False)
            # If hypocenter within that region
            if region.contains_point(hypo_point):
                self.hypo_patch_index = p
                d1  = h_strike - p_strike 
                d2  = h_dip    - p_dip    
                d2h = d2 * np.cos(dip)                
                dx = d1*np.sin(strike)+d2h*np.cos(strike)
                dy = d1*np.cos(strike)-d2h*np.sin(strike)
                dz = d2 * np.sin(dip)
                self.hypo_x = x1 + dx
                self.hypo_y = x2 + dy
                self.hypo_z = x3 + dz
                self.hypo_lon,self.hypo_lat = self.xy2ll(self.hypo_x,self.hypo_y)
                break
                
        # Check if everything is correctly assigned
        assert self.hypo_x != None
        assert self.hypo_y != None
        assert self.hypo_z != None
        assert self.hypo_lon != None
        assert self.hypo_lat != None
        assert self.hypo_patch_index != None

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def read3DsquareGrid(self, filename):
        '''
        This routine read the square fault geometry

        :Format: 
        
        +---+---+-----+-----+-------+------+---+----+--+
        |lon|lat|E[km]|N[km]|Dep[km]|strike|dip|Area|ID|
        +===+===+=====+=====+=======+======+===+====+==+
        |   |   |     |     |       |      |   |    |  |
        +---+---+-----+-----+-------+------+---+----+--+
        |   |   |     |     |       |      |   |    |  |
        +---+---+-----+-----+-------+------+---+----+--+

        Args:
            * filename      : name of output file

        Returns:
            * None
        '''

        # Open the output file
        flld = open(filename,'r')
                
        # Loop over the patches
        self.patch     = []
        self.patchll   = []
        self.z_patches = []
        for l in flld:
            if l.strip()[0]=='#':
                continue
            items  = l.strip().split()
            
            # Get patch properties
            lonc   = float(items[0])
            latc   = float(items[1])
            zc     = float(items[4])
            strike = float(items[5])
            dip    = float(items[6])
            area   = float(items[7])
            PID    = int(items[8])
            #if strike<0.:
            #    strike += 360.
            length = np.sqrt(area)
            width  = np.sqrt(area)
            
            xc,yc = self.ll2xy(lonc,latc)
            
            # Build a patch with that
            strike_rad = strike*np.pi/180.
            dip_rad    = dip*np.pi/180.
            dstrike_x  =  0.5 * length * np.sin(strike_rad)
            dstrike_y  =  0.5 * length * np.cos(strike_rad)
            ddip_x     =  0.5 * width  * np.cos(dip_rad) * np.cos(strike_rad)
            ddip_y     = -0.5 * width  * np.cos(dip_rad) * np.sin(strike_rad)
            ddip_z     =  0.5 * width  * np.sin(dip_rad)    

            x1 = xc - dstrike_x - ddip_x
            y1 = yc - dstrike_y - ddip_y
            z1 = zc - ddip_z
            
            x2 = xc + dstrike_x - ddip_x
            y2 = yc + dstrike_y - ddip_y
            z2 = zc - ddip_z            

            x3 = xc + dstrike_x + ddip_x
            y3 = yc + dstrike_y + ddip_y
            z3 = zc + ddip_z

            x4 = xc - dstrike_x + ddip_x
            y4 = yc - dstrike_y + ddip_y
            z4 = zc + ddip_z     
            
            if self.top == None:
                self.top = z2
            elif self.top > z2:
                self.top = z2
            if self.depth == None:
                self.depth = z1
            elif self.depth > z1:
                self.depth = z2

            # Convert to lat lon
            lon1, lat1 = self.xy2ll(x1, y1)
            lon2, lat2 = self.xy2ll(x2, y2)
            lon3, lat3 = self.xy2ll(x3, y3)
            lon4, lat4 = self.xy2ll(x4, y4)

            # Fill the patch
            p = np.zeros((4, 3))
            pll = np.zeros((4, 3))
            p[0,:] = [x1, y1, z1]
            p[1,:] = [x2, y2, z2]
            p[2,:] = [x3, y3, z3]
            p[3,:] = [x4, y4, z4]
            pll[0,:] = [lon1, lat1, z1]
            pll[1,:] = [lon2, lat2, z2]
            pll[2,:] = [lon3, lat3, z3]
            pll[3,:] = [lon4, lat4, z4]
            p1,p2,p3,p4 = p
            self.patch.append(p)
            self.patchll.append(pll)            
            self.z_patches.append(z1)            
            
            
        # Close the files
        flld.close()
        self.equivpatch   = self.patch
        self.equivpatchll = self.patchll
        # All done
        return    
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    def getHypoToCenter(self, p, ds_dist=False):
        ''' 
        Get patch center coordinates from hypocenter

        Args:
            * p      : Patch number.

        Kwargs:
            * ds_dist: If true, will return along dip (first) and along strike distances

        Returns:
            * Hypocenter coordinates
        '''

        # Check strike/dip/hypo assigmement
        assert self.hypo_x   != None, 'Hypocenter   must be assigned'
        assert self.hypo_y   != None, 'Hypocenter   must be assigned'
        assert self.hypo_z   != None, 'Hypocenter   must be assigned'

        # Get center
        p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)

        # Along dip and along strike distance to hypocenter
        if ds_dist:
            assert self.hypo_patch_index is not None, 'Must provide a hypocenter patch index'
            assert self.fault_map        is not None, 'Must provide a fault map'

            hp_x, hp_y, hp_z, hp_W, hp_L, hp_S, hp_D = self.getpatchgeometry(self.hypo_patch_index,center=True)

            assert np.round(p_width,2)  == np.round(hp_W,2), 'Patch width  must be homogeneous over the fault'
            assert np.round(p_length,2) == np.round(hp_L,2), 'Patch length must be homogeneous over the fault'            

            hp_strike,hp_dip = self.fault_map[self.hypo_patch_index]
            p_strike ,p_dip  = self.fault_map[p]
            
            strike_d = (p_strike - hp_strike) * p_length
            dip_d    = (p_dip    - hp_dip   ) * p_width
            
            dip_d    -= (self.hypo_z-hp_z) / np.sin(hp_D)            
            strike_d -= (self.hypo_x-hp_x) * np.sin(hp_S) + (self.hypo_y-hp_y) * np.cos(hp_S)

            return dip_d, strike_d
        else:
            x = p_x - self.hypo_x
            y = p_y - self.hypo_y
            z = p_z - self.hypo_z
            return x,y,z
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    def setFaultMap(self,Nstrike,Ndip,leading='strike',check_depth=True):
        '''
        Set along dip and along strike indexing for patches

        Args:
            * Nstrike       : number of patches along strike
            * Ndip          : number of patches along dip

        Kwargs:
            * leading       : leadinf index of self.patch (can be 'strike' or 'dip')
            * check_depth   : CHeck patch depths and indexes are consistent

        Returns:
            * None
        '''

        # Check input parameters
        if leading=='strike':
            Nx=Nstrike
            Ny=Ndip
        else:
            Nx=Ndip
            Ny=Nstrike
        assert Nx*Ny==len(self.patch), 'Incorrect Nstrike and Ndip'
        
        # Loop over patches
        self.fault_map = []
        self.fault_inv_map = np.zeros((Nstrike,Ndip),dtype='int')
        for ny in range(Ny):
            for nx in range(Nx):
                p = ny * Nx + nx
                if leading=='strike':
                    self.fault_map.append([nx,ny])
                    self.fault_inv_map[nx,ny] = p
                elif leading=='dip':
                    self.fault_map.append([ny,nx])
                    self.fault_inv_map[ny,nx] = p
        self.fault_map = np.array(self.fault_map)
        
        for n in range(Ndip):
            i = np.where(self.fault_map[:,1]==n)[0]
            assert len(i)==Nstrike, 'Mapping error'

        for n in range(Nstrike):
            i = np.where(self.fault_map[:,0]==n)[0]
            assert len(i)==Ndip, 'Mapping error'

        if check_depth:
            for n in range(Ndip):
                indexes = np.where(self.fault_map[:,1]==n)[0]
                flag = True
                for i in indexes:
                    x,y,z = self.getcenter(self.patch[i])
                    if flag:
                        depth = np.round(z,1)
                        flag  = False
                    assert depth==np.round(z,1), 'Mapping error: inconsistent depth'

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def initializekinmodel(self, n=None):
        '''
        Re-initializes the fault slip array to zero values.

        Kwargs:
            * n     : Number of slip values. If None, it'll take the number of patches.

        Returns:
            * None
        '''
        self.initializeslip(n=n)
        self.tr = np.zeros((self.N_slip,))
        self.vr = np.zeros((self.N_slip,))
        
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildSubGrid(self,nbp_strike,nbp_dip):
        '''
        Define a subgrid of point sources on the fault (multiple point src per patches)
        All patches must have the same size

        Args: 
            * p_nstrike:   Number of subgrid points per patch along strike 
            * p_ndip:      Number of subgrid points per patch along dip            

        Returns:
            * None
        '''
        
        # Init Grid size
        grid_size_strike = None
        grid_size_dip    = None
        
        # Loop over patches        
        self.grid = []
        for p in range(len(self.patch)):
            # Get patch location/size
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)
            
            # Dip direction
            dipdir = (p_strike+np.pi/2.)%(2.*np.pi)

            # grid-size
            if grid_size_strike==None:
                grid_size_strike = p_length/nbp_strike
            else:
                dum = p_length/nbp_strike
                errmsg = 'Heteogeneous grid size not implemented (%f,%f)'%(grid_size_strike,dum)
                assert np.round(grid_size_strike,2) == np.round(dum,2), errmsg

            if grid_size_dip==None:
                grid_size_dip = p_length/nbp_dip
            else:
                errmsg = 'Heteogeneous grid size not implemented (dip)'
                assert np.round(grid_size_dip,2) ==np.round(p_length/nbp_dip,2), errmsg

            # Set grid points coordinates on fault
            grid_strike = np.arange(0.5*grid_size_strike,p_length,grid_size_strike) - p_length/2.
            grid_dip    = np.arange(0.5*grid_size_dip   ,p_width ,grid_size_dip   ) - p_width/2.

            # Check that everything is correct
            assert nbp_strike == len(grid_strike), 'Incorrect length for patch %d'%(p)
            assert nbp_dip    == len(grid_dip),    'Incorrect width for patch  %d'%(p)

            # Get grid points coordinates in UTM  
            xt = p_x + grid_strike * np.sin(p_strike)
            yt = p_y + grid_strike * np.cos(p_strike)
            zt = p_z * np.ones(xt.shape)
            g  = []
            for i in range(nbp_dip):
                x = xt + grid_dip[i] * np.cos(p_dip) * np.sin(dipdir)
                y = yt + grid_dip[i] * np.cos(p_dip) * np.cos(dipdir)
                z = zt + grid_dip[i] * np.sin(p_dip)
                for j in range(x.size):                    
                    g.append([x[j],y[j],z[j]])
            self.grid.append(g)
                
        # All done
        return 
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setMu(self,model_file,modelformat='CPS'):
        '''
        Set shear modulus values for seismic moment calculation
        from model_file:

        +------------------+--------------------------+
        | if format = 'CPS'|Thickness, Vp, Vs, Rho    |
        +==================+==========================+
        | if format = 'KK' |file from Kikuchi Kanamori|
        +------------------+--------------------------+

        Args:
            * model_file        : Input file

        Kwargs:
            * modelformat       : Format of the model file

        Returns:
            * None
        '''

        # Check modelformat
        assert modelformat == 'CPS' or modelformat == 'KK', 'Incorrect model format (CPS or KK)'
        
        # Read model file
        mu = []
        depth  = 0.
        depths = []
        with open(model_file) as f:
            if modelformat == 'CPS':
                for l in f:
                    if l.strip()[0]=='#':
                        continue
                    items = l.strip().split()
                    H   = float(items[0])
                    VS  = float(items[2])
                    RHO = float(items[3])
                    mu.append(VS*VS*RHO*1.0e9)
                    if H==0.:
                        H = np.inf
                    depths.append([depth,depth+H])
                    depth += H
            elif modelformat == 'KK':
                vmodelname = f.readline().strip()
                print('Reading model '+vmodelname)
                items = f.readline().strip().split()
                N = int(items[2]) # Number of layers
                for i in range(N):
                    VS  = float(items[-3])
                    RHO = float(items[-2])
                    H   = float(items[-1])
                    mu.append(VS*VS*RHO*1.0e9)
                    if H == 0 or H==99.0:
                        H = np.inf
                    depths.append([depth,depth+H])
                    items = f.readline().strip().split()
                    depth += H
            else:
                sys.write('Incorrect model format')
                sys.exit(1)
        Nd = len(depths)
        Np = len(self.patch)        
        # Set Mu for each patch
        self.mu = np.zeros((Np,))
        for p in range(Np):
            p_x, p_y, p_z, width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)
            for d in range(Nd):
                if p_z>=depths[d][0] and p_z<depths[d][1]:
                    self.mu[p] = mu[d]

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setMuLambdaRho(self,model_file):
        '''
        Set shear modulus values for seismic moment calculation
        from model_file: Thickness  Vp  Vs  Rho (...)

        Args:
            * model_file    : Input file name

        Retunrs:
            * None
        '''
        
        # Read model file
        mu = []
        la = []
        rho = []
        depth  = 0.
        depths = []
        with open(model_file) as f:
            for l in f:
                if l.strip()[0]=='#':
                    continue
                items = l.strip().split()
                H   = float(items[0])
                VP  = float(items[1])
                VS  = float(items[2])
                RHO = float(items[3])
                mu.append(VS*VS*RHO*1.0e9)
                la.append(VP*VP*RHO*1.0e9 - 2*mu[-1])
                rho.append(RHO*1.0e3)
                if H==0.:
                    H = np.inf
                depths.append([depth,depth+H])
                depth += H
        Nd = len(depths)
        Np = len(self.patch)
        
        # Set Mu for each patch
        self.mu = np.zeros((Np,))
        self.la = np.zeros((Np,))
        self.rho= np.zeros((Np,))
        for p in range(Np):
            p_x, p_y, p_z, width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)
            for d in range(Nd):
                if p_z>=depths[d][0] and p_z<depths[d][1]:
                    self.mu[p] = mu[d]
                    self.la[p] = la[d]
                    self.rho[p] = rho[d]

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildKK(self,data,rakes=[0.,90.],Mu=None, slip=1.0, 
                          coord0=None, causal=False, filter=True):
        '''
        Build Kikuchi-Kanamori Green's functions

        Args:
            * data: teleseismic data object

        Kwargs:
            * rake: Rake angles (default [0.,90.])          
            * Mu: Shear modulus (optional)
            * slip: Slip amplitude for tsunami GF calculation (default: 1. m)
            * coord0: lon,lat,dep of reference point to shift GFs (optional)
            * causal: if True impose causality of the source (no slip before time=0.)
            * filter: if True, filter the Green's functions (according to i_master parameters in the KK run directory)

        Returns:
            * None
        '''

        print ("Building Green's functions for the data set {} of type {}".format(data.name, data.dtype))

        # Check the patch attribute
        assert self.patch != None, 'Patch object should be assigned'        

        # Check the waveform engine
        assert data.waveform_engine is not None, 'KK Waveform engine not initiated (see seismic.waveKK)'
        wave_engine = data.waveform_engine
        
        # Check Mu
        Np = len(self.patch)
        if Mu!=None:
            self.mu = np.ones((Np,)) * Mu
        else:
            assert self.mu is not None, 'Shear modulus must be assigned (use self.setMu)'

        # Set M0 for all patches
        M0 = []
        for p in range(Np):       
            p_x, p_y, p_z, width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)
            M0.append(self.mu[p] * slip * width * length * 1.0e6)
            
        # List all fault parameters
        lon = []
        lat = []
        dep = []
        strike = []
        dip    = []
        rake   = []
        for p_rake in rakes:
            for p in range(Np):            
                p_x, p_y, p_z, width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)
                p_strike = np.round(strike_rad*180./np.pi,5)
                p_dip    = np.round(dip_rad*180./np.pi,5)
                p_z      = np.round(p_z,5)
                p_lon,p_lat = self.xy2ll(p_x,p_y)
                lon.append(p_lon)
                lat.append(p_lat)
                dep.append(p_z)
                strike.append(p_strike)
                dip.append(p_dip)
                rake.append(p_rake)
        lon = np.array(lon)
        lat = np.array(lat)
        dep = np.array(dep)
        strike = np.array(strike)
        dip    = np.array(dip)
        rake   = np.array(rake)
        
        # Build Green's function database
        fault_params = np.array([dep,strike,dip,rake]).T
        o = np.vstack({tuple(row) for row in fault_params})
        udep    = o[:,0]
        ustrike = o[:,1]
        udip    = o[:,2]
        urake   = o[:,3]
        
        wave_engine.computeGFdb(udep,ustrike,udip,urake,filter=filter)

        # Compute Green's functions from GF database
        if coord0 is not None:
            wave_engine.computeGF(coord0[0],coord0[1],coord0[2],lon,lat,dep,strike,dip,rake,causal=causal)
        else:
            wave_engine.computeGF(self.hypo_lon,self.hypo_lat,self.hypo_z,lon,lat,dep,strike,dip,rake,causal=causal)
        #wave_engine.computeGF(-79.940,0.371,19.000,lon,lat,dep,strike,dip,rake)

        # Assign Green's functions dictionary
        self.G[data.name] = {}
        j = 0
        for p_rake in rakes:
            self.G[data.name][p_rake] = []
            for p in range(Np):
                self.G[data.name][p_rake].append({})
                for dkey in wave_engine.GF[j]:
                    self.G[data.name][p_rake][p][dkey] = wave_engine.GF[j][dkey].copy()
                    self.G[data.name][p_rake][p][dkey].depvar *= M0[p]
                j += 1
                
        # All done
        return                        
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    def buildKinGFsFromDB(self, data, wave_engine, slip, rakes, 
                                rake_key=None, Mu=None, filter_coef=None, 
                                differentiate=False):
        '''
        Build Kinematic Green's functions based on the discretized fault and 
        a pre-calculated GF database. Green's functions will be calculated for a given shear modulus and a 
        given slip (cf., slip) along a given rake angle (cf., rake)

        Args:
            * data          : Seismic data object
            * wave_engine   : waveform generator
            * slip          : slip amplitude (in m)

        Kwargs:
            * rakes         : rake angle (in deg). Can be a scalar or an array of len(self.patch)
            * rake_key      : By default, GFs are stored in a dictionnarry 
            * Mu            : Shear modulus (optional)
            * filter_coef   : Array or dictionnary of second-order filter coefficients (optional), see scipy.signal.sosfilt

        Returns:
            * None
        '''        
        
        print ("Building Green's functions for the data set {} of type {}"\
                .format(data.name, data.dtype))
        print ("Using GF_path: {}".format(wave_engine.GF_path))        

        # Check the patch attribute
        assert self.patch != None, 'Patch object should be assigned'

        # Check rakes
        if type(rakes) is list:
            rakes = np.array(rakes)
        if type(rakes) is np.ndarray:
            assert len(rakes) == len(self.patch), 'rakes must be a scalar or an array of length {}'.format(len(self.patch))
            assert rake_key is not None, 'You must provide a keyword for this GFs (ex. AlongRake, RakePerp, etc...'
        else: # If scalar rake
            rake = rakes 
            rake_key = rake # Set dictionnary keyword to rake value

        # Check Mu
        Np = len(self.patch)
        if Mu!=None:
            self.mu = np.ones((Np,)) * Mu
        else:
            assert self.mu is not None

        # Init Green's functions
        if data.name not in self.G:
            self.G[data.name] = {}
        self.G[data.name][rake_key] = []
        
        # Init station lat/lon
        assert len(data.lat)>0, 'Station lat must be assigned'
        assert len(data.lon)>0, 'Station lon must be assigned'
        assert len(data.lon)==len(data.lat), 'Inconsistent station lat/lon'
        assert len(data.sta_name)==len(data.lat), 'Inconsistent station name/lat/lon'
        Ns = len(data.sta_name)
        s_name = data.sta_name
        s_lat  = data.lat
        s_lon  = data.lon

        # Get delta
        delta = data.d[data.sta_name[0]].delta
        
        # Loop over each patch
        G = self.G[data.name][rake_key]
        for p in range(Np):
            
            # Get rake of that patch
            if type(rakes) is np.ndarray:
                rake = rakes[p]

            # Get point source location and patch geometry
            p_x, p_y, p_z, width, length, strike_rad, dip_rad = self.getpatchgeometry(p,center=True)  
            p_lon,p_lat = self.xy2ll(p_x,p_y)
            strike = strike_rad*180./np.pi
            dip    = dip_rad*180./np.pi

            # Seismic moment
            M0 = self.mu[p] * slip * width * length * 1.0e13 # M0  
            
            # Compute GFs for each station
            synth = {}
            for s in range(Ns):
                # Get station name and component
                dkey = data.sta_name[s]
                ori  = data.d[dkey].kcmpnm[2]
                # Station Azimuth and distance                
                [az,baz,dist] = self.geod.inv(p_lon,p_lat,s_lon[s],s_lat[s])
                dist /= 1000. # km -> m                
                # Compute synthetics
                #print(dkey,dist)
                o_sac,L_sac,T_sac = wave_engine.synthSDR(p_z,az,dist,M0,strike,dip,rake)
                if ( ori == 'N' or ori == 'E' or ori == '1' or ori == '2' ): 
                    assert (data.d[dkey].cmpaz>=-360 and data.d[dkey].cmpaz<=360.), '{} cmpaz must be within [-360,360]'.format(dkey)                   
                    o_sac = wave_engine.rotTraces(L_sac,T_sac,baz,data.d[dkey].cmpaz)
                # Check delta
                assert np.round(data.d[dkey].delta,4) == np.round(delta,4), 'Sampling frequency must be identical for each station'
                assert np.round(o_sac.delta,4) == np.round(delta,4),        'Sampling frequency must be identical for each GFs'
                # Differentiate
                if differentiate:
                    o_sac.depvar = np.diff(o_sac.depvar)/delta
                    o_sac.b    += 0.5*delta
                    o_sac.npts -= 1
                # GFs filtering
                if filter_coef is not None:
                    if filter_coef.__class__ is dict:
                        sos = filter_coef[dkey]
                    else:
                        sos = filter_coef
                    o_sac.depvar = signal.sosfilt(sos,o_sac.depvar)
                # GFs time-windowing
                b = data.d[dkey].b - data.d[dkey].o
                npts = data.d[dkey].npts
                t = np.arange(o_sac.npts)*o_sac.delta+o_sac.b-o_sac.o
                dtb = np.absolute(t-b)
                ib  = np.where(dtb==dtb.min())[0][0]
                assert np.absolute(dtb[ib])<o_sac.delta,'Incomplete GFs'                
                o_sac.depvar = o_sac.depvar[ib:ib+npts]
                # Sac headers
                o_sac.kstnm  = data.d[dkey].kstnm
                o_sac.kcmpnm = data.d[dkey].kcmpnm
                o_sac.knetwk = data.d[dkey].knetwk
                o_sac.khole  = data.d[dkey].khole
                o_sac.stlo   = data.d[dkey].stlo
                o_sac.stla   = data.d[dkey].stla
                o_sac.npts   = npts
                o_sac.b      = t[ib]+o_sac.o
                #if p==91:
                #    o_sac.write('bidon/'+dkey+'_gf%d'%(rake))
                # Assemble GFs                
                synth[s_name[s]] = o_sac.copy()
            G.append(copy.deepcopy(synth))

        # All done
        return        
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildBigCd(self,seismic_data):
        '''
        Assemble Cd from multiple kinematic datasets

        Args:
            * seismic_data      : Data to take care of

        Returns:
            * None
        '''
        assert self.bigD is not None, 'bigD must be assigned'
        assert self.bigD_map is not None, 'bigD_map must be assigned (use setbigDmap)'
        self.bigCd = np.zeros((self.bigD.size,self.bigD.size))

        if type(seismic_data) != list:
            data_list = [seismic_data]
        else:
            data_list = seismic_data
            
        for data in data_list:
            i = self.bigD_map[data.name]
            self.bigCd[i[0]:i[1],i[0]:i[1]] = data.Cd
        # All done return
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def saveBigCd(self, bigCdfile = 'kinematicG.Cd', dtype='np.float64'):
        '''
        Save bigCd matrix

        Kwargs:
            * bigCdfile     : Output filename
            * dtype         : binary type for output

        Returns:    
            * None
        '''

        # Check if Cd exists
        assert self.bigCd is not None, 'bigCd must be assigned'
        
        # Convert Cd to dtype
        Cd = self.bigCd.astype(dtype)

        # Write t file
        Cd.tofile(bigCdfile)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    def setBigDmap(self,seismic_data):
        '''
        Assign data_idx map for kinematic data

        Args:
            * seismic_data      : Data to take care of

        Returns:
            * None
        '''
        if type(seismic_data) != list:
            data_list = [seismic_data]
        else:
            data_list = seismic_data
            
        # Set the data index map
        d1 = 0
        d2 = 0        
        self.bigD_map = {}
        for data in data_list:
            for dkey in data.sta_name:
                d2 += data.d[dkey].npts
            self.bigD_map[data.name]=[d1,d2]
            d1 = d2
        # All done
        return
    # ----------------------------------------------------------------------
        
    # ----------------------------------------------------------------------
    def buildBigGD(self,eik_solver,seismic_data,rakes,vmax,Nt,Dt,
                        rakes_key=None,dtype='np.float64',
                        fastsweep=False,indexing='Altar'):
        '''
        Build BigG and bigD matrices from Green's functions and data dictionaries

        Args:
            * eik_solver: Eikonal solver (e.g., FastSweep or None)
            * data:       Seismic data object or list of objects
            * rakes:      List of rake angles
            * vmax:       Maximum rupture velocity
            * Nt:         Number of rupture time-steps
            * Dt:         Rupture time-steps

        Kwargs:
            * rakes_key:  If GFs are stored under different keywords than rake value, provide them here
            * fastsweep:  If True and vmax is set, solves min arrival time using fastsweep algo. If false, uses analytical solution.

        Returns:
            * tmin:       Array of ???
        '''

        if type(seismic_data) != list:
            data_list = [seismic_data]
        else:
            data_list = seismic_data
           
        # set rake keywords for dictionnary
        if rakes_key is None:
            rakes_key = rakes
         
        # Set eikonal solver grid for vmax
        Np = len(self.patch)
        if vmax != np.inf and vmax > 0.:
            vr = copy.deepcopy(self.vr)
            self.vr[:] = vmax 
            if fastsweep and (eik_solver is not None): # Uses fastsweep
                eik_solver.setGridFromFault(self,1.0)
                eik_solver.fastSweep()
                self.vr[:] = copy.deepcopy(vr)
        
                # Get tmin for each patch
                tmin = []
                for p in range(Np):
                    # Location at the patch center
                    dip_c, strike_c = self.getHypoToCenter(p,True)
                    tmin.append(eik_solver.getT0([dip_c],[strike_c])[0])

            else: # Uses analytical solution
                # Get tmin for each patch
                tmin = []
                for p in range(Np):
                    dip_c, strike_c = self.getHypoToCenter(p,True)
                    tmin.append(np.sqrt(dip_c**2+strike_c**2)/vmax)                    
        else:
            tmin = np.zeros((Np,))

        
        # Build up bigD
        self.bigD = []
        for data in data_list:
            for dkey in data.sta_name:
                self.bigD.extend(data.d[dkey].depvar)
        self.bigD = np.array(self.bigD)
        
        # Build Big G matrix
        self.bigG = np.zeros((len(self.bigD),Nt*Np*len(rakes_key)))
        j  = 0
        if indexing == 'Altar':
            for nt in range(Nt):
                #print('Processing %d'%(nt))
                for r in rakes_key:
                    for p in range(Np):                    
                        di = 0
                        for data in data_list:
                            for dkey in data.sta_name:
                                depvar = self.G[data.name][r][p][dkey].depvar
                                npts   = self.G[data.name][r][p][dkey].npts
                                delta  = self.G[data.name][r][p][dkey].delta
                                its = int(np.round((tmin[p] + nt * Dt)/delta,0)) 
                                i = its + di     
                                l = npts - its
                                if l>0:
                                    self.bigG[i:i+l,j] = depvar[:l]                        
                                di += npts
                        j += 1
        else:
            for r in rakes_key:
                for p in range(Np):                    
                    di = 0
                    for nt in range(Nt):                
                        for data in data_list:
                            for dkey in data.sta_name:
                                depvar = self.G[data.name][r][p][dkey].depvar
                                npts   = self.G[data.name][r][p][dkey].npts
                                delta  = self.G[data.name][r][p][dkey].delta
                                its = int(np.round((tmin[p] + nt * Dt)/delta,0)) 
                                i = its + di     
                                l = npts - its
                                if l>0:
                                    self.bigG[i:i+l,j] = depvar[:l]                        
                                di += npts
                        j += 1                                    

        # All done
        return tmin
    # ----------------------------------------------------------------------
            
    # ----------------------------------------------------------------------
    def saveBigGD(self, bigDfile='kinematicG.data', bigGfile='kinematicG.gf', 
                        dtype='np.float64'):
        '''
        Save bigG and bigD to binary file

        Kwargs:
            * bigDfile  : bigD filename (optional)
            * bigGfile  : bigG gilename (optional)
            * dtype     : Data binary type 

        Returns:
            * None
        '''
        
        # Check bigG and bigD
        assert self.bigD is not None or self.bigG is not None
        assert bigDfile is not None or bigGfile is not None

        # Write files
        if bigDfile != None:
            self.bigD.astype(dtype).tofile(bigDfile)
        if bigGfile != None:
            self.bigG.astype(dtype).T.tofile(bigGfile)
        
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def loadBigGD(self, bigDfile='data.kin', bigGfile='gf.kin', dtype='np.float64'):
        '''
        Load bigG and bigD to binary file

        Kwargs:
            * bigDfile  : bigD filename (optional)
            * bigGfile  : bigG gilename (optional)
            * dtype     : data binary type
        '''
        
        # Check file names
        assert bigDfile  != None or bigGfile  != None
        
        # Load bigG and/or bigD files and convert them to double precision
        if bigDfile != None:
            self.bigD = np.fromfile(bigDfile, dtype=dtype).astype('np.float64')

        if bigGfile is not None:
            assert self.bigD is not None
            Nd = self.bigD.size
            self.bigG = np.fromfile(bigGfile, dtype=dtype).astype('np.float64')
            assert self.bigG.size%Nd == 0
            Nm = int(self.bigG.size/Nd)
            # Reshape bigG matrix
            self.bigG = self.bigG.reshape(Nm,Nd).T
        
        # All done
        return  
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def saveKinGF(self, data, outputDir='GFs', prefix='gf', rmdir=True):
        '''
        Save kinematic Green's functions in outputDir

        Args:
            * data        : seismic data object

        Kwargs:
            * outputDir   : output directory where GFs will be stored
            * prefix      : Prefix for GFs files
            * rmdir       : CLeanup?

        Returns:
            * None
        '''
           
        # Print stuff
        print('Writing Kinematic GFs to directory {} for fault {}'.format(outputDir,self.name))

        # Check i_path
        assert os.path.exists(outputDir), '%s: No such directory'%(i_path)
        
        # Main loop
        G = self.G[data.name]
        Np = len(self.patch)
        for r in G.keys():
            o_dir = os.path.join(outputDir,'rake_{}'.format(r))
            if os.path.exists(o_dir) and rmdir:
                sh.rmtree(o_dir)
            if not os.path.exists(o_dir):
                os.mkdir(o_dir)
            for p in range(Np):
                for dkey in data.sta_name:                    
                    o_file = os.path.join(o_dir,'%s_p%d_%s.kin'%(prefix,p,dkey))
                    self.G[data.name][r][p][dkey].write(o_file)
    
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def loadKinGF(self, data, rakes, inputDir = 'GFs', prefix='gf'):
        '''
        Load kinematic Green's functions in i_path

        Args:
            * data        : seismic data object
            * rakes       : list of rake angle

        Kwargs:
            * inputDir    : input directory where GFs are stored
            * prefix      : GFs files prefix

        Returns:
            * None
        '''

        # Print stuff
        print('Loading Kinematic GFs from directory {} for fault {}'\
                .format(inputDir,self.name))        

        # Import sacpy 
        import sacpy
        i_sac = sacpy.sac()
        
        # Check the patch attribute
        assert self.patch != None, 'Patch object should be assigned'
        
        # Check inputDir
        assert os.path.exists(inputDir), '%s: No such directory'%(inputDir)
        
        # Init Green's functions
        if data.name not in self.G:
            self.G[data.name] = {}

        # Main loop
        Np = len(self.patch)
        for r in rakes:

            # Check subdirectories
            i_dir = os.path.join(inputDir,'rake_{}'.format(r))
            assert os.path.exists(i_dir), '%s: no such directory'%(i_dir)

            self.G[data.name][r] = []
            for p in range(Np):
                # Read GFs for each station
                synth = {}
                for dkey in data.sta_name:
                    # Read sac
                    i_file = os.path.join(i_dir,'%s_p%d_%s.kin'%(prefix,p,dkey))
                    i_sac.read(i_file)
                    synth[dkey] = i_sac.copy()
                self.G[data.name][r].append(copy.deepcopy(synth))
        
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def castbigM(self,n_ramp_param,eik_solver,npt=4,Dtriangles=1.,grid_space=1.0):
        '''
        Cast kinematic model into bigM for forward modeling using bigG
        (model should be specified in slip, tr and vr attributes, hypocenter 
        must be specified)

        Args:
            * n_ramp_param  : number of nuisance parameters (e.g., InSAR orbits)
            * eik_solver    : eikonal solver

        Kwargs:
            * npt**2        : numper of point sources per patch 
            * Dtriangles    : ??
            * grid_space    : ??

        Returns:
            * bigM matrix
        '''

        print('Casting model into bigM')        
        
        # Eikonal resolution
        eik_solver.setGridFromFault(self,grid_space)
        eik_solver.fastSweep()
        
        # BigG x BigM (on the fly time-domain convolution)
        Np = len(self.patch)  
        Ntriangles = int(self.bigG.shape[1]/(2*Np))
        bigM = np.zeros((self.bigG.shape[1],))
        for p in range(Np):
            # Location at the patch center
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)
            dip_c, strike_c = self.getHypoToCenter(p,True)
            # Grid location
            grid_size_dip = p_length/npt
            grid_size_strike = p_length/npt
            grid_strike = strike_c+np.arange(0.5*grid_size_strike,p_length,grid_size_strike) - p_length/2.
            grid_dip    = dip_c+np.arange(0.5*grid_size_dip   ,p_width ,grid_size_dip   )    - p_width/2.
            time = np.arange(Ntriangles)*Dtriangles#+Dtriangles
            T    = np.zeros(time.shape)
            Tr2  = self.tr[p]/2.            
            for i in range(npt):
                for j in range(npt):
                    t = eik_solver.getT0([grid_dip[i]],[grid_strike[j]])[0]
                    tc = t+Tr2
                    ti = np.where(np.abs(time-tc)<Tr2)[0]            
                    T[ti] += (1/Tr2 - np.abs(time[ti]-tc)/(Tr2*Tr2))*Dtriangles
            for nt in range(Ntriangles):
                bigM[2*nt*Np+p]     = T[nt] * self.slip[p,0]/float(npt*npt)
                bigM[(2*nt+1)*Np+p] = T[nt] * self.slip[p,1]/float(npt*npt)

        # All done 
        return bigM  
    # ----------------------------------------------------------------------

#EOF
