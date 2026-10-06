'''
A class that deals planar kinematic faults

Written by Z. Duputel, January 2014
'''

## Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from scipy.linalg import block_diag
import copy
import sys
import os
import shutil

## Personals
major, minor, micro, release, serial = sys.version_info
if major==2:
    import okada4py as ok

# Rectangular patches Fault class
from .planarfault import planarfault



class planarfaultkinematic(planarfault):

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        '''
        Args:
            * name      : Name of the fault.
            * f_strike: strike angle in degrees (from North)
            * f_dip:    dip angle in degrees (from horizontal)
            * f_length: length of the fault (i.e., along strike)
            * f_width: width of the fault (i.e., along dip)        
            * utmzone   : UTM zone.
        '''
        
        # Parent class init
        super(planarfaultkinematic,self).__init__(name,
                                                  utmzone=utmzone,
                                                  ellps=ellps,
                                                  lon0=lon0,
                                                  lat0=lat0)

        # Hypocenter coordinates
        self.hypo_x   = None
        self.hypo_y   = None
        self.hypo_z   = None
        self.hypo_lon = None
        self.hypo_lat = None
                
        # Fault size
        self.f_length  = None
        self.f_width   = None
        self.f_nstrike = None
        self.f_ndip    = None
        self.f_strike  = None
        self.f_dip     = None
        
        # Patch objects
        self.patch = None
        self.grid  = None
        self.vr    = None
        self.tr    = None
        
        # All done
        return


    def getHypoToCenter(self, p, sd_dist=False):
        ''' 
        Get patch center coordinates from hypocenter
        Args:
            * p      : Patch number.
            * sd_dist: If true, will return along dip and along strike distances
        '''

        # Check strike/dip/hypo assigmement
        assert self.f_strike != None, 'Fault strike must be assigned'
        assert self.f_dip    != None, 'Fault dip    must be assigned'
        assert self.hypo_x   != None, 'Hypocenter   must be assigned'
        assert self.hypo_y   != None, 'Hypocenter   must be assigned'
        assert self.hypo_z   != None, 'Hypocenter   must be assigned'

        # Get center
        p_x, p_y, p_z = self.getcenter(self.patch[p])
        x = p_x - self.hypo_x
        y = p_y - self.hypo_y
        z = p_z - self.hypo_z

        # Along dip and along strike distance to hypocenter
        if sd_dist:
            dip_d = z / np.sin(self.f_dip)
            strike_d = x * np.sin(self.f_strike) + y * np.cos(self.f_strike)
            return dip_d, strike_d
        else:
            return x,y,z
            


    def setHypoXY(self,x,y, UTM=True):
        '''
        Set hypocenter attributes from x,y
        Outputs: East/West UTM/Lon coordinates, depth attributes
        Args:
            * x:   east  coordinates 
            * y:   north coordinates
            * UTM: default=True, x and y is in UTM coordinates (in km)
                   if    ==False x=lon and y=lat (in deg)
        '''
        
        # Check strike/dip assigmement
        assert self.f_strike != None, 'Fault strike must be assigned'
        assert self.f_dip    != None, 'Fault dip    must be assigned'

        # If UTM==False, convert x=lon/y=lat to UTM
        if not UTM:
            self.hypo_x,self.hypo_y = self.ll2xy(x,y)
        else:
            self.hypo_x = x
            self.hypo_y = y

        # Get distance from the fault trace axis (in km)
        dist_from_trace = (self.hypo_x-self.xi[0]) * np.cos(self.f_strike) - (self.hypo_y-self.yi[0]) * np.sin(self.f_strike)

        # Get depth on the fault
        self.hypo_z = dist_from_trace * np.tan(self.f_dip) + self.top
        
        # UTM to lat/lon conversion        
        self.hypo_lon,self.hypo_lat = self.xy2ll(self.hypo_x,self.hypo_y)

        # All done
        return
        
    def buildFault(self, lon, lat, dep, f_strike, f_dip, f_length, f_width, grid_size, p_nstrike, p_ndip):
        '''
        Build fault patches/grid
        Args:
            * lat,lon,dep:  coordinates at the center of the top edge of the fault (in degrees)
            * strike:       strike angle in degrees (from North)
            * dip:          dip angle in degrees (from horizontal)
            * f_length: Fault length, km
            * f_width:  Fault width, km
            * grid_size:    Spacing between point sources within each patch
            * p_nstrike:   Number of subgrid points per patch along strike (multiple pts src per patch)
            * p_ndip:      Number of subgrid points per patch along dip    (multiple pts src per patch)
        '''
        
        # Orientation
        self.f_strike = f_strike * np.pi/180.
        self.f_dip    = f_dip    * np.pi/180.

        # Patch size = nb of pts along dip/strike * spacing
        patch_length  = grid_size * p_nstrike
        patch_width   = grid_size * p_ndip

        # Number of patches along strike / along dip
        self.f_nstrike = int(np.round(f_length/patch_length))
        self.f_ndip    = int(np.round(f_width/patch_width))

        # Correct the fault size to match n_strike and n_dip
        self.f_length = self.f_nstrike * patch_length
        self.f_width  = self.f_ndip    * patch_width
        if self.f_length != f_length or self.f_width != f_width:
            sys.stderr.write('!!! Fault size changed to %.2f x %.2f km'%(self.f_length,self.f_width))

                    
        # build patches
        self.buildPatches(lon, lat, dep, f_strike, f_dip, self.f_length, self.f_width, self.f_nstrike, self.f_ndip)
        
        # build subgrid
        self.buildSubGrid(grid_size,p_nstrike,p_ndip)

        # All done
        return

        
    
    def buildSubGrid(self,grid_size,nbp_strike,nbp_dip):
        '''
        Define a subgrid of point sources on the fault (multiple point src per patches)
        Args: 
            * grid_size:    Spacing between point sources within each patch
            * p_nstrike:   Number of subgrid points per patch along strike 
            * p_ndip:      Number of subgrid points per patch along dip            
        '''
        
        # Check prescribed assigments
        assert self.f_strike != None, 'Fault strike must be assigned'
        assert self.f_dip    != None, 'Fault dip must be assigned'
        assert self.patch    != None, 'Patch objects must be assigned'
        
        dipdir = (self.f_strike+np.pi/2.)%(2.*np.pi)
        
        # Loop over patches
        self.grid = []
        for p in range(len(self.patch)):
            # Get patch location/size
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)

            # Set grid points coordinates on fault
            grid_strike = np.arange(0.5*grid_size,p_length,grid_size) - p_length/2.
            grid_dip    = np.arange(0.5*grid_size,p_width, grid_size) - p_width/2.

            # Check that everything is correct
            assert np.round(p_strike,2) == np.round(self.f_strike,2), 'Fault must be planar' 
            assert np.round(p_dip,2)    == np.round(self.f_dip,2)   , 'Fault must be planar' 
            assert nbp_strike == len(grid_strike), 'Incorrect length for patch %d'%(p)
            assert nbp_dip    == len(grid_dip),    'Incorrect width for patch  %d'%(p)

            # Get grid points coordinates in UTM  
            xt = p_x + grid_strike * np.sin(self.f_strike)
            yt = p_y + grid_strike * np.cos(self.f_strike)
            zt = p_z * np.ones(xt.shape)
            g  = []
            for i in range(nbp_dip):
                x = xt + grid_dip[i] * np.cos(self.f_dip) * np.sin(dipdir)
                y = yt + grid_dip[i] * np.cos(self.f_dip) * np.cos(dipdir)
                z = zt + grid_dip[i] * np.sin(self.f_dip)
                for j in range(x.size):                    
                    g.append([x[j],y[j],z[j]])
            self.grid.append(g)
                
        # All done
        return


    def buildKinGFs(self, data, Mu, rake, slip=1., rise_time=2., stf_type='triangle', 
                    rfile_name=None, out_type='D', verbose=True, ofd=sys.stdout, efd=sys.stderr):
        '''
        Build Kinematic Green's functions based on the discretized fault. Green's functions will be calculated 
        for a given shear modulus and a given slip (cf., slip) along a given rake angle (cf., rake)
        Args:
            * data: Seismic data object
            * Mu:   Shear modulus
            * rake: Rake used to compute Green's functions
            * slip: Slip amplitude used to compute Green's functions (in m)
            * rise_time:  Duration of the STF in each patch
            * stf_type:   Type of STF pulse
            * rfile_name: User specified stf file name if stf_type='rfile'
            * out_type:   'D' for displacement, 'V' for velocity, 'A' for acceleration
            * verbose:    True or False
        '''

        # Check the Waveform Engine
        assert self.patch != None, 'Patch object should be assigned'

        # Verbose on/off        
        if verbose:
            import sys
            print ("Building Green's functions for the data set {} of type {}".format(data.name, data.dtype))
            print ("Using waveform engine: {}".format(data.waveform_engine.name))
        

        # Loop over each patch
        Np = len(self.patch)
        rad2deg = 180./np.pi
        if not self.G.has_key(data.name):
            self.G[data.name] = {}
        self.G[data.name][rake] = []
        G = self.G[data.name][rake]
        for p in range(Np):
            if verbose:
                sys.stdout.write('\r Patch: {} / {} '.format(p+1,Np))
                sys.stdout.flush()  

            # Get point source location and patch geometry
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)
            src_loc = [p_x, p_y, p_z]

            # Angles in degree
            p_strike_deg = p_strike * rad2deg
            p_dip_deg    = p_dip    * rad2deg

            # Seismic moment
            M0 = Mu * slip * p_width * p_length * 1.0e13 # M0 assuming 1m slip
            
            # Compute Green's functions using data waveform engine
            data.calcSynthetics('GF_tmp',p_strike_deg,p_dip_deg,rake,M0,rise_time,stf_type,rfile_name,
                                out_type,src_loc,cleanup=True,ofd=ofd,efd=efd)
        
            # Assemble GFs
            G.append(copy.deepcopy(data.waveform_engine.synth))
        sys.stdout.write('\n')

        # All done
        return

    def buildKinDataTriangleMRF(self, data, eik_solver, Mu, rake_para=0., out_type='D', 
                                verbose=True, ofd=sys.stdout, efd=sys.stderr):
        '''
        Build Kinematic Green's functions based on the discretized fault. Green's functions will be calculated 
        for a given shear modulus and a given slip (cf., slip) along a given rake angle (cf., rake)
        Args:
            * data: Seismic data object
            * eik_solver: eikonal solver
            * Mu:   Shear modulus
            * rake_para: Rake of the slip parallel component in deg (default=0. deg)
            * out_type:   'D' for displacement, 'V' for velocity, 'A' for acceleration (default='D')
            * verbose:    True or False (default=True)

        WARNING: ONLY VALID FOR HOMOGENEOUS RUPTURE VELOCITY

        '''

        # Check the Waveform Engine
        assert self.patch  != None, 'Patch object must be assigned'
        assert self.hypo_x != None, 'Hypocenter location must be assigned'
        assert self.hypo_y != None, 'Hypocenter location must be assigned'
        assert self.hypo_z != None, 'Hypocenter location must be assigned'
        assert self.slip   != None, 'Slip values must be assigned'
        assert self.vr     != None, 'Rupture velocities must be assigned'
        assert self.tr     != None, 'Rise times must be assigned'
        assert len(self.patch)==len(self.slip)==len(self.vr)==len(self.tr), 'Patch attributes must have same length'
        
        # Verbose on/off        
        if verbose:
            import sys
            print ("Building predictions for the data set {} of type {}".format(data.name, data.dtype))
            print ("Using waveform engine: {}".format(data.waveform_engine.name))

        # Max duration
        max_dur = np.sqrt(self.f_length*self.f_length + self.f_width*self.f_width)/np.min(self.vr)
        Nt      = np.ceil(max_dur/data.waveform_engine.delta)

        # Calculate timings using eikonal solver
        print('-- Compute rupture front')
        eik_solver.setGridFromFault(self,0.3)
        eik_solver.fastSweep()

        # Loop over each patch
        print('-- Compute and sum-up synthetics')
        Np = len(self.patch)
        rad2deg = 180./np.pi
        if not self.d.has_key(data.name):
            self.d[data.name] = {}
        D = self.d[data.name]   
        for p in range(Np):
            if verbose:
                sys.stdout.write('\r Patch: {} / {} '.format(p+1,Np))
                sys.stdout.flush()  

            # Get point source location and patch geometry
            p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = self.getpatchgeometry(p,center=True)
            src_loc = [p_x, p_y, p_z] 

            # Angles in degree
            p_strike_deg = p_strike * rad2deg
            p_dip_deg    = p_dip    * rad2deg
            
            # Total slip
            s_para = self.slip[p][0]
            s_perp = self.slip[p][1]
            total_slip = np.sqrt(s_para*s_para + s_perp*s_perp)

            # Compute Rake
            rake = rake_para + np.arctan2(s_perp,s_para)*rad2deg
            
            # Seismic moment
            M0  = Mu * total_slip * p_width * p_length * 1.0e13 # M0 assuming 1m slip
            
            # Moment rate function
            rfile = 'rfile.p%03d'%(p)
            MRF = np.zeros((Nt,),dtype='np.float64')
            t   = np.arange(Nt,dtype='np.float64')*data.waveform_engine.delta
            hTr = 0.5 * self.tr[p]
            for g in range(len(self.grid[p])):
                g_t0 = eik_solver.getT0FromFault(self,self.grid[p][g][0],self.grid[p][g][1],
                                                 self.grid[p][g][2])
                g_tc = g_t0 + hTr
                g_t1 = g_t0 + 2*hTr
                g_i  = np.where((t>=g_t0)*(t<=g_t1))
                MRF[g_i] += (1.0 - np.abs(t[g_i]-g_tc)/hTr)*(1.0/hTr)/len(self.grid[p])
            data.waveform_engine.writeRfile(rfile,MRF)
            rfile = os.path.abspath(rfile)
            # Compute Green's functions using data waveform engine
            data.calcSynthetics('GF_tmp',p_strike_deg,p_dip_deg,rake,M0,None,'rfile',rfile,
                                out_type,src_loc,cleanup=True,ofd=ofd,efd=efd)
                        
            
            # Assemble GFs
            for stat in data.sta_name:
                if not D.has_key(stat):
                    D[stat] = copy.deepcopy(data.waveform_engine.synth[stat])
                else:
                    for c in data.waveform_engine.synth[stat].keys():
                        D[stat][c].depvar += data.waveform_engine.synth[stat][c].depvar
        sys.stdout.write('\n')
        print('-- Done')
        
        # All done
        return

    def creaWav(self,data,include_G=True,include_d=True):
        '''
        Create a list of Waveform dictionaries
        Args:
            * data: Data object 
            * include_G: if True, include G (default=True)
            * include_d: if True, include d (default=True)
        '''
        # Create a list of waveform dictionaries
        Wav = []
        if include_G==True:
            assert self.G.has_key(data.name), 'G must be implemented for {}'.format(data.name)
            for r in self.G[data.name].keys():
                for p in range(len(self.patch)):
                    Wav.append(self.G[data.name][r][p])
        if include_d==True:
            assert self.d.has_key(data.name), 'd must be implemented for {}'.format(data.name)
            Wav.append(self.d[data.name])
        
        # All done
        return Wav

    def trim(self,data,mint,maxt,trim_G=True,trim_d=True):
        '''
        Waveform windowing
        Args:
            * data: Data object 
            * mint: Minimum time
            * maxt: Maximum time
            * trim_G: if True, trim G (default=True)
            * trim_d: if True, trim d (default=True)
        '''

        # Create waveform dictionary list
        Wav = self.creaWav(data,include_G=trim_G,include_d=trim_d)

        # Trim waveforms
        for w in Wav:
            for s in data.sta_name:
                for c in w[s].keys():
                    t = np.arange(w[s][c].npts,dtype='np.float64') * w[s][c].delta + w[s][c].o + w[s][c].b
                    ta = np.abs(t-mint)
                    tb = np.abs(t-maxt)
                    ita = np.where(ta==ta.min())[0][0]
                    itb = np.where(tb==tb.min())[0][0]
                    w[s][c].b      = t[ita]- w[s][c].o
                    w[s][c].depvar = w[s][c].depvar[ita:itb+1]
                    w[s][c].npts   = len(w[s][c].depvar)

        # All done
        return

    def filter(self,data,a,b,filtFunc,mean_npts=None,filter_G=True,filter_d=True):
        '''
        Waveform filtering
        Args:
            * data: Data object 
            * a: numerator polynomial of the IIR filter
            * b: denominator polynomial of the IIR filter
            * filtFunc: filter function
            * mean_npts: remove mean over the leading mean_npts points (default=None)
            * filter_G: if True, filter G (default=True)
            * filter_d: if True, filter d (default=True)        
        '''
        # Create waveform dictionary list
        Wav = self.creaWav(data,include_G=filter_G,include_d=filter_d)

        # Trim waveforms
        for w in Wav:
            for s in data.sta_name:
                for c in w[s].keys():
                    if mean_npts != None:
                        w[s][c].depvar -= np.mean(w[s][c].depvar[:mean_npts])
                    w[s][c].depvar = filtFunc(b,a,w[s][c].depvar)

        # All done
        return        
        

    def saveKinGFs(self, data, o_dir='gf_kin'):
        '''
        Writing Green's functions (1 sac file per channel per patch for each rake)
        Args:
            data  : Data object corresponding to the Green's function to be saved
            o_dir : Output directory name
        '''
        
        # Print stuff
        print('Writing Kinematic Greens functions in directory {} for fault {} and dataset {}'.format(o_dir,self.name,data.name))

        # Write Green's functions
        G = self.G[data.name]
        for r in G: # Slip direction: Rake (integer)
            for p in range(len(self.patch)): # Patch number (integer)
                for s in G[r][p]: # station name (string)
                    for c in G[r][p][s]: # component name (string)
                        o_file = os.path.join(o_dir,'gf_rake%d_patch%d_%s_%s.sac'%(r,p,s,c))
                        G[r][p][s][c].write(o_file)
        
        # Write list of stations
        f = open(os.path.join(o_dir,'stat_list'),'w')
        for s in G[r][p]:
            f.write('%s\n'%(s))
        f.close()

        # All done
        return

    def loadKinGFs(self, data, rake=[0,90],i_dir='gf_kin',station_file=None):
        '''
        Reading Green's functions (1 sac file per channel per patch for each rake)
        Args:
            data       : Data object corresponding to the Green's function to be loaded
            rake       : List of rake values (default=0 and 90 deg)
            i_dir      : Output directory name (default='gf_kin')
            station_file: read station list from 'station_file'
        '''
        
        # Import sac for python (ask Zach)
        import sacpy

        # Print stuff
        print('Loading Kinematic Greens functions from directory {} for fault {} and dataset {}'.format(i_dir,self.name,data.name))

        # Init G
        self.G[data.name] = {}
        G = self.G[data.name]
        
        # Read list of station names
        if station_file != None:
            sta_name = []
            f = open(os.path.join(o_dir,'stat_list'),'r')
            for l in f:
                sta_name.append(l.strip().split()[0])
            f.close()
        else:
            sta_name = data.sta_name

        # Read Green's functions
        for r in rake: # Slip direction: Rake (integer)
            G[r] = []
            for p in range(len(self.patch)): # Patch number (integer)
                G[r].append({})
                for s in sta_name: # station name (string)
                    G[r][p][s] = {}
                    for c in ['Z','N','E']: # component name (string)
                        i_file = os.path.join(i_dir,'gf_rake%d_patch%d_%s_%s.sac'%(r,p,s,c))
                        if os.path.exists(i_file):                            
                            G[r][p][s][c] = sacpy.sac()
                            G[r][p][s][c].read(i_file)
                        else:
                            print('Skipping GF for {} {}'.format(s,c))

        # All done
        return

    def saveKinData(self, data, o_dir='data_kin'):
        '''
        Write Data (1 sac file per channel)
        Args:
            data  : Data object corresponding to the Green's function to be saved
            o_dir : Output file name
        '''

        # Print stuff
        print('Writing Kinematic Data to file {} for fault {} and dataset {}'.format(o_dir,self.name,data.name))

        # Write data in sac file
        d = self.d[data.name]
        f = open(os.path.join(o_dir,'stat_list'),'w') # List of stations
        for s in d: # station name (string)
            f.write('%s\n'%(s))
            for c in d[s]: # component name (string)
                o_file = os.path.join(o_dir,'data_%s_%s.sac'%(s,c))
                d[s][c].write(o_file)
        f.close()
        
        # All done
        return

    def loadKinData(self, data, i_dir='data_kin', station_file=None):
        '''
        Read Data (1 sac file per channel)
        Args:
            data  : Data object corresponding to the Green's function to be loaded
            i_dir : Input directory
            station_file: read station list from 'station_file'
        '''

        # Import sac for python (ask Zach)
        import sacpy

        # Print stuff
        print('Loading Kinematic Data from directory {} for fault {} and dataset {}'.format(i_dir,self.name,data.name))

        # Check list of station names
        if station_file != None:
            sta_name = []
            f = open(os.path.join(o_dir,'stat_list'),'r')
            for l in f:
                sta_name.append(l.strip().split()[0])
            f.close()
        else:
            sta_name = data.sta_name

        # Read data from sac files
        self.d[data.name] = {}
        d = self.d[data.name]
        for s in sta_name: # station name (string)
            d[s]={}
            for c in ['Z','N','E']:      # component name (string)
                o_file = os.path.join(i_dir,'data_%s_%s.sac'%(s,c))
                if os.path.exists(o_file):
                    d[s][c] = sacpy.sac()
                    d[s][c].read(o_file)
                else:
                    print('Skipping Data for {} {}'.format(s,c))                    
        # All done
        return


#EOF
