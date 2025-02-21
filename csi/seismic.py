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
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import scipy.signal as signal

# Personals
#xfrom WaveMod    import sac
from .SourceInv import SourceInv

class seismic(SourceInv):
    
    '''
    A class that handles optical correlation results

    Args:
       * name      : Name of the dataset.

    Kwargs:
       * dtype     : Specifies a data type
       * utmzone   : UTM zone  (optional, default=None)
       * lon0      : Longitude of the center of the UTM zone
       * lat0      : Latitude of the center of the UTM zone
       * ellps     : ellipsoid (optional, default='WGS84')

    '''

    def __init__(self,name,dtype='seismic',utmzone=None,ellps='WGS84',lon0=None,lat0=None):
        
        super(self.__class__,self).__init__(name,utmzone,ellps,lon0,lat0) 

        # Initialize the data set 
        self.dtype = dtype
        
        # Initialize Waveform Engine
        self.waveform_engine = None

        # Initialize some things
        self.sta_name = []
        self.lat  = np.array([],dtype=np.float64)
        self.lon  = np.array([],dtype=np.float64)
        self.x    = np.array([],dtype=np.float64)
        self.y    = np.array([],dtype=np.float64)
    
        # Data
        self.d = {}

        # Covariance matrix
        self.Cd = None

        # All done
        return

    def setStat(self,sta_name,x,y,loc_format='LL'):
        '''
        Set station names and locations attributes

        Args:
            * sta_name: station names
            * x: x coordinate (longitude or UTM) 
            * y: y coordinate (latitude or UTM)

        Kwargs:
            * loc_format: location format ('LL' for lon/lat or 'XY' for UTM)

        Returns:
            * None
        '''

        # Check input parameters
        assert len(sta_name)==len(x)==len(y), 'sta_name, x and y must have the same length'
        assert loc_format=='LL' or loc_format=='XY', 'loc_format can be LL or XY'        
        if type(x)==list:
            x = np.array(x)
        if type(y)==list:
            y = np.array(y)

        # Assign input parameters to station attributes
        self.sta_name = copy.deepcopy(sta_name)
        if loc_format=='LL':            
            self.lon = np.append(self.lon,x)
            self.lat = np.append(self.lat,y)
            self.x, self.y = self.ll2xy(self.lon,self.lat)
        else:
            self.x = np.append(self.x,x)
            self.y = np.append(self.y,y)
            self.lon, self.lat = self.ll2xy(self.x,self.y)            

        # All done
        return


    def buildDiagCd(self,std):
        '''
        Build a diagonal Cd from standard deviations

        Args:
            * std: array of standard deviations

        Returns:
            * None
        '''

        assert len(std) == len(self.sta_name)

        # Set variance vector
        var_vec = np.array([])
        for i in range(len(self.sta_name)):
            stanm = self.sta_name[i]
            var_vec_sta = np.ones((self.d[stanm].npts,))*std[i]*std[i]
            var_vec = np.append(var_vec,var_vec_sta)
        
        # Build Cd from variance vector
        self.Cd = np.diag(var_vec)        
        
        # All done
        return
    

    def buildCdFromRes(self,fault,model,n_ramp_param=None,eik_solver=None,npt=4,nmesh=None,relative_error=0.2,
                       add_to_previous_Cd=False,average_correlation=False,exp_cor=False,exp_cor_len=10.):
        '''
        Build Cd from residuals

        Args:
            * fault: An instance of a fault class 
            * model: Can be a AlTar kinematic model file (posterior mean model in a txt file) or a bigM vector

        Kwargs:
            * n_ramp_param: number of nuisance parameters (e.g., InSAR orbits, used with a model file)
            * eik_solver: eikonal solver (to be used with an AlTar kinematic model file)
            * npt**2: numper of point sources per patch (to be used with an AlTar kinematic model file)
            * relative_error: standard deviation = relative_error * max(data). It can be a dictionnary
            * add_to_previous_Cd: if True, will add Cd to previous Cd
            * average_correlation: Compute average correlation for the entire set of stations
            * exp_corr: Use an exponential correlation function
            * exp_corr_len: Correlation length

        Returns:
            * None
        '''
        
        print('Computing Cd from residuals')
        G = fault.bigG
        
        # Make relative_error dictionnary
        if type(relative_error) is not dict:
            dic = {}
            for dkey in self.sta_name:
                dic[dkey] = relative_error
            relative_error = copy.deepcopy(dic)


        if type(model)==str: # Use an AlTar Kin
            print('Use model file: %s to compute residuals for Cd'%(model))
            Dtriangles = 1. # HACK: We assume Dtriangles=1 !!!
            print('Warning: Dtriangle=1 is hardcoded in buildCdFromRes')
            
            Np = len(fault.patch)        
            # Read model file
            post     = np.loadtxt(model)
            assert len(post)==4*Np + n_ramp_param + 2
            
            # Assign fault parameters
            fault.slip[:,0] = post[:Np]
            fault.slip[:,1] = post[Np:2*Np]
            fault.tr = post[2*Np+n_ramp_param:3*Np+n_ramp_param]
            fault.vr = post[3*Np+n_ramp_param:4*Np+n_ramp_param]
            h_strike = post[4*Np+n_ramp_param]
            h_dip    = post[4*Np+n_ramp_param+1]
            fault.setHypoOnFault(h_strike,h_dip)
            
            # Eikonal resolution    
            if nmesh is None:
                eik_solver.setGridFromFault(fault,1.0)
            else:            
                p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = fault.getpatchgeometry(0,center=True)
                eik_solver.setGridFromFault(fault,p_length/nmesh)
            eik_solver.fastSweep()
            
            # BigG x BigM (on the fly time-domain convolution)
            Ntriangles = int(fault.bigG.shape[1]/(2*Np))           
            m = np.zeros((G.shape[1],))
            for p in range(len(fault.patch)):
                # Location at the patch center
                p_x, p_y, p_z, p_width, p_length, p_strike, p_dip = fault.getpatchgeometry(p,center=True)
                dip_c, strike_c = fault.getHypoToCenter(p,True)
                # Grid location
                grid_size_dip = p_length/npt
                grid_size_strike = p_length/npt
                grid_strike = strike_c+np.arange(0.5*grid_size_strike,p_length,grid_size_strike) - p_length/2.
                grid_dip    = dip_c+np.arange(0.5*grid_size_dip   ,p_width ,grid_size_dip   ) - p_width/2.
                time = np.arange(Ntriangles)*Dtriangles#+Dtriangles
                T  = np.zeros(time.shape)
                Tr2 = fault.tr[p]/2.
                for i in range(npt):
                    for j in range(npt):
                        t = eik_solver.getT0([grid_dip[i]],[grid_strike[j]])[0]
                        tc = t+Tr2
                        ti = np.where(np.abs(time-tc)<Tr2)[0]            
                        T[ti] += (1/Tr2 - np.abs(time[ti]-tc)/(Tr2*Tr2))*Dtriangles
                for nt in range(Ntriangles):
                    m[2*nt*Np+p]     = T[nt] * fault.slip[p,0]/float(npt*npt)
                    m[(2*nt+1)*Np+p] = T[nt] * fault.slip[p,1]/float(npt*npt)
        else: # Use a bigM vector
            print('Use a bigM vector')
            m = model
            
        P = np.dot(G,m)

        # Select only relevant observations/predictions
        if fault.bigD_map is not None:
            idx = fault.bigD_map[self.name]
            P = P[idx[0]:idx[1]]
            D = fault.bigD[idx[0]:idx[1]]

        # Compute residual autocorrelation for each station
        n = 0
        R = P - D # Residual vector
        Cd = np.zeros((len(D),len(D)))
        if average_correlation:
            cor = signal.correlate(R,R)
            cor /= cor.max()
        if exp_cor:
            tcor = (np.arange(2*len(R)-1)-len(R)+1).astype('np.float64') 
            plt.plot(tcor,cor)
            #cor = np.exp(-(tcor*tcor)/(gauss_cor_std*gauss_cor_std))
            cor = np.exp(-np.abs(tcor)/(exp_cor_len))
            plt.plot(tcor,cor)
            plt.show()
        
        for dkey in self.sta_name:
            npts = self.d[dkey].npts
            res  = R[n:n+npts]
            obs  = self.d[dkey].depvar
            if not average_correlation:
                cor = signal.correlate(res,res)
                cor /= cor.max()
                Nc = npts
            else:
                Nc = len(R)
            
            std = obs.max()*relative_error[dkey]
            C = np.zeros((npts,npts))
            for k1 in range(npts):
                for k2 in range(npts):
                    dk = k1-k2
                    C[k1,k2] = cor[Nc+dk-1]*std*std
            Cd[n:n+npts,n:n+npts] = C.copy()
            #plt.figure()
            #plt.plot(obs)
            #plt.plot(P[n:n+npts])
            #plt.plot(res)
            #plt.show()
            n += npts
        #plt.plot(np.arange(100),cor[len(R)-1:len(R)-1+100])
        #plt.show()

        # Assign Cd attribute
        if add_to_previous_Cd:
            self.Cd += Cd
        else:
            self.Cd = Cd.copy()

        # All done return
        return

    def readCdFromBinaryFile(self,infile='kinematicG.Cd',dtype='np.float64'):
        '''
        Read kinematic Cd from a input file

        Kwargs:
            * infile: Name of the input file
            * dtype: type of data to read

        Returns:   
            * None
        '''
        
        Cd = np.fromfile(infile,dtype=dtype)
        nd = int(np.sqrt(len(Cd)))
        self.Cd = Cd.reshape(nd,nd)

        # All done
        return
    
    def writeCd2BinaryFile(self,outfile='kinematicG.Cd',dtype='np.float64'):
        '''
        Write Kinematic Cd to an output file

        :kwargs:
            * outfile: Name of the output file
            * dtype:   Type of data to write. 

        Returns:
            * None
        '''
        
        # Check if Cd exists
        assert self.Cd != None, 'Cd must be assigned'
        
        # Convert Cd
        Cd = self.Cd.astype(dtype)
        
        # Write t file
        Cd.tofile(outfile)
        
        # All done
        return
    
    def readStat(self,station_file,loc_format='LL'):
        '''
        Read station file and populate the Xr attribute (station coordinates)

        :If loc_format is 'XY':
        
        +--------+---------+---------+
        | STNAME | X_COORD | Y_COORD |
        +--------+---------+---------+

        :If loc_format is 'LL':

        +--------+-----+-----+
        | STNAME | LON | LAT |
        +--------+-----+-----+

        Args:
            * station_file: station filename including station coordinates

        Kwargs:
            * loc_format:  station file format (default= 'LL')

        Returns:
            * None
        '''
        
        # Assert if station file exists
        assert os.path.exists(station_file), 'Cannot read %s (no such file)'%(station_file)

        # Assert file format
        assert loc_format=='LL' or loc_format=='XY', 'loc_format can be either LL or XY'
        
        # Read the file 
        X = []
        Y = []
        sta_name = []
        for l in open(station_file):
            if (l.strip()[0]=='#'):
                continue
            items = l.strip().split()
            sta_name.append(items[0].strip())
            X.append(float(items[1]))
            Y.append(float(items[2]))

        # Set station attributes
        self.setStat(sta_name,X,Y,loc_format)

        # All done
        return    

    def readSac(self,sacfiles):
        '''
        Read sac data files

        Args:
            * sacfiles  : A list of input file names

        Returns:
            * None
        '''
        # Import personnal sac module
        import sacpy

        # Read sac files
        self.lon  = []
        self.lat  = []
        self.d = {}
        for sacfile in sacfiles:
            sac = sacpy.sac()
            sac.read(sacfile)
            self.lon.append(sac.stlo)
            self.lat.append(sac.stla)
            stanm = sac.kstnm+'_'+sac.kcmpnm
            self.sta_name.append(stanm)
            #if not self.d.has_key(sac.kstnm):
            #    self.d = {}
            #if not self.d[sac.kstnm].has_key(sac.kcmpnm):
            #    self.d[sac.kstnm][sac.kcmpnm] = {}
            #self.d[sac.kstnm][sac.kcmpnm[-1]] = sac.copy()
            assert stanm not in self.d, 'Multiple data for {}'.format(stanm)
            self.d[stanm] = sac.copy()

        # All done
        return


    def initWave(self,waveform_engine):
        '''
        Initialize Green's function database engine

        Args:
            * waveform_engine:  Green's function database engine

        Returns:
            * None
        '''
        
        # Assign reference to waveform_engine
        self.waveform_engine = copy.deepcopy(waveform_engine)

        # All done
        return


    def initWaveInt(self,waveform_engine):
        '''
        Initialize Bob Hermann's wavenumber integration engine

        Args:
            * waveform_engine   : Bob Hermann's wavenumber intergration engine

        Returns:
            * None
        '''
        
        # Assign reference to waveform_engine
        self.initWave(waveform_engine)

        # Assign receiver location
        self.waveform_engine.setXr(self.sta_name,self.x,self.y)

        # All done
        return

    def initWaveKK(self,waveform_engine):
        '''
        Initialize Kikuchi Kanamori waveform engine

        Args:
            * waveform_engine: Kikuchi-Kanamori waveform engine

        Returns:
            * None
        '''

        # Assign reference to waveform engine
        self.initWave(waveform_engine)

        # Get data channel names
        self.sta_name = self.waveform_engine.chans
        
        # Get waveforms and other stuff
        self.d = self.waveform_engine.data
        self.lon = []
        self.lat = []
        for dkey in self.sta_name:
            sac = self.d[dkey]
            self.lon.append(sac.stlo)
            self.lat.append(sac.stla)

        # All done
        return

    def calcSynthetics(self,dir_name,strike,dip,rake,M0,rise_time,stf_type='triangle',rfile_name=None,
                       out_type='D',src_loc=None,cleanup=True,ofd=sys.stdout,efd=sys.stderr):
        '''
        Build Green's functions for a particular source location

        Args:
            * dir_name:  Name of the directory where synthetics will be created
            * strike:    Fault strike (in deg)
            * dip:       Fault dip (in deg)
            * rake:      Fault rake (in deg)
            * M0:        Seismic moment
            * rise_time: Rise time (in sec)

        Kwargs:
            * stf_type: Type of source time function (default is 'triangle')
            * src_loc:  Point source coordinates (ndarray)
            * rfile_name: pulse file name if stf_type='rfile'
            * ofd:       stream for standard output (default=sys.stdout)
            * efd:       stream for standard error  (default=sys.stdout)        

        Returns:
            * None
        '''
        
        # Check Waveform Engine
        assert self.waveform_engine != None, 'waveform_engine must be assigned'
        if src_loc == None:
            assert self.waveform_engine.Xs != None, 'Source location must be assigned'
        else:
            self.waveform_engine.Xs = copy.deepcopy(src_loc)

        # Assign receiver locations
        assert self.waveform_engine.Xr != None, 'Recever locations must be assigned'

        # Go in dir_name
        cwd = os.getcwd()
        if cleanup and os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        os.chdir(dir_name)

        # Waveform simulation        
        self.waveform_engine.synthSDR(out_type,strike,dip,rake,M0,stf_type,rise_time,rfile_name,True,ofd,efd)
        
        # Go back
        os.chdir(cwd)
        
        if cleanup:
            shutil.rmtree(dir_name)

        # All done
        return
        
    def plot(self,synth_vector=None,nc=3,nl=5, title = 'Seismic data', sta_lst=None, basename=None,
             figsize=[11.69,8.270],xlims=None,ylims=[-20.,20.],bottom=0.06,top=0.87,left=0.06,right=0.95,wspace=0.25,
             hspace=0.35,grid=True,axis_visible=True,inc=False,Y_max=False,Y_units='mm',fault=None,
             basemap=True,globalbasemap=False,basemap_dlon=2.5,basemap_dlat=2.5,endclose=True,sort=None,alignENZ=False,
             stationYlims=False):
        '''
        Plot seismic traces

        :Note: Please complement explanations

        Kwargs:
           * synth_vector:      concatenated synthetic waveforms
           * nc:                number of collumns per page
           * nl:                number of rows per page
           * title:             figure title
           * sta_lst:           station list
           * basename:          used as prefix for figure name
           * fault:             fault object used for epicenter loc 
           * basemap:           plot basemap with epicenter and stations location
           * basemap_dlon:      Longitude steps for map
           * basemap_dlat:      Latitude steps for map
           * globalbasemap:     plot whole globe for teleseismic loc
           * endclose:          if True, close figure
           * sort:              ['distance' or 'azimuth'] you can choose to sort the stations by distance to hypocenter or by azimuth
           * alignENZ:          if True, 3 columns are plotted (ENU) and missing traces are left blank
           * stationYlims       if True, every channels of each stations will have the same ylim

        Returns:
           * None
        '''
        
        # Station list
        if sta_lst==None:
            sta_name=copy.deepcopy(self.sta_name)
        else:
            sta_name=copy.deepcopy(sta_lst)

        # Check something
        if nc != 3 and alignENZ:
            nc = 3
            print('nc was forced to 3 because alignENU is True')

        # if sort, do it
        if sort is not None:
            par = []
            names = []    
            # Get list of names and associated parameter to sort along 
            if sort.lower() in ['dist','distance']: # if can be distance...
                for dkey in self.d.keys():
                    par.append(self.d[dkey].dist)
                    names.append(self.d[dkey].kstnm+'_'+self.d[dkey].kcmpnm)
            elif sort.lower() in ['az','azimuth']: # ... or azimuth
                for dkey in self.d.keys():
                    par.append(self.d[dkey].az)
                    names.append(self.d[dkey].kstnm+'_'+self.d[dkey].kcmpnm)

            # Sort them by station name, then channel name (E,N,Z)
            names = np.array(names)[np.argsort(par)]
            sta_name = copy.deepcopy(names) 
            for i in range(len(names)-1):
                if sta_name[i][:-1]==sta_name[i+1][:-1]: 
                    if sta_name[i][-1]>sta_name[i+1][-1]:# if N before E, or Z before N, of Z before E...
                        sta_name[i],sta_name[i+1] = sta_name[i+1], sta_name[i] # ... invert their positions
                    if sta_name[i][:-1]==sta_name[i-1][:-1]:
                        if sta_name[i][-1]<sta_name[i-1][-1]: # second round to check previous element in list
                            sta_name[i],sta_name[i-1] = sta_name[i-1], sta_name[i]
            

        # Base name
        if basename==None:
            basename=self.name
        # Set station index limits in synth_vector:
        if synth_vector is not None:
            i = 0
            sta_lims  = {}
            for dkey in self.sta_name:
                sta_lims[dkey] = i
                i += self.d[dkey].npts
        # Plots per page
        perpage = nl*nc
        # Figure object
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(bottom=bottom,top=top,left=left,right=right,wspace=wspace,hspace=hspace)
        # Number of pages
        count = 1; pages = 1; nchan = 1
        ntot   = len(sta_name)
        npages = np.ceil(float(ntot)/float(perpage))
        # Main loop
        sa = 0.; sb = 0.

        # Make basemap object first to save time
        if basemap==True and fault is not None and globalbasemap==False:
            carte = plt.figure()
            m = carte.add_subplot(111, projection=ccrs.PlateCarree())
            m.set_extent([fault.hypo_lon-basemap_dlon, fault.hypo_lon+basemap_dlon, fault.hypo_lat-basemap_dlat, fault.hypo_lat+basemap_dlat],
                         projection=ccrs.PlateCarree())
            m.add_feature(cfeature.COASTLINE)
        
        for dkey in sta_name:
            # Data vector
            data  = self.d[dkey].depvar
            nsamp = self.d[dkey].npts            
            # Page change
            if count > perpage:
                if title!=None:
                    plt.suptitle(title+ ',   p %d/%d'%(pages,npages), fontsize=16, y=0.95)
                #fig.set_rasterized(True)
                o_pdf_name = '%s_page_%d.pdf'%(basename,pages)
                if os.path.exists(o_pdf_name):
                    os.remove(o_pdf_name)
                plt.savefig(o_pdf_name,orientation='landscape')
                pages += 1
                count = 1
                fig = plt.figure(figsize=figsize)
                fig.subplots_adjust(bottom=bottom,top=top,left=left,right=right,wspace=wspace,hspace=hspace)
            t1 = np.arange(nsamp,dtype='double')*self.d[dkey].delta + self.d[dkey].b - self.d[dkey].o
            
            # Skip subplot if alignENZ is set
            comp = self.d[dkey].kcmpnm[-1] # Get waveform orientation
            if alignENZ:
                col={'E':0,'N':1,'Z':2}
                while (count-1)%3 != col[comp]:
                    count+=1

            ax = plt.subplot(nl,nc,count)
            if synth_vector is not None:   
                i = sta_lims[dkey]
                if len(synth_vector.shape)==2:
                    synth = synth_vector[i:i+nsamp,:]
                    ax.plot(t1,synth*1000.,'0.6',alpha=0.05)
                    synth = synth_vector[i:i+nsamp,:].mean(axis=1)
                    ax.plot(t1,synth*1000.,'r',lw=1.5)  
                else:
                    synth = synth_vector[i:i+nsamp]
                    ax.plot(t1,synth*1000.,'r',lw=1)  
                sa = synth.min()*1000.
                sb = synth.max()*1000.
            ax.plot(t1,data*1000.,'k',lw=1)
            a = data.min()*1000.
            b = data.max()*1000.
            if sa<a:
                ymin = 1.1*sa
            else:
                ymin = 1.1*a
            if sb>b:
                ymax = 1.1*sb
            else:
                ymax = 1.1*b                
            if ymin>ylims[0]:
                ymin = ylims[0]
            if ymax<ylims[1]:
                ymax=ylims[1]

            if stationYlims: # If true, scaling is the same for every channel of each stations. 
                ymin = +999999
                ymax = -999999
                for kkk in self.d.keys():
                    if self.d[kkk].kstnm==self.d[dkey].kstnm:
                        if ymin>self.d[kkk].depvar.min()*1000.:
                            ymin=self.d[kkk].depvar.min()*1000.
                        if ymax<self.d[kkk].depvar.max()*1000.:
                            ymax=self.d[kkk].depvar.max()*1000.
            
            ax.set_ylim([ymin*1.1,ymax*1.1])
            if Y_max:                
                # label = r'%s %s %s %s $(\phi,\Delta, A) = %6.1f^{\circ}, %6.1f^{\circ}, %.0f%s$'%(
                #     self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm[-1], self.d[dkey].khole,
                #     self.d[dkey].az, self.d[dkey].gcarc,b,Y_units)
                label = r'%s %s %s $(\phi,\Delta, A) = %6.1f^{\circ}, %6.1f^{\circ}, %.0f%s$'%(
                    self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm, 
                    self.d[dkey].az, self.d[dkey].gcarc,b,Y_units)                   
            elif len(self.d[dkey].kcmpnm)>2 and self.d[dkey].kcmpnm[2] == 'Z' or inc==False:
                 #label = r'%s %s %s %s $(\phi,\Delta) = %6.1f^{\circ}, %6.1f^{\circ}$'%(
                 #    self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm[-1], self.d[dkey].khole,
                 #    self.d[dkey].az, self.d[dkey].gcarc)
                label = r'%s %s %s $(\phi,\Delta) = %6.1f^{\circ}, %6.1f^{\circ}$'%(
                    self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm, 
                    self.d[dkey].az, self.d[dkey].gcarc)                
            else:
                # label  = r'%s %s %s %s $(\phi,\Delta,\alpha) = %6.1f^{\circ},'
                # label += '%6.1f^{\circ}, %6.1f^{\circ}$'
                # label  = label%(self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm[-1], self.d[dkey].khole,
                #                 self.d[dkey].az, self.d[dkey].gcarc, self.d[dkey].cmpaz)
                label  = r'%s %s %s $(\phi,\Delta,\alpha) = %6.1f^{\circ},'
                label += '%6.1f^{\circ}, %6.1f^{\circ}$'
                label  = label%(self.d[dkey].knetwk,self.d[dkey].kstnm, self.d[dkey].kcmpnm[-1], 
                                self.d[dkey].az, self.d[dkey].gcarc, self.d[dkey].cmpaz)	                
            plt.title(label,fontsize=9.0,va='center',ha='center')                        
            if not (count-1)%nc:
                plt.ylabel(Y_units,fontsize=10)
            if (count-1)/nc == nl-1 or nchan+nc > ntot:
                plt.xlabel('time, sec',fontsize=10) 
            elif not axis_visible:
                ax.xaxis.set_visible(False)
            if not axis_visible:
                ax.yaxis.set_visible(False)
            if xlims!=None:
                plt.xlim(xlims)
            if grid:
                plt.grid()            

            if basemap==True and fault is not None and globalbasemap==False:      
                pos  = ax.get_position().get_points()
                W  = pos[1][0]-pos[0][0] ; H  = pos[1][1]-pos[0][1] ;		
                #ax2 = plt.axes([pos[1][0]-W*0.6,pos[0][1]+H*0.01,H*1.08,H*1.00])
                ax2 = plt.axes([pos[1][0]-W*0.2,pos[0][1]+H*0.01,W*0.2,H*1.00])
                m.gridlines(linewidth=0.2, linestyle=(0, (1, 1)), 
                            xloc=np.arange(fault.hypo_lat-basemap_dlat,fault.hypo_lat+basemap_dlat,3.0), 
                            yloc=np.arange(fault.hypo_lon-basemap_dlon,fault.hypo_lon+basemap_dlon,3.0))
                m.plot(self.lon,self.lat,'o',color=(1.00000,  0.74706,  0.00000),ms=4.0,alpha=1.0,zorder=1000)
                m.plot([self.d[dkey].stlo],[self.d[dkey].stla],'o',color=(1,.27,0),ms=8,alpha=1.0,zorder=1001)
                m.scatter([fault.hypo_lon],[fault.hypo_lat],c='b',marker=(5,1,0),s=120,zorder=1002)	     

            elif globalbasemap==True:
                ax = plt.axes(projection=ccrs.Orthographic(fault.hypo_lon, fault.hypo_lat))
                ax.add_feature(COASTLINE)
                m.plot(self.lon,self.lat,'o',color=(1.00000,  0.74706,  0.00000),ms=4.0,alpha=1.0,zorder=1000)
                m.plot([self.d[dkey].stlo],[self.d[dkey].stla],'o',color=(1,.27,0),ms=8,alpha=1.0,zorder=1001)
                m.scatter([fault.hypo_lon],[fault.hypo_lat],c='b',marker=(5,1,0),s=120,zorder=1002)	     
                        
            count += 1
            nchan += 1
        #fig.set_rasterized(True)
        if title!=None:
            plt.suptitle(title + ',    p %d/%d'%(pages,npages), fontsize=16, y=0.95)
        o_pdf_name = '%s_page_%d.pdf'%(basename,pages)
        if os.path.exists(o_pdf_name):
            os.remove(o_pdf_name)
        plt.savefig(o_pdf_name,orientation='landscape')
        if endclose:
            plt.close()

#EOF
