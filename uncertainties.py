'''
A fault class dealing with uncertainties in the forward model (fault geometry, crustal elastic properties)

Written by T. Ragon September 2018
'''

# Externals 
import numpy as np
import os
from scipy.stats import linregress

# Personals
from .multifaultsolve import multifaultsolve as multiflt

class uncertainties(object):
       
    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, faults, datasets, export=None, verbose=True):
        '''
        Class calculating the covariance matrix of the predictions.
    
        Args:
            * name      : Name of the object
            * faults    : Prior fault geometry, can be a list of several faults
            * datasets  : List of data objects
                          ex: dataset=[gps]+[insar1,insar2]
            * export    : None by default
                          or string with path to directory
    
        '''
        
        self.verbose = verbose
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing uncertainties object")
            
        # Store things into self
        self.name = name
        self.faults = faults
        self.datasets = datasets
        self.export = export
        # check the utm zone
        if np.shape(self.faults)!=():
            self.utmzone = faults[0].utmzone
            for fault in faults:
                if fault.utmzone is not self.utmzone:
                    print("UTM zones are not equivalent, this is a problem")
                    self.ready = False
                    return
            self.putm = faults[0].putm   
        else:
            self.utmzone = faults.utmzone
                                     
        # check that Cd have been assembled prior to initialization
        if np.shape(self.faults)!=():                      
            for fault in faults:
    #            if fault.dassembled is None:
    #                print("d has not been assembled in fault structure {}".format(fault.name))
                if fault.Cd is None:
                    print("Cd has not been assembled in fault structure {}".format(fault.name))
                else:
                    self.Cd = fault.Cd
        else:
            if self.faults.Cd is None:
                print("Cd has not been assembled in fault structure {}".format(self.faults.name))
            else:
                self.Cd = self.faults.Cd
        
        # Set the kernels and covariance matrices as empty arrays
        self.KDip = []
        self.KStrike = []
        self.KPosition = []
        self.KElastic = []
        self.KernelsFull = []
        
        self.CovDip = []
        self.CovStrike = []
        self.CovPosition = []
        self.CovElastic = []
        self.CovFull = []
        
        self.CpDip = []
        self.CpStrike = []
        self.CpPosition = []
        self.CpElastic = []
        self.CpFull = []
        self.CdFull = []
        
        return
    # ----------------------------------------------------------------------
       
    # ----------------------------------------------------------------------
    # Calculate uncertainties related to the fault dip 
    def calcCpDip(self, dip_range, sigma, mprior, edks=False, **edks_params):
        '''
        Calculate the uncertainties of the predictions deriving from uncertainties in the fault dip.
        From Ragon et al. (2018) GJI
        
        Args:
            * dip_range : array, dip values (in degrees) to test around the prior fault dip (of Fault)
                    ex: dip_range = [-1,2]
                    If there are several fault segments (e.g. several faults in fault), len(dip_range) must be equal to len(fault)
                    ex: dip_range = [[-1,2],[-2,3],[-4,1],[0,5]] for a fault with 4 segments
            * sigma : prior uncertainty (standard deviation) in the dip parameter (in degrees)
                    ex: sigma = 5
                    If there are several fault segments (e.g. several faults in fault), len(sigma) must be equal to len(fault)
                    ex: sigma = [5,7,2,3] for a fault with 4 segments
            * mprior : initial model used to calculate Cp
                    length must be equal to two times the number of patches
                    can be uniform and derived from Mo, ex: meanslip = -Mo*10**(-7)/(3*10**10*length*width)
                    OR derived from a first inversion without accounting for uncertainties
        
        Kwargs:
            * edks : If True, GFs calculated using a layered Earth model calculated with EDKS.
                     If False, GFs with Okada
                     
            If edks is True, please specify in **edks_params: 
                ex: Cp_dip(fault,datasets,[40,50],sigma,mprior,edks=True,edksdir='path/to/kernels',modelname='CIA',sourceSpacing=0.5)
                * edksdir : path of the edks kernels
                * modelname : Filename of the EDKS kernels
                * sourceSpacing      : source spacing to calculate the Green's Functions
                    OR sourceNumber   : Number of sources per patches.
                    OR sourceArea     : Maximum Area of the sources.
        
        Returns:
            * CpDip
            
        /!\ To save you from having to recalculate a new fault geometry for each new dip parameter,
            each fault patches are rotated along the fault tip. When Equivalent Rectangles are 
            calculated, fault patches are approximated and it may bias slighly (0 to 5% -?)
            the value of Cp. To avoid this effect, you could instead recalculate your fault geometry.
        '''
        # For a fault with one segment
        if np.shape(self.faults)==() or np.shape(self.faults)[0]==1 :   
            
            if np.shape(self.faults)==():
                fault = self.faults
            else:
                fault = self.faults[0]
                
            # Define the range on which calculate Kernels
            dip_rg = np.linspace(dip_range[0],dip_range[1],6)
            
            # Calculate the Green's functions for the various dips in dip_rg
            gfs = []
            
            for d in dip_rg: 
                print('---------------------------------')
                print('---------------------------------')
                print('Calculating new Fault Geometry and Greens Functions')
                print('For dip = dip_prior + %.2f degrees' % (d))
                
                #  Modify the fault geometry
                p_axis1 = np.array([fault.xf[-1],fault.yf[-1],fault.top])
                p_axis2 = np.array([fault.xf[0],fault.yf[0],fault.top])
                # Copy the fault
                Fg = fault.duplicateFault()
                # Rotate the patches
                for p in range(len(fault.patch)):
                    Fg.rotatePatch(p,d, p_axis1, p_axis2)                     
                Fg.computeEquivRectangle()
              
                if edks is True:
                    os.chdir(modelname=edks_params['edksdir'])
                    if 'Spacing' in str(edks_params.items()[1][0]):
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceSpacing=edks_params.items()[1][1]))
                    elif 'Number' in str(edks_params.items()[1][0]):
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceNumber=edks_params.items()[1][1]))
                    else: 
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceArea=edks_params.items()[1][1]))
                else:
                    gfs.append(Fg.calcGFsCp(self.datasets, edks=False))
            
            # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
            slope = np.empty(gfs[0].shape)
            rvalue = np.empty(gfs[0].shape)
            pvalue = np.empty(gfs[0].shape)
            stderr = np.empty(gfs[0].shape)
            inter = np.empty(gfs[0].shape)
            for i in range(gfs[0].shape[0]):
                for j in range(gfs[0].shape[1]):
                    # Do a linear regression for each couple parameter/data, along the dip range
                    slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(dip_rg,[gfs[k][i,j] for k in range(len(dip_rg))])
            
            Kdip= slope
            K = []
            K.append(Kdip)
            kernels = np.asarray(K)
            k = np.transpose(np.matmul(kernels, mprior))
            C1 = np.matmul(k, [[np.float(sigma)**2]])
            CpDip = np.matmul(C1, np.transpose(k))
            
            self.KDip = kernels
            self.CovDip = np.array([[np.float(sigma)**2]])
            if self.KernelsFull==[]:
                self.KernelsFull = self.KDip
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KDip))
            if self.CovFull==[]:
                self.CovFull = self.CovDip
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovDip)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovDip]]))
                
            self.CpDip = CpDip
            if self.CpFull==[]:
                self.CpFull = self.CpDip
            else:
                self.CpFull = np.add(self.CpFull, self.CpDip)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KDip.astype('f').tofile(os.path.join(self.export,self.name+'KDip.bin'))
                self.CovDip.astype('f').tofile(os.path.join(self.export,self.name+'CovDip.bin'))
                
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpDip')
            
        # For a multi-segmented fault
        else:
            Kdip = []
            
            # Initialize the faults
            Faults = []
            for f in range(len(self.faults)):
                fault = self.faults[f]
                dip = dip_range[f][0]
                p_axis1 = np.array([fault.xf[0],fault.yf[0],fault.top])
                p_axis2 = np.array([fault.xf[-1],fault.yf[-1],fault.top])
                
                Fg = fault.duplicateFault()
                for p in range(len(fault.patch)):
                     Fg.rotatePatch(p,dip, p_axis1, p_axis2)                     
                Fg.computeEquivRectangle()
                    
                # Calculate GFs 
                if edks is True:
                    os.chdir(modelname=edks_params['edksdir'])
                    if 'Spacing' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceSpacing=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    elif 'Number' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceNumber=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    else: 
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceArea=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                else:
                    Fg.calcGFsCp(self.datasets,edks=False)
                    Fg.assembleGFs(self.datasets)
                    Fg.assembled(self.datasets)
                    Fg.assembleCd(self.datasets)
                
                Faults.append(Fg)
                
            for f in range(len(self.faults)):
                Faults_cp = Faults
                fault = self.faults[f]
                gfs = []
                
                # Define the range on which calculate Kernels
                dip_rg = np.linspace(dip_range[f][0],dip_range[f][1],6)
                
                p_axis1 = np.array([fault.xf[0],fault.yf[0],fault.top])
                p_axis2 = np.array([fault.xf[-1],fault.yf[-1],fault.top])
                
                # Calculate the Green's functions for the various dips in dip_rg
                for d in dip_rg: 
                    print('---------------------------------')
                    print('---------------------------------')
                    print('Calculating new Fault Geometry and Greens Functions')
                    print('For dip = dip_prior + %.2f degrees' % (d))
                    
                    # Modify the fault geometry
                    Fg = fault.duplicateFault()
                    for p in range(len(fault.patch)):
                         Fg.rotatePatch(p,d, p_axis1, p_axis2)                     
                    Fg.computeEquivRectangle()
                        
                    # Calculate GFs for each different fault geometry
                    if edks is True:
                        os.chdir(modelname=edks_params['edksdir'])
                        if 'Spacing' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceSpacing=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        elif 'Number' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceNumber=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        else: 
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceArea=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                    else:
                        Fg.calcGFsCp(self.datasets,edks=False)
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    
                    Faults_cp[f] = Fg
                    slv = multiflt('Multi-seg fault', Faults_cp)
                    slv.assembleGFs()
                    gfs.append(slv.G)
                
                # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
                slope = np.empty(gfs[0].shape)
                rvalue = np.empty(gfs[0].shape)
                pvalue = np.empty(gfs[0].shape)
                stderr = np.empty(gfs[0].shape)
                inter = np.empty(gfs[0].shape)
                
                for i in range(gfs[0].shape[0]):
                    for j in range(gfs[0].shape[1]):
                        # Do a linear regression for each couple parameter/data, along the dip range
                        slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(dip_rg,[gfs[k][i,j] for k in range(len(dip_rg))])
                
                Kdip.append(slope)
            
            kernels = np.asarray(Kdip)
            k = np.transpose(np.matmul(kernels, mprior))
            Covdip = np.zeros((len(self.faults),len(self.faults)))
            for f in range(len(self.faults)):
                Covdip[f,f] = sigma[f]**2
            C1 = np.matmul(k, Covdip)
            CpDip = np.matmul(C1, np.transpose(k))
            
            self.KDip = kernels
            self.CovDip = Covdip
            if self.KernelsFull==[]:
                self.KernelsFull = self.KDip
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KDip))
            if self.CovFull==[]:
                self.CovFull = self.CovDip
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovDip)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovDip]]))
                
            self.CpDip = CpDip
            if self.CpFull==[]:
                self.CpFull = self.CpDip
            else:
                self.CpFull = np.add(self.CpFull, self.CpDip)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KDip.astype('f').tofile(os.path.join(self.export,self.name+'KDip.bin'))
                self.CovDip.astype('f').tofile(os.path.join(self.export,self.name+'CovDip.bin'))            
            
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpDip')          
                
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Calculate uncertainties related to the fault strike    
    def calcCpStrike(self, strike_range, rotation_axis, sigma, mprior, whole_fault=True, edks=False, **edks_params):
        '''
        Calculate the uncertainties of the predictions deriving from uncertainties in the fault strike.
        If the fault is multi-segmented, you have two options:
            - investigate the variation of strike for each segment independently
                    option whole_fault = False
            - investigate the variation of strike for the whole fault (whole_fault is True by default)
        From Ragon et al. (2018) GJI
        
        Args:
            * strike_range : array, strike values (in degrees) to test around the prior fault strike (of Fault)
                    Positive angles are counter-clockwise looking toward the center of the Earth
                    ex: strike_range = [-1,2]
                    If whole_fault is False (multi-segmented fault), len(strike_range) must be equal to len(fault)
                    ex: strike_range = [[-1,2],[-2,3],[-4,1],[0,5]] for a fault with 4 segments
            * rotation_axis : can be 'center', 'first', 'second' or an had-oc position [X,Y]
                    define the axis around which the fault will rotate
                    'center' : the center of the fault
                    'first'  : the first point of the fault trace (Fault.xf[0],Fault.yf[0])
                    'last'   : the last point of the fault trace (Fault.xf[-1],Fault.yf[-1])
                    [X,Y]    : ad-hoc coordinates
            * sigma : prior uncertainty (standard deviation) in the strike parameter (in degrees)
                    ex: sigma = 5
                    If whole_fault is False (multi-segmented fault), len(sigma) must be equal to len(fault)
                    ex: sigma = [5,7,2,3] for a fault with 4 segments
            * mprior : initial model used to calculate Cp
                    length must be equal to two times the number of patches
                    can be uniform and derived from Mo, ex: meanslip = -Mo*10**(-7)/(3*10**10*length*width)
                    OR derived from a first inversion without accounting for uncertainties
            * whole_fault : True by default.
                    if True for a multi-segmented fault, the strike variation for the whole fault will be investigated
                            the first strike_range value (strike_range[0]) will be assumed as rotation angles for the whole fault
                            around the center or the whole fault, the first point of the first segment or the last point of the last segment
                    if False for a multi-segmented fault, the strike variation of each segment will be investigated independently
                            for either the center, first or last point of each fault segment
        
        Kwargs:
            * edks : If True, GFs calculated using a layered Earth model calculated with EDKS.
                     If False, GFs with Okada
                         
            If edks is True, please specify in **edks_params: 
                ex: Cp_strike(fault,datasets,[40,50],'center',sigma,mprior,edks=True,edksdir='path/to/kernels',modelname='CIA',sourceSpacing=0.5)
                * edksdir : path of the edks kernels
                * modelname : Filename of the EDKS kernels
                * sourceSpacing      : source spacing to calculate the Green's Functions
                    OR sourceNumber   : Number of sources per patches.
                    OR sourceArea     : Maximum Area of the sources.
                
        Returns:
            * CpStrike
        '''
        # For a fault with one segment
        if np.shape(self.faults)==() or np.shape(self.faults)[0]==1 :  
            
            if np.shape(self.faults)==():
                fault = self.faults
            else:
                fault = self.faults[0]
                
            # Define the range on which calculate Kernels
            stk_rg = np.linspace(strike_range[0],strike_range[1],6)
            
            # Calculate the Green's functions for the various strike values in stk_rg
            gfs = []
            
            # Define the rotation axis
            if rotation_axis == 'center':
                p_axis1 = np.array([(fault.xf[0]+fault.xf[-1])/2,(fault.yf[0]+fault.yf[-1])/2,fault.top])
                p_axis2 = np.array([(fault.xf[0]+fault.xf[-1])/2,(fault.yf[0]+fault.yf[-1])/2,fault.top+10])
            elif rotation_axis == 'first':
                p_axis1 = np.array([fault.xf[0],fault.yf[0],fault.top])
                p_axis2 = np.array([fault.xf[0],fault.yf[0],fault.top+10])
            elif rotation_axis == 'last':
                p_axis1 = np.array([fault.xf[-1],fault.yf[-1],fault.top])
                p_axis2 = np.array([fault.xf[-1],fault.yf[-1],fault.top+10])
            else:
                p_axis1 = np.array([rotation_axis[0],rotation_axis[1],fault.top])
                p_axis2 = np.array([rotation_axis[0],rotation_axis[1],fault.top+10])
            
            for s in stk_rg: 
                print('---------------------------------')
                print('---------------------------------')
                print('Calculating new Fault Geometry and Greens Functions')
                print('For strike = strike_prior + %.2f degrees' % (s))
                
               #  Modify the fault geometry                
                Fg = fault.duplicateFault()
                # rotate the patches
                for p in range(len(fault.patch)):
                    Fg.rotatePatch(p,s, p_axis1, p_axis2)                     
                Fg.computeEquivRectangle()
              
                if edks is True:
                    os.chdir(modelname=edks_params['edksdir'])
                    if 'Spacing' in str(edks_params.items()[1][0]):
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceSpacing=edks_params.items()[1][1]))
                    elif 'Number' in str(edks_params.items()[1][0]):
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceNumber=edks_params.items()[1][1]))
                    else: 
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceArea=edks_params.items()[1][1]))
                else:
                    gfs.append(Fg.calcGFsCp(self.datasets,edks=False))
            
            # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
            slope = np.empty(gfs[0].shape)
            rvalue = np.empty(gfs[0].shape)
            pvalue = np.empty(gfs[0].shape)
            stderr = np.empty(gfs[0].shape)
            inter = np.empty(gfs[0].shape)
            for i in range(gfs[0].shape[0]):
                for j in range(gfs[0].shape[1]):
                    # Do a linear regression for each couple parameter/data, along the range of strike values
                    slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(stk_rg,[gfs[k][i,j] for k in range(len(stk_rg))])
            
            Kstk= slope
            K = []
            K.append(Kstk)
            kernels = np.asarray(K)
            k = np.transpose(np.matmul(kernels, mprior))
            C1 = np.matmul(k, [[np.float(sigma)**2]])
            CpStrike = np.matmul(C1, np.transpose(k))
            
            self.KStrike = kernels
            self.CovStrike = np.array([[np.float(sigma)**2]])
            if self.KernelsFull==[]:
                self.KernelsFull = self.KStrike
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KStrike))
            if self.CovFull==[]:
                self.CovFull = self.CovStrike
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovStrike)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovStrike]]))            
            
            self.CpStrike = CpStrike
            if self.CpFull==[]:
                self.CpFull = self.CpStrike
            else:
                self.CpFull = np.add(self.CpFull, self.CpStrike)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KStrike.astype('f').tofile(os.path.join(self.export,self.name+'KStrike.bin'))
                self.CovStrike.astype('f').tofile(os.path.join(self.export,self.name+'CovStrike.bin'))
                
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with Cpstrike')
            
        # For a multi-segmented fault
        elif np.shape(self.faults)!=() and whole_fault==False:
            Kstk = []
            
            # Initialize the faults
            Faults = []
            for f in range(len(self.faults)):
                fault = self.faults[f]
                strike = strike_range[f][0]
                # Define the rotation axis
                if rotation_axis == 'center':
                    p_axis1 = np.array([(fault.xf[0]+fault.xf[-1])/2,(fault.yf[0]+fault.yf[-1])/2,fault.top])
                    p_axis2 = np.array([(fault.xf[0]+fault.xf[-1])/2,(fault.yf[0]+fault.yf[-1])/2,fault.top+10])
                elif rotation_axis == 'first':
                    p_axis1 = np.array([fault.xf[0],fault.yf[0],fault.top])
                    p_axis2 = np.array([fault.xf[0],fault.yf[0],fault.top+10])
                elif rotation_axis == 'last':
                    p_axis1 = np.array([fault.xf[-1],fault.yf[-1],fault.top])
                    p_axis2 = np.array([fault.xf[-1],fault.yf[-1],fault.top+10])
                else:
                    p_axis1 = np.array([rotation_axis[0],rotation_axis[1],fault.top])
                    p_axis2 = np.array([rotation_axis[0],rotation_axis[1],fault.top+10])
                
                Fg = fault.duplicateFault()
                for p in range(len(fault.patch)):
                     Fg.rotatePatch(p,strike, p_axis1, p_axis2)                     
                Fg.computeEquivRectangle()
                    
                # Calculate GFs 
                if edks is True:
                    os.chdir(modelname=edks_params['edksdir'])
                    if 'Spacing' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceSpacing=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    elif 'Number' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceNumber=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    else: 
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceArea=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                else:
                    Fg.calcGFsCp(self.datasets,edks=False)
                    Fg.assembleGFs(self.datasets)
                    Fg.assembled(self.datasets)
                    Fg.assembleCd(self.datasets)
                
                Faults.append(Fg)
                
                
            for f in range(len(self.faults)):
                Faults_cp = Faults
                fault = self.faults[f]
                gfs = []
                
                # Define the range on which calculate Kernels
                stk_rg = np.linspace(strike_range[f][0],strike_range[f][1],6)
                
                # Define the rotation axis
                if rotation_axis == 'center':
                    p_axis1 = np.array([(fault.xf[0]+fault.xf[-1])/2,(fault.yf[0]+fault.yf[-1])/2,fault.top])
                    p_axis2 = np.array([(fault.xf[0]+fault.xf[-1])/2,(fault.yf[0]+fault.yf[-1])/2,fault.top+10])
                elif rotation_axis == 'first':
                    p_axis1 = np.array([fault.xf[0],fault.yf[0],fault.top])
                    p_axis2 = np.array([fault.xf[0],fault.yf[0],fault.top+10])
                elif rotation_axis == 'last':
                    p_axis1 = np.array([fault.xf[-1],fault.yf[-1],fault.top])
                    p_axis2 = np.array([fault.xf[-1],fault.yf[-1],fault.top+10])
                else:
                    p_axis1 = np.array([rotation_axis[0],rotation_axis[1],fault.top])
                    p_axis2 = np.array([rotation_axis[0],rotation_axis[1],fault.top+10])
                
                # Calculate the Green's functions for the various  strike values in stk_rg
                for s in stk_rg: 
                    print('---------------------------------')
                    print('---------------------------------')
                    print('Calculating new Fault Geometry and Greens Functions')
                    print('For strike = strike_prior + %.2f degrees' % (s))
                    
                    # Modify the fault geometry                    
                    Fg = fault.duplicateFault()
                    for p in range(len(fault.patch)):
                        Fg.rotatePatch(p,s, p_axis1, p_axis2)  
                    Fg.computeEquivRectangle()
                    
                    # Calculate GFs for each different fault geometry
                    if edks is True:
                        os.chdir(modelname=edks_params['edksdir'])
                        if 'Spacing' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceSpacing=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        elif 'Number' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceNumber=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        else: 
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceArea=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                    else:
                        Fg.calcGFsCp(self.datasets,edks=False)
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    
                    Faults_cp[f] = Fg
                    slv = multiflt('Multi-seg fault', Faults_cp)
                    slv.assembleGFs()
                    gfs.append(slv.G)
                
                # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
                slope = np.empty(gfs[0].shape)
                rvalue = np.empty(gfs[0].shape)
                pvalue = np.empty(gfs[0].shape)
                stderr = np.empty(gfs[0].shape)
                inter = np.empty(gfs[0].shape)
                
                for i in range(gfs[0].shape[0]):
                    for j in range(gfs[0].shape[1]):
                        # Do a linear regression for each couple parameter/data, along the range of strike values
                        slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(stk_rg,[gfs[k][i,j] for k in range(len(stk_rg))])
                
                Kstk.append(slope)
            
            kernels = np.asarray(Kstk)
            k = np.transpose(np.matmul(kernels, mprior))
            Covstk = np.zeros((len(self.faults),len(self.faults)))
            for f in range(len(self.faults)):
                Covstk[f,f] = sigma[f]**2
            C1 = np.matmul(k, Covstk)
            CpStrike = np.matmul(C1, np.transpose(k))
            
            self.KStrike = kernels
            self.CovStrike = Covstk
            if self.KernelsFull==[]:
                self.KernelsFull = self.KStrike
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KStrike))
            if self.CovFull==[]:
                self.CovFull = self.CovStrike
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovStrike)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovStrike]]))    
            
            self.CpStrike = CpStrike
            if self.CpFull==[]:
                self.CpFull = self.CpStrike
            else:
                self.CpFull = np.add(self.CpFull, self.CpStrike)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KStrike.astype('f').tofile(os.path.join(self.export,self.name+'KStrike.bin'))
                self.CovStrike.astype('f').tofile(os.path.join(self.export,self.name+'CovStrike.bin'))
                
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpStrike') 
               
        else:        
            # Define the range on which calculate Kernels
            stk_rg = np.linspace(strike_range[0][0],strike_range[0][1],6)
            
            # Define the rotation axis
            if rotation_axis == 'center':
                p_axis1 = np.array([(self.faults[0].xf[0]+self.faults[-1].xf[-1])/2,(self.faults[0].yf[0]+self.faults[-1].yf[-1])/2,self.faults[0].top])
                p_axis2 = np.array([(self.faults[0].xf[0]+self.faults[-1].xf[-1])/2,(self.faults[0].yf[0]+self.faults[-1].yf[-1])/2,self.faults[0].top+10])
            elif rotation_axis == 'first':
                p_axis1 = np.array([self.faults[0].xf[0],self.faults[0].yf[0],self.faults[0].top])
                p_axis2 = np.array([self.faults[0].xf[0],self.faults[0].yf[0],self.faults[0].top+10])
            elif rotation_axis == 'last':
                p_axis1 = np.array([self.faults[-1].xf[-1],self.faults[-1].yf[-1],self.faults[-1].top])
                p_axis2 = np.array([self.faults[-1].xf[-1],self.faults[-1].yf[-1],self.faults[-1].top+10])
            else:
                p_axis1 = np.array([rotation_axis[0],rotation_axis[1],self.faults.top[0]])
                p_axis2 = np.array([rotation_axis[0],rotation_axis[1],self.faults.top[0]+10])
            
            Kstk = []
            
            # Initialize the faults
            Faults = []
            for f in range(len(self.faults)):
                fault = self.faults[f]
                strike = stk_rg[0]
                Fg = fault.duplicateFault()
                for p in range(len(fault.patch)):
                     Fg.rotatePatch(p,strike, p_axis1, p_axis2)                     
                Fg.computeEquivRectangle()
                # Calculate GFs 
                if edks is True:
                    os.chdir(modelname=edks_params['edksdir'])
                    if 'Spacing' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceSpacing=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    elif 'Number' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceNumber=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    else: 
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceArea=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                else:
                    Fg.calcGFsCp(self.datasets,edks=False)
                    Fg.assembleGFs(self.datasets)
                    Fg.assembled(self.datasets)
                    Fg.assembleCd(self.datasets)
                Faults.append(Fg)
            
            for f in range(len(self.faults)):
                Faults_cp = Faults
                fault = self.faults[f]
                gfs = []
                
                # Calculate the Green's functions for the various  strike values in stk_rg
                for s in stk_rg: 
                    print('---------------------------------')
                    print('---------------------------------')
                    print('Calculating new Fault Geometry and Greens Functions')
                    print('For strike = strike_prior + %.2f degrees' % (s))
                    
                    # Modify the fault geometry                    
                    Fg = fault.duplicateFault()
                    for p in range(len(fault.patch)):
                        Fg.rotatePatch(p,s, p_axis1, p_axis2)                     
                    Fg.computeEquivRectangle()
                    
                    # Calculate GFs for each different fault geometry
                    if edks is True:
                        os.chdir(modelname=edks_params['edksdir'])
                        if 'Spacing' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceSpacing=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        elif 'Number' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceNumber=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        else: 
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceArea=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                    else:
                        Fg.calcGFsCp(self.datasets,edks=False)
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    
                    Faults_cp[f] = Fg
                    slv = multiflt('Multi-seg fault', Faults_cp)
                    slv.assembleGFs()
                    gfs.append(slv.G)
                
                # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
                slope = np.empty(gfs[0].shape)
                rvalue = np.empty(gfs[0].shape)
                pvalue = np.empty(gfs[0].shape)
                stderr = np.empty(gfs[0].shape)
                inter = np.empty(gfs[0].shape)
                
                for i in range(gfs[0].shape[0]):
                    for j in range(gfs[0].shape[1]):
                        # Do a linear regression for each couple parameter/data, along the range of strike values
                        slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(stk_rg,[gfs[k][i,j] for k in range(len(stk_rg))])
                
                Kstk.append(slope)
            
            kernels = np.asarray(Kstk)
            k = np.transpose(np.matmul(kernels, mprior))
            Covstk = np.zeros((len(self.faults),len(self.faults)))
            for f in range(len(self.faults)):
                Covstk[f,f] = sigma[f]**2
            C1 = np.matmul(k, Covstk)
            CpStrike = np.matmul(C1, np.transpose(k))
            
            self.KStrike = kernels
            self.CovStrike = Covstk
            if self.KernelsFull==[]:
                self.KernelsFull = self.KStrike
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KStrike))
            if self.CovFull==[]:
                self.CovFull = self.CovStrike
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovStrike)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovStrike]]))                
            
            self.CpStrike = CpStrike
            if self.CpFull==[]:
                self.CpFull = self.CpStrike
            else:
                self.CpFull = np.add(self.CpFull, self.CpStrike)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KStrike.astype('f').tofile(os.path.join(self.export,self.name+'KStrike.bin'))
                self.CovStrike.astype('f').tofile(os.path.join(self.export,self.name+'CovStrike.bin'))
                
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpStrike')       
            
        return CpStrike
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Calculate uncertainties related to the fault position
    def calcCpPosition(self, position_range, sigma, mprior, whole_fault=True, edks=False, **edks_params):
        '''
        Calculate the uncertainties of the predictions deriving from uncertainties in the fault position.
        For a one-segment fault, will vary the position perpendicularly to the strike
        For a multi-segmented fault, you have two options:
            - investigate the variation of the position for each segment independently (may have offset between segments)
                    the position will vary perpendicularly to the strike of each segment
                    option whole_fault = False
            - investigate the variation of the position for the whole fault (whole_fault is True by default)
                    the position will vary perpendicularly to the global strike of the fault
        From Ragon et al. (2018) GJI
        
        Args:
            * position_range : array, position values (in km) to test around the prior fault position (of Fault)
                    Positive values are +90 degrees from the prior strike
                    ex: position_range = [-1,2]
                    If there are several fault segments (e.g. several faults in fault), len(position_range) must be equal to len(fault)
                    ex: position_range = [[-1,2],[-2,1],[-0.5,0.2],[0,0.5]] for a fault with 4 segments
            * sigma : prior uncertainty (standard deviation) in the fault position (in km)
                    ex: sigma = 5
                    If there are several fault segments (e.g. several faults in fault), len(sigma) must be equal to len(fault)
                    ex: sigma = [1,0.5,1,0.5] for a fault with 4 segments
            * mprior : initial model used to calculate Cp
                    length must be equal to two times the number of patches
                    can be uniform and derived from Mo, ex: meanslip = -Mo*10**(-7)/(3*10**10*length*width)
                    OR derived from a first inversion without accounting for uncertainties
            * whole_fault : True by default.
                    if True for a multi-segmented fault, the position of the whole fault will be investigated
                    if False for a multi-segmented fault, the position of each segment will be investigated independently
        
        Kwargs:
            * edks : If True, GFs calculated using a layered Earth model calculated with EDKS.
                     If False, GFs with Okada
                     
            If edks is True, please specify in **edks_params: 
                ex: Cp_strike(fault,datasets,[40,50],'center',sigma,mprior,edks=True,edksdir='path/to/kernels',modelname='CIA',sourceSpacing=0.5)
                * edksdir : path of the edks kernels
                * modelname : Filename of the EDKS kernels
                * sourceSpacing      : source spacing to calculate the Green's Functions
                    OR sourceNumber  : Number of sources per patches.
                    OR sourceArea    : Maximum Area of the sources.
                    
        Returns:
            * CpPosition
        '''
           
        # For a fault with one segment
        if np.shape(self.faults)==() or np.shape(self.faults)[0]==1 :  
            
            if np.shape(self.faults)==():
                fault = self.faults
            else:
                fault = self.faults[0]
                
            # Define the range on which calculate Kernels
            pos_rg = np.linspace(position_range[0],position_range[1],6)
            
            # Calculate the Green's functions for the various strike values in stk_rg
            gfs = []
            for r in pos_rg: 
                print('---------------------------------')
                print('---------------------------------')
                print('Calculating new Fault Geometry and Greens Functions')
                print('For position = position_prior + %.2f km' % (r))
                  
                # Get the fault strike (in radians)
                strike = np.arccos( (fault.yf[-1]-fault.yf[0]) / 
                         np.sqrt( (fault.xf[-1]-fault.xf[0])**2 
                                  + (fault.yf[-1]-fault.yf[0])**2 ))  
                        
                # Calculate the translation vector
                x_translation = np.sin(np.pi/2 - strike)* r
                y_translation = np.cos(np.pi/2 - strike)* r
                z_translation = 0
                vector = [x_translation, y_translation, z_translation]
                
                #  Modify the fault geometry
                Fg = fault.duplicateFault()
                for p in range(len(fault.patch)):
                    Fg.translatePatch(p,vector)      
                Fg.computeEquivRectangle()
              
                if edks is True:
                    os.chdir(modelname=edks_params['edksdir'])
                    if 'Spacing' in str(edks_params.items()[1][0]):
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceSpacing=edks_params.items()[1][1]))
                    elif 'Number' in str(edks_params.items()[1][0]):
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceNumber=edks_params.items()[1][1]))
                    else: 
                        gfs.append(Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceArea=edks_params.items()[1][1]))
                else:
                    gfs.append(Fg.calcGFsCp(self.datasets,edks=False))
            
            # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
            slope = np.empty(gfs[0].shape)
            rvalue = np.empty(gfs[0].shape)
            pvalue = np.empty(gfs[0].shape)
            stderr = np.empty(gfs[0].shape)
            inter = np.empty(gfs[0].shape)
            for i in range(gfs[0].shape[0]):
                for j in range(gfs[0].shape[1]):
                    # Do a linear regression for each couple parameter/data, along the range of positions
                    slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(pos_rg,[gfs[k][i,j] for k in range(len(pos_rg))])
            
            Kpos = slope
            K = []
            K.append(Kpos)
            kernels = np.asarray(K)
            k = np.transpose(np.matmul(kernels, mprior))
            C1 = np.matmul(k, [[np.float(sigma)**2]])
            CpPosition = np.matmul(C1, np.transpose(k))
            
            self.KPosition = kernels
            self.CovPosition = np.array([[np.float(sigma)**2]])
            if self.KernelsFull==[]:
                self.KernelsFull = self.KPosition
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KPosition))
            if self.CovFull==[]:
                self.CovFull = self.CovPosition
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovPosition)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovPosition]]))                
            
            self.CpPosition = CpPosition
            if self.CpFull==[]:
                self.CpFull = self.CpPosition
            else:
                self.CpFull = np.add(self.CpFull, self.CpPosition)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KPosition.astype('f').tofile(os.path.join(self.export,self.name+'KPosition.bin'))
                self.CovPosition.astype('f').tofile(os.path.join(self.export,self.name+'CovPosition.bin'))
                
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpPosition')
            
        # For a multi-segmented fault
        elif np.shape(self.faults)!=() and whole_fault==False:
            Kpos = []
            
            # Initialize the faults
            Faults = []
            for f in range(len(self.faults)):
                fault = self.faults[f]
                # Get the fault strike (in radians)
                strike = np.arccos( (fault.yf[-1]-fault.yf[0]) / 
                         np.sqrt( (fault.xf[-1]-fault.xf[0])**2 
                                + (fault.yf[-1]-fault.yf[0])**2 ))  
                    
                # Calculate the translation vector
                x_translation = np.sin(np.pi/2 - strike)* position_range[f][0]
                y_translation = np.cos(np.pi/2 - strike)* position_range[f][0]
                z_translation = 0
                vector = [x_translation, y_translation, z_translation]
            
                #  Modify the fault geometry
                Fg = fault.duplicateFault()
                for p in range(len(fault.patch)):
                    Fg.translatePatch(p,vector)      
                Fg.computeEquivRectangle()
                    
                # Calculate GFs 
                if edks is True:
                    os.chdir(modelname=edks_params['edksdir'])
                    if 'Spacing' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceSpacing=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    elif 'Number' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceNumber=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    else: 
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceArea=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                else:
                    Fg.calcGFsCp(self.datasets,edks=False)
                    Fg.assembleGFs(self.datasets)
                    Fg.assembled(self.datasets)
                    Fg.assembleCd(self.datasets)
                
                Faults.append(Fg)
            
            for f in range(len(self.faults)):
                Faults_cp = Faults
                fault = self.faults[f]                
                gfs = []
                
                # Define the range on which calculate Kernels
                pos_rg = np.linspace(position_range[f][0],position_range[f][1],6)
                
                # Calculate the Green's functions for the various  strike values in stk_rg
                for r in pos_rg: 
                    print('---------------------------------')
                    print('---------------------------------')
                    print('Calculating new Fault Geometry and Greens Functions')
                    print('For position = position_prior + %.2f km' % (r))
                    
                    # Get the fault strike (in radians)
                    strike = np.arccos( (fault.yf[-1]-fault.yf[0]) / 
                             np.sqrt( (fault.xf[-1]-fault.xf[0])**2 
                                    + (fault.yf[-1]-fault.yf[0])**2 ))  
                        
                    # Calculate the translation vector
                    x_translation = np.sin(np.pi/2 - strike)* r
                    y_translation = np.cos(np.pi/2 - strike)* r
                    z_translation = 0
                    vector = [x_translation, y_translation, z_translation]
                
                    #  Modify the fault geometry
                    Fg = fault.duplicateFault()
                    for p in range(len(fault.patch)):
                        Fg.translatePatch(p,vector)      
                    Fg.computeEquivRectangle()
                    
                    # Calculate GFs for each different fault geometry
                    if edks is True:
                        os.chdir(modelname=edks_params['edksdir'])
                        if 'Spacing' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceSpacing=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        elif 'Number' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceNumber=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        else: 
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceArea=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                    else:
                        Fg.calcGFsCp(self.datasets,edks=False)
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    
                    Faults_cp[f] = Fg
                    slv = multiflt('Multi-seg fault', Faults_cp)
                    slv.assembleGFs()
                    gfs.append(slv.G)
                
                # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
                slope = np.empty(gfs[0].shape)
                rvalue = np.empty(gfs[0].shape)
                pvalue = np.empty(gfs[0].shape)
                stderr = np.empty(gfs[0].shape)
                inter = np.empty(gfs[0].shape)
                
                for i in range(gfs[0].shape[0]):
                    for j in range(gfs[0].shape[1]):
                        # Do a linear regression for each couple parameter/data, along the range of strike values
                        slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(pos_rg,[gfs[k][i,j] for k in range(len(pos_rg))])
                
                Kpos.append(slope)
            
            kernels = np.asarray(Kpos)
            k = np.transpose(np.matmul(kernels, mprior))
            Covpos = np.zeros((len(self.faults),len(self.faults)))
            for f in range(len(self.faults)):
                Covpos[f,f] = sigma[f]**2
            C1 = np.matmul(k, Covpos)
            CpPosition = np.matmul(C1, np.transpose(k))
            
            self.KPosition = kernels
            self.CovPosition = Covpos
            if self.KernelsFull==[]:
                self.KernelsFull = self.KPosition
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KPosition))
            if self.CovFull==[]:
                self.CovFull = self.CovPosition
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovPosition)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovPosition]]))                  
            
            self.CpPosition = CpPosition
            if self.CpFull==[]:
                self.CpFull = self.CpPosition
            else:
                self.CpFull = np.add(self.CpFull, self.CpPosition)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KPosition.astype('f').tofile(os.path.join(self.export,self.name+'KPosition.bin'))
                self.CovPosition.astype('f').tofile(os.path.join(self.export,self.name+'CovPosition.bin'))
                
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpPosition')     
    
        else:        
            # Define the range on which calculate Kernels
            pos_rg = np.linspace(position_range[0][0],position_range[0][1],6)
            
            Kpos = []
            
            # Get the global fault strike (in radians)
            strike = np.arccos( (self.faults[-1].yf[-1]-self.faults[0].yf[0]) / 
                     np.sqrt( (self.faults[-1].xf[-1]-self.faults[0].xf[0])**2 
                            + (self.faults[-1].yf[-1]-self.faults[0].yf[0])**2 ))  
                            
            Faults = []
            for f in range(len(self.faults)):
                fault = self.faults[f]                    
                # Calculate the translation vector
                x_translation = np.sin(np.pi/2 - strike)* position_range[f][0]
                y_translation = np.cos(np.pi/2 - strike)* position_range[f][0]
                z_translation = 0
                vector = [x_translation, y_translation, z_translation]
            
                #  Modify the fault geometry
                Fg = fault.duplicateFault()
                for p in range(len(fault.patch)):
                    Fg.translatePatch(p,vector)      
                Fg.computeEquivRectangle()
                    
                # Calculate GFs 
                if edks is True:
                    os.chdir(modelname=edks_params['edksdir'])
                    if 'Spacing' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceSpacing=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    elif 'Number' in str(edks_params.items()[1][0]):
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceNumber=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    else: 
                        Fg.calcGFsCp(self.datasets,edks=True,
                            modelname=edks_params['edksdir']+edks_params['modelname'],
                            sourceArea=edks_params.items()[1][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                else:
                    Fg.calcGFsCp(self.datasets,edks=False)
                    Fg.assembleGFs(self.datasets)
                    Fg.assembled(self.datasets)
                    Fg.assembleCd(self.datasets)
                
                Faults.append(Fg)                
            
            
            for f in range(len(self.faults)):
                fault = self.faults[f]          
                gfs = []
                
                # Calculate the Green's functions for the various  strike values in stk_rg
                for r in pos_rg: 
                    print('---------------------------------')
                    print('---------------------------------')
                    print('Calculating new Fault Geometry and Greens Functions')
                    print('For position = position_prior + %.2f km' % (r))
                    
                    # Calculate the translation vector
                    x_translation = np.sin(np.pi/2 - strike)* r
                    y_translation = np.cos(np.pi/2 - strike)* r
                    z_translation = 0
                    vector = [x_translation, y_translation, z_translation]
                
                    #  Modify the fault geometry
                    Fg = fault.duplicateFault()
                    for p in range(len(fault.patch)):
                        Fg.translatePatch(p,vector)      
                    Fg.computeEquivRectangle()
                    
                    # Calculate GFs for each different fault geometry
                    if edks is True:
                        os.chdir(modelname=edks_params['edksdir'])
                        if 'Spacing' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceSpacing=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        elif 'Number' in str(edks_params.items()[1][0]):
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceNumber=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                        else: 
                            Fg.calcGFsCp(self.datasets,edks=True,
                                modelname=edks_params['edksdir']+edks_params['modelname'],
                                sourceArea=edks_params.items()[1][1])
                            Fg.assembleGFs(self.datasets)
                            Fg.assembled(self.datasets)
                            Fg.assembleCd(self.datasets)
                    else:
                        Fg.calcGFsCp(self.datasets,edks=False)
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    
                    Faults_cp[f] = Fg
                    slv = multiflt('Multi-seg fault', Faults_cp)
                    slv.assembleGFs()
                    gfs.append(slv.G)
                
                # Calculate the sensitivity Kernels of the Green's Functions (the derivatives) by linearizing their variation
                slope = np.empty(gfs[0].shape)
                rvalue = np.empty(gfs[0].shape)
                pvalue = np.empty(gfs[0].shape)
                stderr = np.empty(gfs[0].shape)
                inter = np.empty(gfs[0].shape)
                
                for i in range(gfs[0].shape[0]):
                    for j in range(gfs[0].shape[1]):
                        # Do a linear regression for each couple parameter/data, along the range of strike values
                        slope[i,j], inter[i,j], rvalue[i,j], pvalue[i,j], stderr[i,j] = linregress(pos_rg,[gfs[k][i,j] for k in range(len(pos_rg))])
                
                Kpos.append(slope)
            
            kernels = np.asarray(Kpos)
            k = np.transpose(np.matmul(kernels, mprior))
            Covpos = np.zeros((len(self.faults),len(self.faults)))
            for f in range(len(self.faults)):
                Covpos[f,f] = sigma[f]**2
            C1 = np.matmul(k, Covpos)
            CpPosition = np.matmul(C1, np.transpose(k))
            
            self.KPosition = kernels
            self.CovPosition = Covpos
            if self.KernelsFull==[]:
                self.KernelsFull = self.KPosition
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KPosition))
            if self.CovFull==[]:
                self.CovFull = self.CovPosition
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovPosition)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovPosition]]))                   
            
            self.CpPosition = CpPosition
            if self.CpFull==[]:
                self.CpFull = self.CpPosition
            else:
                self.CpFull = np.add(self.CpFull, self.CpPosition)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KPosition.astype('f').tofile(os.path.join(self.export,self.name+'KPosition.bin'))
                self.CovPosition.astype('f').tofile(os.path.join(self.export,self.name+'CovPosition.bin'))
                
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpPosition')   
                
        return CpPosition
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Calculate the uncertainties related to the elastic properties of the Earth
    def calcCpElastic(self,pert,sigma,mprior,edksdir,modelname,pert_kernels_name,**edks_source_spec):
        '''
        Calculate the uncertainties of the predictions deriving from uncertainties in the Earth elastic structure.
        From Duputel et al. (2014) GJI
        
        Before using this function, you need to precalculate the EDKS kernels with perturbed layers.
        For each layer independently, modify the elastic modulus by a coefficient "pert" in the velocity model,
        and subsequently modify velocities Vs and Vp. (you can use the function calcPerturbatedVelocity)
        
        Args:
            * pert : value of the perturbation applied to the elastic modulus of each layer to precompute the kernels
                     (difference between logarithms)
                     pert = log(mu) - log(mu0), with mu new value and mu0 initial value
                     ex: pert = 1
            * sigma : list, prior uncertainty (standard deviation) in the logarithm of mu
                      1 sigma has to be specified for each layer
                      ex: sigma = [0.12, 0.12, 0.12, 0.4, 0.4, 0.4] for a velocity model with 6 layers
            * mprior : array, initial model used to calculate Cp
                    length must be equal to two times the number of patches
                    can be uniform and derived from Mo, ex: meanslip = -Mo*10**(-7)/(3*10**10*length*width)
                    OR derived from a first inversion without accounting for uncertainties
            * edksdir : patch of the edks kernels
            * modelname : filename of the "regular" (without any perturbation) EDKS kernels (without ".edks")
            * pert_kernels_name : list, filenames of the perturbated kernels 
                                  1 filename has to be secified for each layer
            * do not forget the Kwargs!!
        Kwargs:
                * sourceSpacing    : source spacing to calculate the Green's Functions
                  OR sourceNumber  : Number of sources per patches.
                  OR sourceArea    : Maximum Area of the sources.
                    
        Returns:
            * CpElastic
        '''
        
        fi = open(edksdir+modelname+'.model','r')
        ri = fi.readlines()
        nbr_layers = int(ri[0].split(' ')[0])
        fi.close()
        
        if sigma is not list and nbr_layers>1:
            print('You need to specify sigma for each layer')
            print('by default, same sigma used for each layer')
        elif sigma is list and len(sigma)!=nbr_layers:
            print('You need to specify sigma for each layer')
        if len(pert_kernels_name) != nbr_layers:
            print('You need to specify a name for the perturbated EDKS kernel of each layer')
        
        # For a fault with one segment
        if np.shape(self.faults)==() or np.shape(self.faults)[0]==1 :
            
            if np.shape(self.faults)==():
                fault = self.faults
            else:
                fault = self.faults[0]
            
            os.chdir(edksdir)
            Fg = fault.duplicateFault()
            if 'Spacing' in str(edks_source_spec.items()[0][0]):
                gfs_ini = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+modelname,\
                                 sourceSpacing=edks_source_spec.items()[0][1])
            elif 'Number' in str(edks_source_spec.items()[0][0]):
                gfs_ini = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+modelname,\
                                 sourceNumber=edks_source_spec.items()[0][1])
            else: 
                gfs_ini = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+modelname,\
                                 sourceArea=edks_source_spec.items()[0][1])
                                                          
            K = []
            for l in range(nbr_layers):                              
                # Calculate GFs
                if 'Spacing' in str(edks_source_spec.items()[0][0]):
                    gfs = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+pert_kernels_name[l],\
                                     sourceSpacing=edks_source_spec.items()[0][1])
                elif 'Number' in str(edks_source_spec.items()[0][0]):
                    gfs = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+pert_kernels_name[l],\
                                     sourceNumber=edks_source_spec.items()[0][1])
                else: 
                    gfs = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+pert_kernels_name[l],\
                                     sourceArea=edks_source_spec.items()[0][1])
                                     
                # Calculate Kernel for the l layer
                K.append((gfs-gfs_ini)/pert)
            
            kernels = np.asarray(K)
            k = np.transpose(np.matmul(kernels, mprior))
            Covmu = np.zeros((nbr_layers,nbr_layers))
            if sigma is not list:
                for l in range(nbr_layers):
                    Covmu[l,l] = (sigma)**2
            else:
                for l in range(nbr_layers):
                    Covmu[l,l] = (sigma[l])**2
            C1 = np.matmul(k, Covmu)
            CpElastic = np.matmul(C1, np.transpose(k))
            
            self.KElastic = kernels
            self.CovElastic = Covmu
            if self.KernelsFull==[]:
                self.KernelsFull = self.KElastic
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KElastic))
            if self.CovFull==[]:
                self.CovFull = self.CovElastic
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovElastic)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovElastic]]))                   
            
            self.CpElastic = CpElastic
            if self.CpFull==[]:
                self.CpFull = self.CpElastic
            else:
                self.CpFull = np.add(self.CpFull, self.CpElastic)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KElastic.astype('f').tofile(os.path.join(self.export,self.name+'KElastic.bin'))
                self.CovElastic.astype('f').tofile(os.path.join(self.export,self.name+'CovElastic.bin'))
                            
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpElastic')
        
        # For a multi-segmented fault
        else:
            Faults=[]
            os.chdir(edksdir)
            for f in range(len(self.faults)):
                fault = self.faults[f]            
                Fg = fault.duplicateFault()
                if 'Spacing' in str(edks_source_spec.items()[0][0]):
                    Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+modelname,\
                                     sourceSpacing=edks_source_spec.items()[0][1])
                    Fg.assembleGFs(self.datasets)
                    Fg.assembled(self.datasets)
                    Fg.assembleCd(self.datasets)
                elif 'Number' in str(edks_source_spec.items()[0][0]):
                    Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+modelname,\
                                     sourceNumber=edks_source_spec.items()[0][1])
                    Fg.assembleGFs(self.datasets)
                    Fg.assembled(self.datasets)
                    Fg.assembleCd(self.datasets)
                else: 
                    Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+modelname,\
                                     sourceArea=edks_source_spec.items()[0][1])
                    Fg.assembleGFs(self.datasets)
                    Fg.assembled(self.datasets)
                    Fg.assembleCd(self.datasets)
                    
                Faults.append(Fg)
                
            slv = multiflt('Multi-seg fault', Faults)
            slv.assembleGFs()
            gfs_ini = slv.G
            K = []
            for l in range(nbr_layers):                 
                # Calculate GFs
                Faults=[]
                for f in range(len(self.faults)):
                    fault = self.faults[f]            
                    Fg = fault.duplicateFault()
                    if 'Spacing' in str(edks_source_spec.items()[0][0]):
                        gfs = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+pert_kernels_name[l],\
                                         sourceSpacing=edks_source_spec.items()[0][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    elif 'Number' in str(edks_source_spec.items()[0][0]):
                        gfs = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+pert_kernels_name[l],\
                                         sourceNumber=edks_source_spec.items()[0][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                    else: 
                        gfs = Fg.calcGFsCp(self.datasets,edks=True,modelname=edksdir+pert_kernels_name[l],\
                                         sourceArea=edks_source_spec.items()[0][1])
                        Fg.assembleGFs(self.datasets)
                        Fg.assembled(self.datasets)
                        Fg.assembleCd(self.datasets)
                        
                    Faults.append(Fg)
                    
                slv = multiflt('Multi-seg fault', Faults)
                slv.assembleGFs()
                gfs = slv.G
                
                # Calculate Kernel for the l layer
                K.append((gfs-gfs_ini)/pert)
            
            kernels = np.asarray(K)
            k = np.transpose(np.matmul(kernels, mprior))
            Covmu = np.zeros((nbr_layers,nbr_layers))
            if sigma is not list:
                for l in range(nbr_layers):
                    Covmu[l,l] = (sigma)**2
            else:
                for l in range(nbr_layers):
                    Covmu[l,l] = (sigma[l])**2
            C1 = np.matmul(k, Covmu)
            CpElastic = np.matmul(C1, np.transpose(k))
            
            self.KElastic = kernels
            self.CovElastic = Covmu
            if self.KernelsFull==[]:
                self.KernelsFull = self.KElastic
            else:
                self.KernelsFull = np.concatenate((self.KernelsFull,self.KElastic))
            if self.CovFull==[]:
                self.CovFull = self.CovElastic
            else:
                Z = np.zeros((np.shape(self.CovFull)[0],np.shape(self.CovElastic)[0]),dtype=int)
                self.CovFull = np.asarray(np.bmat([[self.CovFull, Z], [Z, self.CovElastic]]))                      
            
            self.CpElastic = CpElastic
            if self.CpFull==[]:
                self.CpFull = self.CpElastic
            else:
                self.CpFull = np.add(self.CpFull, self.CpElastic)
            self.CdFull = np.add(self.Cd, self.CpFull)
            
            if self.export is not None:
                self.CdFull.astype('f').tofile(os.path.join(self.export,self.name+'CdFull.bin'))
                self.KernelsFull.astype('f').tofile(os.path.join(self.export,self.name+'KernelsFull.bin'))
                self.CovFull.astype('f').tofile(os.path.join(self.export,self.name+'CovFull.bin'))
                self.KElastic.astype('f').tofile(os.path.join(self.export,self.name+'KElastic.bin'))
                self.CovElastic.astype('f').tofile(os.path.join(self.export,self.name+'CovElastic.bin'))
                            
            print('---------------------------------')
            print('---------------------------------')
            print('CdFull successfully updated with CpElastic')
            
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Calculate perturbated velocity model to calculate EDKS kernels for CpElastic
    def calcPerturbatedVelocity(self,pert,velocity_mod_name):
        '''
        Calculate perturbated velocity models
        
        Args:
            * pert : value of the perturbation applied to the elastic modulus of each layer
                     ex: pert = 0.1 or pert = 10
            * velocity_mod_name = path + name of the original velocity model (without ".model")
            
        Returns:
        nothing
        
        '''
        fi = open(velocity_mod_name+'.model','r')
        ri = fi.readlines()
        nbr_layers = int(ri[0].split(' ')[0])
        fi.close()
        
        mu=[]
        for l in range(1,nbr_layers+1):
            print ('----- Layer '+str(l))
            ro = float(ri[l].split('    ')[2])
            vs = float(ri[l].split('    ')[0])
            vp = float(ri[l].split('    ')[1])
            h  = float(ri[l].split('    ')[3])
            lamb = ro*(vp**2-2*vs**2)
            mu.append(ro*vs**2)
    
            mu2 = mu[l-1]+pert
            vs2 = np.sqrt(mu2/ro)
            vp2=np.sqrt((lamb+2*mu2)/ro)
                        
            # Create new velocity model
            fil = open(velocity_mod_name+'_l'+str(l)+'.model','w')
            for li in range(len(ri)):
                if li != l:
                    fil.write("%s" %(ri[li]))
                else:
                    fil.write(" %.2f    %.2f    %.3f    %.1f \n" %(vs2,vp2,ro,h))
            fil.close()
        
        print('---------------------------------')
        print('---------------------------------')
        print('Perturbated velocity models succesfully calculated') 
        return
    # ----------------------------------------------------------------------
        
        
#EOF    
        
