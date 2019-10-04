'''
A Class to assemble several faults into a single inverse problem. All the faults must have been intialized and constructed using the same data set.
This class allows then to:
    1. Spit the G, m, Cm, and Cd elements for a third party solver (such as Altar, for instance)
    2. Proposes a simple solution based on a least-square optimization.

Written by R. Jolivet, April 2013.

Updated by T. Shreve, May 2019, to include pressure sources in describeParams and distributem.

'''

import copy
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt

class multifaultsolve(object):
    '''
    A class that assembles the linear inverse problem for multiple faults and multiple datasets. This class can also solve the problem using simple linear least squares (bounded or unbounded).

    Args:
        * name          : Name of the project.
        * faults        : List of faults from verticalfault or pressure .

    '''

    def __init__(self, name, faults, verbose=True):

        self.verbose = verbose
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing solver object")

        # Ready to compute?
        self.ready = False
        self.figurePath = './'

        # Store things into self
        self.name = name
        self.faults = faults

        # check the utm zone
        self.utmzone = faults[0].utmzone
        for fault in faults:
            if fault.utmzone is not self.utmzone:
                print("UTM zones are not equivalent, this is a problem")
                self.ready = False
                return
        self.putm = faults[0].putm

        # check that G and d have been assembled prior to initialization
        for fault in faults:
            if fault.Gassembled is None:
                self.ready = False
                print("G has not been assembled in fault structure {}".format(fault.name))
            if fault.dassembled is None:
                self.ready = False
                print("d has not been assembled in fault structure {}".format(fault.name))

        # Check that the sizes of the data vectors are consistent
        self.d = faults[0].dassembled
        for fault in faults:
            if (fault.dassembled != self.d).all():
                print("Data vectors are not consistent, please re-consider your data in fault structure {}".format(fault.name))

        # Check that the data covariance matrix is the same
        self.Cd = faults[0].Cd
        for fault in faults:
            if (fault.Cd != self.Cd).all():
                print("Data Covariance Matrix are not consistent, please re-consider your data in fault structure {}".format(fault.name))

        # Initialize things
        self.fault_indexes = None

        # Store an array of the patch areas
        patchAreas = []
        for fault in faults:
            if fault.type is "Fault":
                if fault.patchType == 'triangletent':
                    fault.computeTentArea()
                    for tentIndex in range(fault.slip.shape[0]):
                        patchAreas.append(fault.area_tent[tentIndex])
                else:
                    fault.computeArea()
                    for patchIndex in range(fault.slip.shape[0]):
                        patchAreas.append(fault.area[patchIndex])
                self.patchAreas = np.array(patchAreas)
                self.type = "Fault"
            elif fault.type is "Pressure":
                self.type = "Pressure"
            elif fault.type in ('notafault', 'transformation'):
                print('Not a fault detected')

        # All done
        return

    def assembleGFs(self):
        '''
        Assembles the Green's functions matrix G for the concerned faults or pressure sources.

        Returns:
            * None
        '''

        # Get the faults
        faults = self.faults

        # Get the size of the total G matrix
        Nd = self.d.size
        Np = 0
        st = []
        se = []
        if self.fault_indexes is None:
            self.fault_indexes = {}
        for fault in faults:
            st.append(Np)
            Np += fault.Gassembled.shape[1]
            se.append(Np)
            self.fault_indexes[fault.name] = [st[-1], se[-1]]

        # Allocate the big G matrix
        self.G = np.zeros((Nd, Np))

        # Store the guys
        for fault in faults:
            # get the good indexes
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            # Store the G matrix
            self.G[:,st:se] = fault.Gassembled
            # Keep track of indexing
            if fault.type is "Fault":
                self.affectIndexParameters(fault)

        # self ready
        self.ready = True

        # Set the number of parameters
        self.Nd = Nd
        self.Np = Np

        # CHeck
        if self.verbose:
            print('Number of data: {}'.format(self.Nd))
            print('Number of parameters: {}'.format(self.Np))

        # Describe which parameters are what
        self.describeParams(faults)

        # All done
        return

    def OrganizeGBySlipmode(self):

        '''
        Organize G by slip mode instead of fault segment Return the new G matrix.

        Returns:
            * array

        '''

        assert len(self.faults) !=1, 'You have only one fault, why would you want to do that?'
        assert self.ready, 'You need to assemble the GFs before'

        info = self.paramDescription

        Gtemp = np.zeros((self.G.shape))

        N = 0
        slipmode = ['Strike Slip', 'Dip Slip', 'Tensile Slip', 'Coupling', 'Extra Parameters']
        for mode in slipmode:
            for fault in self.faults:
                if info[fault.name][mode].replace(' ','') != 'None':
                    ib = int(info[fault.name][mode].replace(' ','').partition('-')[0])
                    ie = int(info[fault.name][mode].replace(' ','').partition('-')[2])
                    Gtemp[:,N:N+ie-ib] = self.G[:,ib:ie]
                    N += ie-ib

        return Gtemp

    def sensitivity(self):
        '''
        Calculates the sensitivity matrix of the problem, :math:`S = \\text{diag}( G^t C_d^{-1} G )`

        Returns:
            * array
        '''
        # Import things
        import scipy.linalg as scilin
        # Invert Cd
        iCd = scilin.inv(self.Cd)
        s = np.diag(np.dot(self.G.T,np.dot(iCd,self.G)))

        # All done
        return s

    def describeParams(self, faults):
        '''
        Prints to screen  which parameters are what...

        Args:
            * faults: list of faults

        Returns:
            * None
        '''

        # initialize the counters
        ns = 0
        ne = 0
        nSlip = 0

        # Store that somewhere
        self.paramDescription = {}

        # Loop over the faults
        for fault in faults:

            # Where does this fault starts
            nfs = copy.deepcopy(ns)

            if fault.type is "Fault" or fault.type is 'transformation':
                #Prepare the table
                if self.verbose:
                    print('-----------------')
                    print('{:30s}||{:12s}||{:12s}||{:12s}||{:12s}||{:12s}'.format('Fault Name', 'Strike Slip', 'Dip Slip', 'Tensile', 'Coupling', 'Extra Parms'))
                # Initialize the values
                ss = 'None'
                ds = 'None'
                ts = 'None'
                cp = 'None'

                # Conditions on slip
                if 's' in fault.slipdir:
                    ne += fault.slip.shape[0]
                    ss = '{:12s}'.format('{:4d} - {:4d}'.format(ns,ne))
                    ns += fault.slip.shape[0]
                if 'd' in fault.slipdir:
                    ne += fault.slip.shape[0]
                    ds = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    ns += fault.slip.shape[0]
                if 't' in fault.slipdir:
                    ne += fault.slip.shape[0]
                    ts = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    ns += fault.slip.shape[0]
                if 'c' in fault.slipdir:
                    ne += fault.slip.shape[0]
                    cp = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    ns += fault.slip.shape[0]

                # How many slip parameters
                if ne>nSlip:
                    nSlip = ne

                # conditions on orbits (the rest is orbits)
                np = ne - nfs
                no = fault.Gassembled.shape[1] - np
                if no>0:
                    ne += no
                    op = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    ns += no
                else:
                    op = 'None'

                # print things
                if self.verbose:
                    print('{:30s}||{:12s}||{:12s}||{:12s}||{:12s}||{:12s}'.format(fault.name, ss, ds, ts, cp, op))

                # Store details
                self.paramDescription[fault.name] = {}
                self.paramDescription[fault.name]['Strike Slip'] = ss
                self.paramDescription[fault.name]['Dip Slip'] = ds
                self.paramDescription[fault.name]['Tensile Slip'] = ts
                self.paramDescription[fault.name]['Coupling'] = cp
                self.paramDescription[fault.name]['Extra Parameters'] = op

            elif fault.type is "Pressure":
                #Prepare the table
                if self.verbose:
                    print('{:30s}||{:12s}||{:12s}'.format('Fault Name', 'Pressure', 'Extra Parms'))
                # Initialize the values
                dp = 'None'
                if fault.source is "pCDM":
                    ne += 3
                    dp = '{:12s}'.format('{:4d} - {:4d}'.format(ns,ne))
                    ns += 3 #fault.slip.shape[0]
                else:
                    ne += 1
                    dp = '{:12s}'.format('{:4d} - {:4d}'.format(ns,ne))
                    ns += 1 #fault.slip.shape[0]


                # How many slip parameters
                if ne>nSlip:
                    nSlip = ne

                # conditions on orbits (the rest is orbits)
                np = ne - nfs
                no = fault.Gassembled.shape[1] - np
                if no>0:
                    ne += no
                    op = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    ns += no
                else:
                    op = 'None'

                # print things
                if self.verbose:
                    print('{:30s}||{:12s}||{:12s}'.format(fault.name, dp, op))

                # Store details
                self.paramDescription[fault.name] = {}
                self.paramDescription[fault.name]['pressure'] = dp
                self.paramDescription[fault.name]['Extra Parameters'] = op


        # Store the number of slip parameters
        self.nSlip = nSlip

        # all done
        return

    def assembleCm(self):
        '''
        Assembles the Model Covariance Matrix for the concerned faults.

        Returns:
            * None
        '''

        # Get the faults
        faults = self.faults

        # Get the size of Cm
        Np = 0
        st = []
        se = []
        if self.fault_indexes is None:
            self.fault_indexes = {}
        for fault in faults:
            st.append(Np)
            Np += fault.Gassembled.shape[1]
            se.append(Np)
            self.fault_indexes[fault.name] = [st[-1], se[-1]]

        # Allocate Cm
        self.Cm = np.zeros((Np, Np))

        # Store the guys
        for fault in faults:
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            self.Cm[st:se, st:se] = fault.Cm

        # Store the number of parameters
        self.Np = Np

        # All done
        return

    def affectIndexParameters(self, fault):
        '''
        Build the index parameter for a fault.

        Args:
            * fault : instance of a fault

        Returns:
            * None
        '''

        # Get indexes
        st = self.fault_indexes[fault.name][0]
        se = self.fault_indexes[fault.name][1]

        # Save the fault indexes
        fault.index_parameter = np.zeros((fault.slip.shape))
        fault.index_parameter[:,:] = 9999999
        if 's' in fault.slipdir:
            fault.index_parameter[:,0] = range(st, st+fault.slip.shape[0])
            st += fault.slip.shape[0]
        if 'd' in fault.slipdir:
            fault.index_parameter[:,1] = range(st, st+fault.slip.shape[0])
            st += fault.slip.shape[0]
        if 't' in fault.slipdir:
            fault.index_parameter[:,2] = range(st, st+fault.slip.shape[0])

        # All done
        return

    def distributem(self, verbose=False):
        '''
        After computing the m_post model, this routine distributes the m parameters to the faults.

        Kwargs:
            * verbose   : talk to me

        Returns:
            * None
        '''

        # Get the faults
        faults = self.faults

        # Loop over the faults
        for fault in faults:

            if verbose:
                print ("---------------------------------")
                print ("---------------------------------")
                print("Distribute the slip values to fault {}".format(fault.name))

            # Store the mpost
            st = self.fault_indexes[fault.name][0]
            se = self.fault_indexes[fault.name][1]
            fault.mpost = self.mpost[st:se]

            # Transformation object
            if fault.type=='transformation':
                
                # Distribute simply
                fault.distributem()

            # Fault object
            if fault.type is "Fault":

                # Affect the indexes
                self.affectIndexParameters(fault)

                # put the slip values in slip
                st = 0
                if 's' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,0] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 'd' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,1] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 't' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.slip[:,2] = fault.mpost[st:se]
                    st += fault.slip.shape[0]
                if 'c' in fault.slipdir:
                    se = st + fault.slip.shape[0]
                    fault.coupling = fault.mpost[st:se]
                    st += fault.slip.shape[0]

                # check
                if hasattr(fault, 'NumberCustom'):
                    fault.custom = {} # Initialize dictionnary
                    # Get custom params for each dataset
                    for dset in fault.datanames:
                        if 'custom' in fault.G[dset].keys():
                            nc = fault.G[dset]['custom'].shape[1] # Get number of param for this dset
                            se = st + nc
                            fault.custom[dset] = fault.mpost[st:se]
                            st += nc

            # Pressure object
            elif fault.type is "Pressure":

                st = 0
                if fault.source in {"Mogi", "Yang"}:
                    se = st + 1
                    print(np.asscalar(fault.mpost[st:se]*fault.mu))
                    fault.deltapressure = np.asscalar(fault.mpost[st:se]*fault.mu)
                    st += 1
                elif fault.source is "pCDM":
                    se = st + 1
                    fault.DVx = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    se = st + 1
                    fault.DVy = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    se = st + 1
                    fault.DVz = np.asscalar(fault.mpost[st:se]*fault.scale)
                    st += 1
                    print("Total potency scaled by", fault.scale)

                    if fault.DVtot is None:
                        fault.computeTotalpotency()
                elif fault.source is "CDM":
                    se = st + 1
                    print(np.asscalar(fault.mpost[st:se]*fault.mu))
                    fault.deltaopening = np.asscalar(fault.mpost[st:se])
                    st += 1

            # Get the polynomial/orbital/helmert values if they exist
            if fault.type in ('Fault', 'Pressure'):
                fault.polysol = {}
                fault.polysolindex = {}
                for dset in fault.datanames:
                    if dset in fault.poly.keys():
                        if (fault.poly[dset] is None):
                            fault.polysol[dset] = None
                        else:

                            if (fault.poly[dset].__class__ is not str) and (fault.poly[dset].__class__ is not list):
                                if (fault.poly[dset] > 0):
                                    se = st + fault.poly[dset]
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += fault.poly[dset]
                            elif (fault.poly[dset].__class__ is str):
                                if fault.poly[dset] is 'full':
                                    nh = fault.helmert[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                                if fault.poly[dset] in ('strain', 'strainnorotation', 'strainonly', 'strainnotranslation', 'translation', 'translationrotation'):
                                    nh = fault.strain[dset]
                                    se = st + nh
                                    fault.polysol[dset] = fault.mpost[st:se]
                                    fault.polysolindex[dset] = range(st,se)
                                    st += nh
                            elif (fault.poly[dset].__class__ is list):
                                nh = fault.transformation[dset]
                                se = st + nh
                                fault.polysol[dset] = fault.mpost[st:se]
                                fault.polysolindex[dset] = range(st,se)
                                st += nh

        # All done
        return

    def SetSolutionFromExternal(self, soln):
        '''
        Takes a vector where the solution of the problem is and affects it to mpost.

        Args:
            * soln      : array

        Returns:
            * None
        '''

        # Check if array
        if type(soln) is list:
            soln = np.array(soln)

        # put it in mpost
        self.mpost = soln

        # All done
        return

    def NonNegativeBruteSoln(self):
        '''
        Solves the least square problem argmin_x || Ax - b ||_2 for x>=0.
        No Covariance can be used here, maybe in the future.

        Returns:
            * None
        '''

        # Import what is needed
        import scipy.optimize as sciopt

        # Get things
        d = self.d
        G = self.G

        # Solution
        mpost, rnorm = sciopt.nnls(G, -1*d)

        # Store results
        self.mpost = mpost
        self.rnorm = rnorm

        # All done
        return

    def SimpleLeastSquareSoln(self):
        '''
        Solves the simple least square problem.

            :math:`\\textbf{m}_{post} = (\\textbf{G}^t \\textbf{G})^{-1} \\textbf{G}^t \\textbf{d}`

        Returns:
            * None
        '''

        # Import things
        import scipy.linalg as scilin

        # Print
        print ("---------------------------------")
        print ("---------------------------------")
        print ("Computing the Simple Least Squares")

        # Get the matrixes and vectors
        G = self.G
        d = self.d

        # Copmute
        mpost = np.dot( np.dot( scilin.inv(np.dot( G.T, G )), G.T ), d)

        # Store mpost
        self.mpost = mpost

        # All done
        return

    def UnregularizedLeastSquareSoln(self, mprior=None):
        '''
        Solves the unregularized generalized least-square problem using the following formula (Tarantolla, 2005, "Inverse Problem Theory", SIAM):

            :math:`\\textbf{m}_{post} = \\textbf{m}_{prior} + (\\textbf{G}^t \\textbf{C}_d^{-1} \\textbf{G})^{-1} \\textbf{G}^t \\textbf{C}_d^{-1} (\\textbf{d} - \\textbf{Gm}_{prior})`

        Kwargs:
            * mprior        : A Priori model. If None, then mprior = np.zeros((Nm,)).

        Returns:
            * None

        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # Import things
        import scipy.linalg as scilin

        if self.verbose:
            # Print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Computing the Unregularized Least Square Solution")

        # Get the matrixes and vectors
        G = self.G
        d = self.d
        Cd = self.Cd

        # Get the number of model parameters
        Nm = G.shape[1]

        # Check If Cm is symmetric and positive definite
        if (Cd.transpose() != Cd).all():
            print("Cd is not symmetric, Return...")
            return

        # Get the inverse of Cd
        print ("Computing the inverse of the data covariance")
        iCd = scilin.inv(Cd)

        # Construct mprior
        if mprior is None:
            mprior = np.zeros((Nm,))

        # Compute mpost
        print ("Computing m_post")
        One = scilin.inv(np.dot(  np.dot(G.T, iCd), G ) )
        Res = d - np.dot( G, mprior )
        Two = np.dot( np.dot( G.T, iCd ), Res )
        mpost = mprior + np.dot( One, Two )

        # Store m_post
        self.mpost = mpost

        # All done
        return

    def GeneralizedLeastSquareSoln(self, mprior=None, rcond=None, useCm=True):
        '''
        Solves the generalized least-square problem using the following formula (Tarantolla, 2005,         Inverse Problem Theory, SIAM):

            :math:`\\textbf{m}_{post} = \\textbf{m}_{prior} + (\\textbf{G}^t \\textbf{C}_d^{-1} \\textbf{G} + \\textbf{C}_m^{-1})^{-1} \\textbf{G}^t \\textbf{C}_d^{-1} (\\textbf{d} - \\textbf{Gm}_{prior})`

        Args:
            * mprior        : A Priori model. If None, then mprior = np.zeros((Nm,)).

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # Import things
        import scipy.linalg as scilin

        if self.verbose:
            # Print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Computing the Generalized Inverse")

        def computeMwDiff(m, Mw_thresh, patchAreas, mu):
            """
            Ahhhhh hard coded shear modulus.
            Probably need to edit this to include tensile as well ???
            """
            Npatch = len(self.patchAreas)
            shearModulus = mu #22.5e9
            if len(m) < 2*Npatch:             #If only one component of slip (dip or strikeslip)
                slip = np.sqrt(m[:Npatch]**2)
            else:                                #If both components of slip (dip or strikeslip)
                slip = np.sqrt(m[:Npatch]**2+m[Npatch:2*Npatch]**2)
            moment =  np.abs(np.dot(shearModulus * patchAreas, slip))
            if moment>0.:
                Mw = 2.0 / 3.0 * (np.log10(moment) - 9.1)
                print("Magnitude is")
                print(Mw)
            else:
                Mw = -6.0
            return np.array([Mw_thresh - Mw])

        # Get the matrixes and vectors
        G = self.G
        d = self.d
        Cd = self.Cd
        Cm = self.Cm
        # Get the number of model parameters
        Nm = Cm.shape[0]

        # Check If Cm is symmetric and positive definite
        if useCm and (Cm.transpose() != Cm).all():
            print("Cm is not symmetric, Return...")
            return

        # Get the inverse of Cm
        if useCm:
            print ("Computing the inverse of the model covariance")
            iCm = scilin.inv(Cm)
        else:
            iCm = np.zeros(Cm.shape)

        # Check If Cm is symmetric and positive definite
        if (Cd.transpose() != Cd).all():
            print("Cd is not symmetric, Return...")
            return

        # Get the inverse of Cd
        print ("Computing the inverse of the data covariance")
        if rcond is None:
            iCd = scilin.inv(Cd)
        else:
            iCd = np.linalg.pinv(Cd, rcond=rcond)

        # Construct mprior
        if mprior is None:
            mprior = np.zeros((Nm,))

        # Compute mpost
        print ("Computing m_post")
        One = scilin.inv(np.dot(  np.dot(G.T, iCd), G ) + iCm )
        Res = d - np.dot( G, mprior )
        Two = np.dot( np.dot( G.T, iCd ), Res )
        mpost = mprior + np.dot( One, Two )
        Err = d - np.dot( G, mpost )
        # Store m_post
        self.mpost = mpost
        print ("Compute cost function")
        print(np.sum(Err**2))
        mu = 22.5e9
        Mw_thresh = 10
        if self.type is "Fault":
            computeMwDiff(self.mpost, Mw_thresh, self.patchAreas*1.e6, mu)

        # Compute Cmpost
        self.Cmpost = np.linalg.inv(G.T.dot(iCd).dot(G) + iCm)

        # All done
        return

    def ConstrainedLeastSquareSoln(self, mprior=None, Mw_thresh=None, bounds=None,
                                   method='SLSQP', rcond=None,
                                   iterations=100, tolerance=None, maxfun=100000,
                                   checkIter=False, checkNorm=False):
        """
        Solves the least squares problem:

            :math:`\\text{min} [ (\\textbf{d} - \\textbf{Gm})^t \\textbf{C}_d^{-1} (\\textbf{d} - \\textbf{Gm}) + \\textbf{m}^t \\textbf{C}_m^{-1} \\textbf{m}]`

        Args:
            * mprior          : a priori model; if None, mprior = np.zeros((Nm,))
            * Mw_thresh       : upper bound on moment magnitude
            * bounds          : list of tuple bounds for every parameter
            * method          : solver for constrained minimization: SLSQP, COBYLA, or nnls
            * rcond           : Add some conditionning for all inverse matrix to compute
            * iterations      : Modifies the maximum number of iterations for the solver (default=100).
            * tolerance       : Solver's tolerance
            * maxfun          : maximum number of funcrtion evaluation
            * checkIter       : Show Stuff
            * checkNorm       : prints the norm

        Returns:
            * None
        """
        assert self.ready, 'You need to assemble the GFs'

        # Import things
        import scipy.linalg as scilin
        from scipy.optimize import minimize, nnls

        # Check the provided method is valid
        assert method in ['SLSQP', 'COBYLA', 'nnls', 'TNC', 'L-BFGS-B'], 'unsupported minimizing method'

        if self.verbose:
            # Print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Computing the Constrained least squares solution")

        # Get the matrixes and vectors
        G = self.G
        d = self.d
        Cd = self.Cd
        Cm = self.Cm

        # Get the number of model parameters
        Nm = Cm.shape[0]

        # Check If Cm is symmetric and positive definite
        if (Cm.transpose() != Cm).all():
            print("Cm is not symmetric, Return...")
            return

        # Get the inverse of Cm
        if self.verbose:
            print ("Computing the inverse of the model covariance")
        if rcond is None:
            iCm = scilin.inv(Cm)
        else:
            iCm = scilin.pinv(Cm, rcond=rcond)

        # Check If Cm is symmetric and positive definite
        if (Cd.transpose() != Cd).all():
            print("Cd is not symmetric, Return...")
            return

        # Get the inverse of Cd
        if self.verbose:
            print ("Computing the inverse of the data covariance")
        if rcond is None:
            iCd = scilin.inv(Cd)
        else:
            iCd = np.linalg.pinv(Cd, rcond=rcond)

        # Construct mprior
        if mprior is None:
            mprior = np.zeros((Nm,))

        # Define the cost function
        def costFunction(m, G, d, iCd, iCm, mprior, verbose=False):
            """
            Compute data + prior misfits.
            """
            dataMisfit = d - np.dot(G,m)
            dataLikely = np.dot(dataMisfit, np.dot(iCd, dataMisfit))
            priorMisfit = m - mprior
            priorLikely = np.dot(priorMisfit, np.dot(iCm, priorMisfit))
            if verbose:
                print(0.5 * dataLikely + 0.5 * priorLikely)
            return 0.5 * dataLikely + 0.5 * priorLikely

        # Define the moment magnitude inequality constraint function
        def computeMwDiff(m, Mw_thresh, patchAreas, mu):
            """
            Ahhhhh hard coded shear modulus.
            """
            Npatch = len(self.patchAreas)
            shearModulus = mu #22.5e9
            slip = np.sqrt(m[:Npatch]**2+m[Npatch:2*Npatch]**2)
            moment =  np.abs(np.dot(shearModulus * patchAreas, slip))
            if moment>0.:
                Mw = 2.0 / 3.0 * (np.log10(moment) - 9.1)
                print("Magnitude is"+ Mw)
            else:
                Mw = -6.0
            return np.array([Mw_thresh - Mw])

        # Define the constraints dictionary
        if Mw_thresh is not None:
            # Get shear modulus values
            mu = np.array(())
            for fault in self.faults:
                mu = np.append(mu,fault.mu)
            if None in mu.tolist(): # If mu not set in one fault, fix it for all of them
                mu = 22.5e9

            constraints = {'type': 'ineq',
                           'fun': computeMwDiff,
                           'args': (Mw_thresh, self.patchAreas*1.e6, mu)}
        else:
            constraints = ()

        # Call solver
        if method == 'nnls':
            if self.verbose:
                print("Performing non-negative least squares")
            # Compute cholesky decomposition of iCd and iCm
            L = np.linalg.cholesky(iCd)
            M = np.linalg.cholesky(iCm)
            # Form augmented matrices and vectors
            d_zero = d - np.dot(G, mprior)
            F = np.vstack((np.dot(L.T, G), M.T))
            b = np.hstack((np.dot(L.T, d_zero), np.zeros_like(mprior)))
            m = nnls(F, b)[0] + mprior

        else:
            if self.verbose:
                print("Performing constrained minimzation")
            options = {'disp': checkIter, 'maxiter': iterations}
            if method=='L-BFGS-B':
                options['maxfun']= maxfun
            res = minimize(costFunction, mprior, args=(G,d,iCd,iCm,mprior,checkNorm),
                           constraints=constraints, method=method, bounds=bounds,
                           options=options, tol=tolerance)
            m = res.x
        #final data + prior misfits is
        self.cost = res.fun
        # Store result
        self.mpost = m

        # All done
        return

    def simpleSampler(self, priors, initialSample, nSample, nBurn, plotSampler=False,
                            writeSamples=False, dryRun=False, adaptiveDelay=300):
        '''
        Uses a Metropolis algorithme to sample the posterior distribution of the model
        following Bayes's rule. This is exactly what is done in AlTar, but using an
        open-source library called pymc. This routine is made for simple problems with
        few parameters (i.e. More than 30 params needs a very fast computer).

        Args:
            * priors        : List of priors. Each prior is specified by a list.
                    - Example: priors = [ ['Name of parameter', 'Uniform', min, max], ['Name of parameter', 'Gaussian', center, sigma] ]
            * initialSample : List of initialSample.
            * nSample       : Length of the Metropolis chain.
            * nBurn         : Number of samples burned.

        Kwargs:
            * plotSampler   : Plot some usefull stuffs from the sampler (default: False).
            * writeSamples  : Write the samples to a binary file.
            * dryRun        : If True, builds the sampler, saves it, but does not run. This can be used for debugging.
            * adaptiveDelay : Recompute the covariance of the proposal every adaptiveDelay steps

        The result is stored in self.samples. The variable mpost is the mean of the final sample set.

        Returns:
            * None
        '''

        if self.verbose:
            # Print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Running a Metropolis algorythm to")
            print ("sample the posterior PDFs of the ")
            print ("  model: P(m|d) = C P(m) P(d|m)  ")

        # Import
        try:
            import pymc
        except:
            print('This method uses pymc. Please install it')

        # Get the matrixes and vectors
        assert hasattr(self, 'G'), 'Need an assembled G matrix...'
        G = self.G
        assert hasattr(self, 'd'), 'Need an assembled data vector...'
        dobs = self.d
        assert hasattr(self, 'Cd'), 'Need an assembled data covariance matrix...'
        Cd = self.Cd

        # Assert
        assert len(priors)==G.shape[1], 'Not enough informations to estimate prior information...'
        assert len(priors)==len(initialSample), 'There must be as many \
                                        initialSamples ({}) as priors ({})...'.format(len(initialSample),len(priors))
        if type(initialSample) is not list:
            try:
                initialSample = initialSample.tolist()
            except:
                print('Please provide a list of initialSample')
                sys.exit(1)

        # Build the prior PDFs
        priorFunctions = []
        for prior, init in zip(priors, initialSample):
            name = prior[0]
            function = prior[1]
            params = prior[2:]
            if function is 'Gaussian':
                center = params[0]
                tau = params[1]
                p = pymc.Gaussian(name, center, tau, value=init)
            elif function is 'Uniform':
                boundMin = params[0]
                boundMax = params[1]
                p = pymc.Uniform(name, boundMin, boundMax, value=init)
            else:
                print('This prior type has not been implemented yet...')
                print('Although... You can do it :-)')
                sys.exit(1)
            priorFunctions.append(p)

        # Build the prior function
        @pymc.stochastic
        def prior(value=initialSample):
            prob = 0.
            for prior, val in zip(priorFunctions, value):
                prior.set_value(val)
                prob += prior.logp
            return prob

        # Build the forward model
        @pymc.deterministic(plot=False)
        def forward(theta=[prior]):
            return G.dot(np.array(theta).squeeze())

        # Build the observation
        likelihood = pymc.MvNormal('Data', mu=forward, tau=np.linalg.inv(Cd),
                                           value=dobs, observed=True)

        # PDFs
        PDFs = [prior,likelihood]

        # Create a sampler
        sampler = pymc.MCMC(PDFs)

        # Make sure we use Metropolis
        sampler.use_step_method(pymc.AdaptiveMetropolis, prior, delay=adaptiveDelay,
                                shrink_if_necessary=True)

        # if dryRun:
        if dryRun:
            self.sampler = sampler
            self.priors = Priors
            self.likelihood = likelihood
            print('A dry run has been asked. Nothing has been done')
            print('Sampler object is saved in the solver object')
            return

        # Sample
        sampler.sample(iter=nSample, burn=nBurn)

        # Recover data
        mpost = []
        samples = {}
        for iprior, prior in enumerate(priors):
            name = prior[0]
            samples[name] = sampler.trace('prior')[:,iprior]
            mpost.append(np.mean(samples[name]))

        # Save things
        self.samples = samples
        self.sampler = sampler
        self.priors = priorFunctions
        self.likelihood = likelihood
        self.mpost = np.array(mpost)

        # Write Samples?
        if writeSamples:
            self.writeSamples2hdf5()

        # Plot
        if plotSampler:
            for iprior, prior in enumerate(priors):
                trace = sampler.trace('prior')[:][:,iprior]
                fig = plt.figure()
                plt.subplot2grid((1,4), (0,0), colspan=3)
                plt.plot([0, len(trace)], [trace.mean(), trace.mean()],
                         '--', linewidth=2)
                plt.plot(trace, 'o-')
                plt.title(prior[0])
                plt.subplot2grid((1,4), (0,3), colspan=1)
                plt.hist(trace, orientation='horizontal')
                plt.savefig('{}.png'.format(prior[0]))
            plt.show()

        # All done
        return

    def conditionCd(self, singularValue):
        '''
        Simple Conditioning of Cd.

        Args:
            * singularValue     : minimum of the kept singular Values

        Returns:
            * None
        '''

        # SVD
        u,s,v = np.linalg.svd(self.Cd)

        # Select
        i = np.flatnonzero(s>singularValue)

        # Re-build
        self.Cd = np.dot(u[:,i], np.dot(np.diag(s[i]), v[i,:]))

        # All done
        return

    def writeSamples2hdf5(self):
        '''
        Writes the result of sampling to and HDF5 file.

        Returns:
            * None
        '''

        # Assert
        assert hasattr(self, 'samples'), 'Needs to have samples to wite them...'

        samples = self.samples

        import h5py
        filename = '{}_samples.h5'.format(self.name.replace(' ','_'))
        fout = h5py.File(filename, 'w')
        for name in samples.keys():
            fout.create_dataset(name, data=samples[name])
        fout.close()

        # All done
        return


    def writeMpost2File(self, outfile):
        '''
        Writes the solution to a file.

        Args:
            * outfile   : Output file name

        Returns:
            * None
        '''

        # Check
        assert (hasattr(self, 'mpost')), 'Compute mpost first, you idiot...'

        # Open file
        fout = open(outfile, 'w')

        # Write header
        fout.write('# Param Number | Mean (mm/yr) | Std (mm/yr) \n')

        # Loop over mpost
        for i in range(self.mpost.shape[0]):
            fout.write('{:3d} {} 0.0000 \n'.format(i, self.mpost[i]))

        # Close file
        fout.close()

        # All done
        return

    def writeMpost2BinaryFile(self, outfile, dtype='d'):
        '''
        Writes the solution to a binary file.

        Args:
            * outfile       : Output file name

        Kwargs:
            * dtype         : 'd' for double and 'f' for single

        Returns:
            * None
        '''

        self.mpost.astype(dtype).tofile(outfile)

        # all done
        return

    def writeGFs2BinaryFile(self, outfile='GF.dat', dtype='f'):
        '''
        Writes the assembled GFs to the file outfile.

        Kwargs:
            * outfile       : Name of the output file.
            * dtype         : Type of data to write. Can be 'f', 'float', 'd' or 'double'.

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = np.float64

        # Convert the data
        G = self.G.astype(dtype)

        # Write to file
        G.tofile(outfile)

        # Keep track of the file
        self.Gfile = outfile

        # Print stuff
        print("Writing Green's functions to file {}".format(outfile))
        print("Green's functions matrix size: {} ; {}".format(G.shape[0], G.shape[1]))

        # All done
        return

    def writeData2BinaryFile(self, outfile='d.dat', dtype='f'):
        '''
        Writes the assembled data vector to an output file.

        Args:
            * outfile       : Name of the output file.
            * dtype         : Type of data to write. Can be 'f', 'float', 'd' or 'double'.

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = np.float64

        # Convert the data
        d = self.d.astype(dtype)

        # Write to file
        d.tofile(outfile)

        # Keep track of the file
        self.dfile = outfile

        # Print stuff
        print("Data vector size: {}".format(d.shape[0]))

        # All done
        return

    def writeCd2BinaryFile(self, outfile='Cd.dat', dtype='f', scale=1.):
        '''
        Writes the assembled Data Covariance matrix to a binary file.

        Args:
            * outfile       : Name of the output file.
            * scale         : Multiply the data covariance.
            * dtype         : Type of data to write. Can be 'f', 'float', 'd' or 'double'.

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = np.float64

        # Convert the data
        Cd = self.Cd.astype(dtype) * scale

        # Write to file
        Cd.tofile(outfile)

        # keep track of the file
        self.Cdfile = outfile

        # print stuff
        print("Data Covariance Size: {} ; {}".format(Cd.shape[0], Cd.shape[1]))

        # All done
        return

    def RunAltar(self, tasks=2, chains=1024, steps=100, support=(-10, 10)):
        '''
        Runs Altar on the d = Gm problem with a Cd covariance matrix.

        Kwargs:
            * tasks         : Number of mpi tasks.
            * chains        : Number of chains.
            * steps         : Number of metropolis steps.
            * support       : Upper and Lower bounds of the parameter exploration.

        Returns:
            * None
        '''

        # Create the cfg and py file
        self.writeAltarCfgFile(prefix=self.name, tasks=tasks, chains=chains, steps=steps, support=support)

        # Create the line
        import subprocess as subp
        com = ['python3.3', self.name+'.py']
        subp.call(com)

        # return
        return

    def writeAltarCfgFile(self, prefix='linearfullcov', tasks=2, chains=1024, steps=100, support=(-10, 10), minimumratio=0.000001):
        '''
        Writes a cfg and a py file to be used by altar.

        Kwargs:
            * outfile       : Prefix of problem
            * tasks         : Number of mpi tasks.
            * chains        : Number of chains.
            * steps         : Number of metropolis steps.
            * support       : Upper and Lower bounds of the parameter exploration.
            * minimumratio  : Minimum Eignevalue to cut in the metropolis covariance matrix.

        Returns:
            * None
        '''

        # Open the file and print the opening credits
        fout = open(prefix+'.cfg', 'w')
        fout.write('; \n')
        fout.write('; R Jolivet \n')
        fout.write('; california institute of technology \n')
        fout.write('; (c) 2010-2013 all rights reserved \n')
        fout.write('; \n')
        fout.write(' \n')

        fout.write('; exercising the sequential controller \n')
        fout.write('[ {} ] \n'.format(prefix))
        fout.write('shell = mpi \n')
        fout.write('model = altar.models.lineargm.linearfullcov \n')
        fout.write('controller = catmip.annealer \n')
        fout.write('rng.algorithm = mt19937 \n')
        fout.write(' \n')

        fout.write('; model configuration \n')
        fout.write('[ altar.models.lineargm.linearfullcov #{}.model ] \n'.format(prefix))
        fout.write('dof = {} \n'.format(self.G.shape[1]))
        fout.write('nd = {} \n'.format(self.G.shape[0]))
        fout.write('support = ({}, {}) \n'.format(support[0], support[1]))
        fout.write('Gfile={} \n'.format(self.Gfile))
        fout.write('dfile={} \n'.format(self.dfile))
        fout.write('covfile={} \n'.format(self.Cdfile))
        fout.write(' \n')

        fout.write('; mpi application shell \n')
        fout.write('[ mpi.shells.mpirun #{}.shell ] \n'.format(prefix))
        fout.write('tasks = {} \n'.format(tasks))
        fout.write(' \n')

        fout.write('; annealing schedule\n')
        fout.write('[ catmip.controllers.annealer #{}.controller ]\n'.format(prefix))
        fout.write('chains = {} \n'.format(chains))
        fout.write('tolerance = .005 \n')
        fout.write('scheduler = cov \n')
        fout.write('sampler = metropolis \n')
        fout.write(' \n')

        fout.write('; metropolis sampler\n')
        fout.write('[ catmip.samplers.metropolis #{}.controller.sampler ]\n'.format(prefix))
        fout.write('steps = {} \n'.format(steps))
        fout.write('scaling = .1 \n')
        fout.write(' \n')

        fout.write('; COV schedule\n')
        fout.write('[ catmip.schedulers.cov #{}.controller.scheduler ] \n'.format(prefix))
        fout.write('tolerance = .01 \n')
        fout.write('MinimumRatio = {} \n'.format(minimumratio))
        fout.write(' \n')

        fout.write('; end of file')

        # Close file
        fout.close()

        # Open the py file
        fout = open(prefix+'.py', 'w')

        # Write things
        fout.write('# -*- coding: utf-8 -*- \n')
        fout.write('# \n')
        fout.write('# R Jolivet \n')
        fout.write('# california institute of technology \n')
        fout.write('# (c) 2010-2013 all rights reserved\n')
        fout.write('# \n')

        fout.write('""" \n')
        fout.write("Exercises the linear Gm model with the full data covariance matrix \n")
        fout.write('""" \n')

        fout.write('def test(): \n')
        fout.write('    # externals \n')
        fout.write('    import catmip \n')
        fout.write('    print(catmip.__file__) \n')
        fout.write('    # instantiate the default application\n')
        fout.write("    app = catmip.application(name='{}')\n".format(prefix))
        fout.write('    # run it\n')
        fout.write('    app.run()\n')
        fout.write('    # and return the app object \n')
        fout.write('    return app\n')

        fout.write('# main \n')
        fout.write('if __name__ == "__main__":\n')
        fout.write('    import journal\n')

        fout.write('    # altar\n')
        fout.write("    journal.debug('altar.beta').active = True\n")
        fout.write("    journal.debug('altar.controller').active = False\n")
        fout.write("    journal.debug('altar.initialization').active = False\n")

        fout.write('    # catmip\n')
        fout.write("    journal.debug('catmip.annealing').active = True\n")

        fout.write('    # do...\n')
        fout.write('    test()\n')

        fout.write('# end of file ')

        # Close file
        fout.close()

        # All done
        return


    def writeAltarPriors(self, priors, params, modelName, files=['static.data.txt','static.Cd.txt','static.gf.txt'], prefix= 'model', chains = 2048):

        '''
        Writes the cfg file containing the priors to be used by altar.
        ONLY works with gaussian and uniform prior. And as if it was not bad enough, initial and run priors are the same.

        Args:
            * priors	    : type of prior and corresponding parameters for each slip mode. ex: prior['Strike Slip']='gaussian' & prior['Dip Slip']='uniform'
	    * params	    : Parameters associated with each type of prior
		 - params['gaussian']=[[center,sigma]]
		 - params['uniform']=[[low,high]]
            * modelName     : Name of the model

        Kwargs:
            * files         : Names of problem files
            * prefix        : Prefix of problem
            * chains        : Number of chains.

        Returns:
            * None

        '''

        parDescr = self.paramDescription

        # Open the file and print the first lines
        fout = open(prefix+'.cfg', 'w')
        fout.write('; -*- ini -*- \n')
        fout.write('; ex: set syntax=dosini: \n')
        fout.write('; \n')
        fout.write('; Generated with CSI \n')
        fout.write('; \n')
        fout.write(' \n')

        # Number of chains and number of parameters
        fout.write(' \n')
        fout.write("Ns = {} \n".format(chains))
        fout.write("Nparam = {} \n".format(self.Np))
        fout.write(' \n')

        # Write init priors
        fout.write('; Priors \n')
        fout.write('; Init Priors \n')
        k = 0
        init_priors = []
        init_cudapriors = []
        for fault in self.faults:
            for slipmode in ['Strike Slip', 'Dip Slip', 'Tensile Slip', 'Extra Parameters']:
                if parDescr[fault.name][slipmode].replace(' ','') != 'None':
                    fout.write("; {} parameters of {} fault segment\n".format(slipmode,fault.name))
                    fout.write("[ init_prior_{} ]\n".format(k))

                    ind = parDescr[fault.name][slipmode].replace(' ','').partition('-')
                    idx_begin = int(ind[0])
                    idx_end = int(ind[2])
                    fout.write("idx_begin = {}\n".format(idx_begin))
                    fout.write("idx_end = {}\n".format(idx_end))

                    if priors[slipmode] == 'gaussian':
                        fout.write("center = {}\n".format(params['gaussian'][0]))
                        fout.write("sigma = {}\n".format(params['gaussian'][1]))
                        fout.write(' \n')
                        init_priors.append("altar.priors.gaussian.Gaussian#init_prior_{}".format(k))
                        init_cudapriors.append("altar.priors.gaussian.cudaGaussian#init_prior_{}".format(k))

                    else:
                        fout.write("low = {}\n".format(params['uniform'][0]))
                        fout.write("high = {}\n".format(params['uniform'][1]))
                        fout.write(' \n')
                        init_priors.append("altar.priors.uniform.Uniform#init_prior_{}".format(k))
                        init_cudapriors.append("altar.priors.uniform.cudaUniform#init_prior_{}".format(k))

                    k +=1

                else:
                    continue


        # Write run priors
        fout.write('; Run Priors \n')
        k = 0
        run_priors = []
        run_cudapriors = []
        for fault in self.faults:
            for slipmode in ['Strike Slip', 'Dip Slip', 'Tensile Slip', 'Extra Parameters']:
                if parDescr[fault.name][slipmode].replace(' ','') != 'None':
                    fout.write("; {} parameters of {} fault segment\n".format(slipmode,fault.name))
                    fout.write("[ run_prior_{} ]\n".format(k))

                    ind = parDescr[fault.name][slipmode].replace(' ','').partition('-')
                    idx_begin = int(ind[0])
                    idx_end = int(ind[2])
                    fout.write("idx_begin = {}\n".format(idx_begin))
                    fout.write("idx_end = {}\n".format(idx_end))

                    if priors[slipmode] == 'gaussian':
                        fout.write("center = {}\n".format(params['gaussian'][0]))
                        fout.write("sigma = {}\n".format(params['gaussian'][1]))
                        fout.write(' \n')
                        run_priors.append("altar.priors.gaussian.Gaussian#run_prior_{}".format(k))
                        run_cudapriors.append("altar.priors.gaussian.cudaGaussian#run_prior_{}".format(k))

                    else:
                        fout.write("low = {}\n".format(params['uniform'][0]))
                        fout.write("high = {}\n".format(params['uniform'][1]))
                        fout.write(' \n')
                        run_priors.append("altar.priors.uniform.Uniform#run_prior_{}".format(k))
                        run_cudapriors.append("altar.priors.uniform.cudaUniform#run_prior_{}".format(k))

                    k +=1

                else:
                    continue

        # Write models parameters
        fout.write("; Models\n")
        fout.write("[ {} ]\n".format(modelName))
        fout.write("parameters = {Nparam}\n")
        fout.write("observations = {}\n".format(self.Nd))
        fout.write("data_file = {{modelDir}}/{}\n".format(files[0]))
        fout.write("Cd_file = {{modelDir}}/{}\n".format(files[1]))
        fout.write("gf_file = {{modelDir}}/{}\n".format(files[2]))
        fout.write("\n")

        # Write Problem
        fout.write("; Problems\n")
        fout.write("[ altar.problem.Problem # problem ] ; if problem is a Problem\n")
        fout.write("init_priors = {}\n".format(','.join(init_priors)))
        fout.write("run_priors = {}\n".format(','.join(run_priors)))
        fout.write("models = altar.models.linear.Linear#{}\n".format(modelName))

        fout.write("\n")
        fout.write("[ altar.problem.cudaProblem # problem ] ; if problem is a cudaProblem\n")
        fout.write("init_priors = {}\n".format(','.join(init_cudapriors)))
        fout.write("run_priors = {}\n".format(','.join(run_cudapriors)))
        fout.write("models = altar.models.linear.cudaLinear#{}\n".format(modelName))

        # All done
        return

    def writePatchAreasFile(self, outfile='PatchAreas.dat', dtype='d',
                            npadStart=None, npadEnd=None):
        """
        Write a binary file for the patch areas to be read into altar.

        Kwargs:
            * outfile               : output file name
            * dtype                 : output data type
            * npadStart             : number of starting zeros to pad output
            * npadEnd               : number of ending zeros to pad output

        Returns:
            * None
        """
        # Construct output vector of patch areas
        vout = self.patchAreas.astype(dtype)
        if npadStart is not None:
            vpad = np.zeros((npadStart,), dtype=dtype)
            vout = np.hstack((vpad, vout))
        if npadEnd is not None:
            vpad = np.zeros((npadEnd,), dtype=dtype)
            vout = np.hstack((vout, vpad))

        # Write to file and return
        vout.tofile(outfile)
        return

#EOF
