'''
A Class to assemble several sources (fault, pressure, or block) into a single inverse problem.
All the sources must have been initialized and constructed using the same data set.
This class allows then to:
    1. Spit the G, m, Cm, and Cd elements for a third party solver (such as Altar, for instance)
    2. Proposes a simple solution based on a least-square optimization.

Written by R. Jolivet, April 2013.

Updated by T. Shreve, May 2019, to include pressure sources in describeParams and distributem.

Updated by E. Denise, March 2025, to include blocks and multiblocks.

'''

import copy
import numpy as np
import matplotlib.pyplot as plt
try:
    import h5py
except:
    print('HDF5 capabilities not available')

from .eulerPoleUtils import MAS2RAD

class multisourcesolve(object):
    '''
    A class that assembles the linear inverse problem for multiple sources (fault, block, or pressure) and multiple datasets.
    This class can also solve the problem using simple linear least squares (bounded or unbounded).

    Args:
        * name          : Name of the project.
        * faults        : List of faults from verticalfault, pressure or block.

    '''

    def __init__(self, name, sources, verbose=True):

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
        self.sources = sources

        # check the utm zone
        self.utmzone = sources[0].utmzone
        for source in sources:
            if source.utmzone is not self.utmzone:
                print("UTM zones are not equivalent, this is a problem")
                self.ready = False
                return
        self.xy2ll = sources[0].xy2ll
        self.ll2xy = sources[0].ll2xy

        # check that G and d have been assembled prior to initialization
        for source in sources:
            if source.Gassembled is None:
                self.ready = False
                print("G has not been assembled in source structure {}".format(source.name))
            if source.dassembled is None:
                self.ready = False
                print("d has not been assembled in source structure {}".format(source.name))

        # Check that the sizes of the data vectors are consistent
        self.d = sources[0].dassembled
        for source in sources:
            if (source.dassembled != self.d).all():
                print("Data vectors are not consistent, please re-consider your data in source structure {}".format(source.name))

        # Check that the data covariance matrix is the same
        self.Cd = sources[0].Cd
        for source in sources:
            if (source.Cd != self.Cd).all():
                print("Data Covariance Matrix are not consistent, please re-consider your data in source structure {}".format(source.name))

        # Initialize things
        self.source_indexes = None

        # Store an array of the patch areas
        patchAreas = []
        
        for source in sources:
            
            if source.type == "Fault":
                if source.patchType == 'triangletent':
                    source.computeTentArea()
                    for tentIndex in range(source.slip.shape[0]):
                        patchAreas.append(source.area_tent[tentIndex])
                else:
                    source.computeArea()
                    for patchIndex in range(source.slip.shape[0]):
                        patchAreas.append(source.area[patchIndex])
                self.patchAreas = np.array(patchAreas)
                
        # All done
        return

    def assembleGFs(self):
        '''
        Assembles the Green's functions matrix G for the deformation sources (faults, blocks, pressure sources, ...).

        Returns:
            * None
        '''

        # Get the sources
        sources = self.sources

        # Get the size of the total G matrix
        Nd = self.d.size
        Np = 0
        st = []
        se = []
        if self.source_indexes is None:
            self.source_indexes = {}
        for source in sources:
            st.append(Np)
            Np += source.Gassembled.shape[1]
            se.append(Np)
            self.source_indexes[source.name] = [st[-1], se[-1]]

        # Allocate the big G matrix
        self.G = np.zeros((Nd, Np))

        # Store the guys
        for source in sources:
            # get the good indexes
            st = self.source_indexes[source.name][0]
            se = self.source_indexes[source.name][1]
            # Store the G matrix
            self.G[:,st:se] = source.Gassembled
            # Keep track of indexing
            if source.type == "Fault":
                self.affectFaultIndexParameters(source)

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
        self.describeParams()

        # All done
        return

    def equalizeParams(self, iparams, Cm=None):
        '''
        This is a step to force parameters to be equal. Effectively, since the 
        problem is linear, we sum columns of G to have a single parameter. The 
        parameter in question is set at the end of G. 
        Cm is modified so that it has 1 on the idagonal or what is provided in Cm
        
        The original G is saved as Goriginal. The original Cm is in Cmoriginal.
        The method distributem accounts for such modification by restoring G and Cm
        and the mpost vector according to the original problem.

        iparams is a list of list of groups of parameters:
            iparams = [ [1,2,3,19,39], [23, 24]]

        Args:
            * iparams: List of lists

        Kwargs:
            * Cm     : List of covariances
        '''

        # Problem must be assembled first, otherwise it is a mess
        assert self.ready, 'Must assemble the problem first'
        
        # Check
        if hasattr(self, 'equalized'):
            print('Beware: The problem already has an equalization.')

        # Save original problem
        if not hasattr(self, 'equalized'):
            self.Goriginal = copy.deepcopy(self.G)
            self.Cmoriginal = copy.deepcopy(self.Cm)

        # Check format of iparams
        if type(iparams[0]) is not list:
            iparams = [iparams]
        for ipar in iparams:
            assert type(ipar) is list, 'Elements of iparams must be lists'
            for i in ipar: 
                assert type(i) in (int, np.int64), 'Indexes in iparams must be int: {}'.format(type(i))

        # Remove the columns in G
        allpars = [i for ipar in iparams for i in ipar]
        self.G = np.delete(self.G, allpars, axis=1)
        self.paramTypes = np.delete(self.paramTypes, allpars)
        self.Cm = np.delete(self.Cm, allpars, axis=1)
        self.Cm = np.delete(self.Cm, allpars, axis=0)

        # Iterate over the groups of parameters to equalize
        if not hasattr(self, 'equalized'):
            self.equalized = {}
        
        for i, ipar in enumerate(iparams):
            
            # Update paramTypes
            p = np.array([None for i in range(len(self.paramTypes)+1)])
            p[:-1] = self.paramTypes
            p[-1] = ('Equalized', ipar)
            self.paramTypes = p
            
            # Keep track of the equalization
            self.equalized[self.G.shape[1]] = ipar
            
            # Update G
            newG = np.zeros((self.G.shape[0], self.G.shape[1]+1))
            newG[:self.G.shape[0], :self.G.shape[1]] = self.G
            newG[:, -1] = self.Goriginal[:,ipar].sum(axis=1)
            self.G = newG
            
            # Update Cm
            Cmnew = np.zeros((self.Cm.shape[0]+1,self.Cm.shape[1]+1))
            Cmnew[:self.Cm.shape[0],:self.Cm.shape[0]] = self.Cm
            if Cm is not None:
                Cmnew[-1, -1] = Cm[i]
            else:
                Cmnew[-1, -1] = 1.
            self.Cm = Cmnew

        # Build mapping between mnew and mpost
        eye = np.eye(self.Goriginal.shape[1])
        eye = np.delete(eye, allpars, axis=1)
        eye[allpars,:] = 0.
        mapping = np.zeros((self.Goriginal.shape[1], self.G.shape[1]))
        mapping[:,:eye.shape[1]] = eye
        for eq in self.equalized:
            ipar = self.equalized[eq]
            mapping[ipar,eq] = 1.
        self.mapping_new2old = mapping

        # Change things
        self.Np = self.G.shape[1]

        # Change the parameter description thing
        self.paramDescription = {}
        couples, inverse = np.unique(self.paramTypes, return_inverse=True)
        for icouple,couple in enumerate(couples):
            uu = np.flatnonzero(inverse==icouple)
            if len(uu)>0:
                if 'Equalized' not in couple:
                    source = couple[0]
                    component = couple[1]
                    if source not in self.paramDescription: self.paramDescription[source] = {}
                    ss = '{:12s}'.format('{:4d} - {:4d}'.format(uu[0], uu[-1]+1))
                    self.paramDescription[source][component] = ss
                else:
                    if 'Equalized' not in self.paramDescription: self.paramDescription['Equalized'] = []
                    self.paramDescription['Equalized'].append([uu[0], couple[1]])
                    
        # All done
        return 
    
    def equalizeParams2(self, iparams, Cm=None):
        '''
        This is a step to force parameters to be equal. Effectively, since the 
        problem is linear, we sum columns of G to have a single parameter. The 
        parameter in question is set at the position of the first parameter in the group.
        Cm is modified so that it has what is provided in Cm if provided (i.e. not None),
        otherwise what corresponds to the original Cm for the first parameter in the group.
        
        The original G is saved as Goriginal. The original Cm is in Cmoriginal.
        The method distributem accounts for such modification by restoring G and Cm
        and the mpost vector according to the original problem.
        
        Beware: if a an equalization has already been performed, you need to provide the indexes of the parameters in the new problem.
        You can use the original indexes of the parameters in the original problem, and then call self.equalized to get the updated parameter indexes.
        
        iparams is a list of list of groups of parameters:
            iparams = [ [1,2,3,19,39], [23, 24]]

        Args:
            * iparams: List of lists

        Kwargs:
            * Cm     : List of covariances
        '''

        assert self.ready
        if hasattr(self, 'equalized'):
            print('Beware: The problem already has an equalization.')

        if not hasattr(self, 'equalized'):
            self.Goriginal = copy.deepcopy(self.G)
            if hasattr(self, 'Cm'):
                self.Cmoriginal = copy.deepcopy(self.Cm)
            n_orig = self.Goriginal.shape[1]
            self.mapping_old2new = np.eye(n_orig)
            self.mapping_new2old = np.eye(n_orig)

        if type(iparams[0]) is not list:
            iparams = [iparams]

        ncur = self.G.shape[1]   # current size, BEFORE this round's deletion

        newG = copy.deepcopy(self.G)
        for i, ipar in enumerate(iparams):
            newG[:, ipar[0]] = self.G[:, ipar].sum(axis=1)
        
        if hasattr(self, 'Cm'):
            newCm = copy.deepcopy(self.Cm)
            for i, ipar in enumerate(iparams):
                newCm[ipar[0], ipar[0]] = Cm[i] if Cm is not None else self.Cm[ipar[0], ipar[0]]

        allpars = [i for ipar in iparams for i in ipar[1:]]
        self.paramTypes = np.delete(self.paramTypes, allpars)
        self.G  = np.delete(newG,  allpars, axis=1)
        if hasattr(self, 'Cm'):
            self.Cm = np.delete(newCm, allpars, axis=1)
            self.Cm = np.delete(self.Cm, allpars, axis=0)

        # Round mapping: old-current index -> new-current index (or None if merged away)
        round_map = {}
        for c in range(ncur):
            if c in allpars:
                round_map[c] = None
            else:
                round_map[c] = c - len([a for a in allpars if a < c])

        n_new_cur = self.G.shape[1]
        round_old2new = np.zeros((n_new_cur, ncur))   # reduction, this round only
        round_new2old = np.zeros((ncur, n_new_cur))   # reconstruction, this round only

        for c in range(ncur):
            if round_map[c] is not None:
                round_old2new[round_map[c], c] = 1
                round_new2old[c, round_map[c]] = 1

        # Merged-away members reconstruct to the same column as their group's representative
        for ipar in iparams:
            new_idx = round_map[ipar[0]]
            for c in ipar[1:]:
                round_new2old[c, new_idx] = 1

        # Compose with whatever mapping already exists (from this fn or equalizeParamsBlockModelFault),
        # preserving any non-binary weights already present in mapping_new2old.
        self.mapping_old2new = round_old2new @ self.mapping_old2new
        self.mapping_new2old = self.mapping_new2old @ round_new2old

        # equalized/unequalized are derived from mapping_old2new, which always stays
        # a clean 0/1 selection matrix (at most one 1 per column), even when
        # mapping_new2old carries weighted substitution rows.
        self.equalized = {}
        for k in range(self.Goriginal.shape[1]):
            nz = np.nonzero(self.mapping_old2new[:, k])[0]
            self.equalized[k] = int(nz[0]) if len(nz) else None
        self.unequalized = {v: k for k, v in self.equalized.items() if v is not None}

        # --- Np, paramDescription unchanged ---

        # source_indexes update uses round_map (current-space only, unaffected by
        # whatever happened in earlier original-space equalizations)
        for src in self.source_indexes:
            i0 = self.source_indexes[src][0]
            i1 = self.source_indexes[src][1]
            if round_map[i0] is not None:
                i0 = round_map[i0]
            else:
                i = i0
                while round_map[i] is None:
                    i += 1
                i0 = round_map[i]
            if round_map[i1-1] is not None:
                i1 = round_map[i1-1] + 1
            else:
                i = i1 - 1
                while round_map[i] is None:
                    i -= 1
                i1 = round_map[i] + 1
            self.source_indexes[src] = [i0, i1]
        
        # Update Np
        self.Np = self.G.shape[1]
        
        # Change the parameter description thing
        self.paramDescription = {}
        couples, inverse = np.unique(self.paramTypes, return_inverse=True)
        for icouple,couple in enumerate(couples):
            uu = np.flatnonzero(inverse==icouple)
            if len(uu)>0:
                source = couple[0]
                component = couple[1]
                if source not in self.paramDescription: self.paramDescription[source] = {}
                ss = '{:12s}'.format('{:4d} - {:4d}'.format(uu[0], uu[-1]+1))
                self.paramDescription[source][component] = ss
                    
    def equalizeParamsBlockModelFault(self, multiblock, faults):
        '''
        This is a step to force slip parameters of a fault to be equal to long term slip rate
        imposed from a block model (MultiBlock object).
        The slip parameters are the bottom patches/tents of the fault.
        
        Effectively, somme slip parameters are removed from G and their contribution is added
        to the rotation parameters. Cm is modified.
        
        The original G is saved as Goriginal. The original Cm is in Cmoriginal.
        The method distributem accounts for such modification by restoring G and Cm
        and the mpost vector according to the original problem.
        
        You can use self.unequalizeParams() to restore the original problem after the inversion is done.
        
        Beware: this function has not been tested when you don't invert the block rotation parameters for one of the block.
        Beware: this function has not been tested with several faults.

        Args:
            * multiblock     : MultiBlock object
            * faults         : List of Fault objects (which must be block boundaries of the multiblock model)

        '''
        
        # Problem must be assembled first, otherwise it is a mess
        assert self.ready, 'Must assemble the problem first'
        
        # Check list fault
        if type(faults) is not list:
            faults = [faults]
        
        # Check the types of the arguments
        assert multiblock.type == "MultiBlock", 'multiblock must be a MultiBlock object'
        assert all(f.type == "Fault" for f in faults), 'faults must be a list of Fault objects'
        
        # Check that the multiblock and faults are in the sources
        assert multiblock.name in [s.name for s in self.sources], 'multiblock must be in sources'
        assert all(f.name in [s.name for s in self.sources] for f in faults), 'all faults must be in sources'
        
        # Check that the faults are block boundaries of the multiblock model
        assert all(f.name in [b.name for b in multiblock.blockboundaries] for f in faults), 'all faults must be block boundaries of the multiblock model'
        
        # Save original G and Cm for later use
        self.Goriginal = copy.deepcopy(self.G)
        if hasattr(self, 'Cm'):
            self.Cmoriginal = copy.deepcopy(self.Cm)
        
        for fault in faults:
        
            if fault.patchType in ('rectangle', 'triangle'):
                
                raise NotImplementedError('This method is only implemented for tent patches for now.')

            elif fault.patchType == 'triangletent':
                
                # Get the slip components
                mask_comp = []
                if 's' in fault.slipdir: mask_comp += [0]
                if 'd' in fault.slipdir: mask_comp += [1]
                if 't' in fault.slipdir: mask_comp += [2]

                # Get bottom tents parameters
                maxDepth = np.array(fault.tent)[:, 2].max()
                itents= [i for i, t in enumerate(fault.tent) if np.abs(t[-1] - maxDepth) < 1e-3]
                iparams = [self.source_indexes[fault.name][0]+i+ic*fault.N_slip for i in itents for ic in range(len(mask_comp))]

                # Number of rotation parameters for the global block model
                mask_rot = np.repeat(['rotation' in multiblock.blockcomponent[b.name] or 'boundary' in multiblock.blockcomponent[b.name] for b in multiblock.blocks], 3)
                Nprot = sum(mask_rot)

                # Update G (get boundary GF for the bottom patches)
                multiblock.GetLongTermSlipRate()

                # Extract Green's functions for bottom patches and active components
                Gr = fault.Glongterm_slip[itents, :, :][:, mask_comp, :]

                # Reshape to (n_bottom_patches * n_components, n_block_params)
                Gr_reshaped = Gr.reshape(-1, Gr.shape[-1])

                # Select only rotation parameters
                Gr_rot = Gr_reshaped[:, mask_rot]

                # Get the fault Green's functions for bottom patches
                # Need to build index array correctly
                bottom_indices = []
                for ic in range(len(mask_comp)):
                    for it in itents:
                        bottom_indices.append(it + ic * fault.N_slip)

                Gf = fault.Gassembled[:, bottom_indices]

                # Compute the contribution of block rotations to data
                G_ = Gf @ Gr_rot * MAS2RAD

                # Add this contribution to the block rotation columns in G
                self.G[:, self.source_indexes[multiblock.name][0]:self.source_indexes[multiblock.name][0]+Nprot] += G_

                # Remove the columns in G corresponding to the bottom patches parameters, and the corresponding lines and columns in Cm
                self.G = np.delete(self.G, iparams, axis=1)
                self.paramTypes = np.delete(self.paramTypes, iparams)
                if hasattr(self, 'Cm'):
                    self.Cm = np.delete(self.Cm, iparams, axis=1)
                    self.Cm = np.delete(self.Cm, iparams, axis=0)

                # Keep track of the equalization
                self.equalized = {}
                self.unequalized = {}

                for k in range(self.Goriginal.shape[1]):
                    if k in iparams:
                        self.equalized[k] = None
                    else:
                        self.equalized[k] = k - len([i for i in iparams if i < k])
                        self.unequalized[k - len([i for i in iparams if i < k])] = k

                # Create the mapping from new parameters to old parameters (for equalization)
                self.mapping_new2old = np.zeros((self.Goriginal.shape[1], self.G.shape[1]))

                for k in self.equalized:
                    if self.equalized[k] is not None:
                        self.mapping_new2old[k, self.equalized[k]] = 1

                # Map the bottom patch parameters to block rotation parameters
                rot_param_indices = list(range(self.source_indexes[multiblock.name][0],
                                                self.source_indexes[multiblock.name][0] + Nprot))
                rot_param_indices_new = [self.equalized[k] for k in rot_param_indices if self.equalized[k] is not None]

                for idx, iparam in enumerate(iparams):
                    self.mapping_new2old[iparam, rot_param_indices_new] = Gr_rot[idx, :] * MAS2RAD

                # Create the mapping from original parameters to new parameters (for equalization)
                self.mapping_old2new = np.zeros((self.G.shape[1], self.Goriginal.shape[1]))

                for k in self.equalized:
                    if self.equalized[k] is not None:
                        self.mapping_old2new[self.equalized[k], k] = 1

                # Change things
                self.Np = self.G.shape[1]

                # Change the parameter description thing
                self.paramDescription = {}
                couples, inverse = np.unique(self.paramTypes, return_inverse=True)
                for icouple,couple in enumerate(couples):
                    uu = np.flatnonzero(inverse==icouple)
                    if len(uu)>0:
                        if 'Equalized' not in couple:
                            source = couple[0]
                            component = couple[1]
                            if source not in self.paramDescription:
                                self.paramDescription[source] = {}
                            ss = '{:12s}'.format('{:4d} - {:4d}'.format(uu[0], uu[-1]+1))
                            self.paramDescription[source][component] = ss
                        else:
                            if 'Equalized' not in self.paramDescription:
                                self.paramDescription['Equalized'] = []
                            self.paramDescription['Equalized'].append([uu[0], couple[1]])

                # Update source indexes
                for src in self.source_indexes:
                    
                    i0 = self.source_indexes[src][0]
                    i1 = self.source_indexes[src][1]
                    
                    if self.equalized[i0] is not None:
                        i0 = self.equalized[i0]
                    else:
                        i = i0
                        while self.equalized[i] is None:
                            i += 1
                        i0 = self.equalized[i]
                    
                    if self.equalized[i1-1] is not None:
                        i1 = self.equalized[i1-1]+1
                    else:
                        i = i1-1
                        while self.equalized[i] is None:
                            i -= 1
                        i1 = self.equalized[i]+1
                    
                    self.source_indexes[src] = [i0, i1]

                print(f"Equalization: removing {len(iparams)} fault slip parameters from fault {fault.name}.")

        # All done
        return
    
    def unequalizeParams(self):
        '''
        Restores the shape of G and Cm and organizes mpost accordingly whem the 
        problem has been altered by equalizedParams.
        '''

        # Check
        assert hasattr(self, 'equalized'), 'Cannot unequalize if equalizedParams or equalizeParamsBlockModelFault were not used.'

        # Restore G
        self.G = copy.deepcopy(self.Goriginal)
        del self.Goriginal
        if hasattr(self, 'Cmoriginal'):
            self.Cm = copy.deepcopy(self.Cmoriginal)
            del self.Cmoriginal

        # Reorganize mpost
        msave = copy.deepcopy(self.mpost)
        self.mpost = self.mapping_new2old @ msave
        self.Np = len(self.mpost)
        del msave
        
        # Restore source indexes
        Nd = self.d.size
        Np = 0
        st = []
        se = []
        self.source_indexes = {}
        for source in self.sources:
            st.append(Np)
            Np += source.Gassembled.shape[1]
            se.append(Np)
            self.source_indexes[source.name] = [st[-1], se[-1]]

        # Parameter Description
        self.makeParamDescription()

        # All done
        return
    
    def strongConstraint(self, iparams, cov=1e-6):
        '''
        Adds a bunch of lines to force the parameters {iparams} to
        be equal, within {cov}. Effectively, it adds a line of +1 and -1
        to the parameters so that all {iparams} are equal to the first one.
        The equality will fall within {cov} as this number is set as the diagonal
        term of the data covariance for the corresponding lines.

        Args:
            * iparams       : List of parameters
        
        Kwargs:
            * cov           : Covariance

        '''

        # Number of constraints
        nc = len(iparams) - 1 

        # Create the new lines
        Glines = np.zeros((nc, self.G.shape[1]))
        dlines = np.zeros((nc,))

        # Iterate 
        for i,ip in enumerate(iparams[1:]):
            Glines[i,iparams[0]] = 1.
            Glines[i,ip] = -1.

        # Concatenate
        self.G = np.concatenate((self.G, Glines))
        self.d = np.concatenate((self.d, dlines))

        # Expand Cd 
        self.Cd = np.concatenate((self.Cd, np.zeros((self.Cd.shape[0], nc))), axis=1)
        self.Cd = np.concatenate((self.Cd, np.zeros((nc, self.Cd.shape[1]))), axis=0)
        cc = np.eye(nc)*cov
        self.Cd[-nc:,-nc:] = cc

        # Update 
        self.Nd = len(self.d)

        # All done
        return

    def OrganizeGBySlipmode(self):

        '''
        Organize G by slip mode instead of fault segment Return the new G matrix.

        Returns:
            * array

        '''

        assert len(self.sources) !=1, 'You have only one fault, why would you want to do that?'
        assert self.ready, 'You need to assemble the GFs before'

        info = self.paramDescription

        Gtemp = np.zeros((self.G.shape))

        N = 0
        slipmode = ['Strike Slip', 'Dip Slip', 'Tensile Slip', 'Coupling', 'Extra Parameters']
        for mode in slipmode:
            for source in self.sources:
                if info[source.name][mode].replace(' ','') != 'None':
                    ib = int(info[source.name][mode].replace(' ','').partition('-')[0])
                    ie = int(info[source.name][mode].replace(' ','').partition('-')[2])
                    Gtemp[:,N:N+ie-ib] = self.G[:,ib:ie]
                    N += ie-ib

        return Gtemp
    
    def sensitivity(self, rcond=None):
        '''
        Calculates the sensitivity matrix of the problem, :math:`\\textbf{S} = \\text{diag}( \\textbf{G}^t \\textbf{C}_d^{-1} \\textbf{G} )`
        
        Args:
            * rcond        : Cutoff for the pseudo-inverse

        Returns:
            * array         : Sensitivity matrix
        '''
        
        Cd = self.Cd
        G = self.G
        
        # Invert
        if hasattr(self, 'iCd'):
            iCd = self.iCd
        else:
            if rcond is None:
                iCd = np.linalg.inv(Cd)
            else:
                iCd = np.linalg.pinv(Cd, rcond=rcond)
            self.iCd = iCd
        
        # Calculate the sensitivity
        S = np.diag(G.T @ iCd @ G)

        # All done
        return S

    def resolution(self, rcond=None):
        '''
        Calculates the resolution matrix of the problem, :math:`\\textbf{R} = \\textbf{C}_m \\textbf{G}^t ( \\textbf{G} \\textbf{C}_m \\textbf{G}^t + \\textbf{C}_d )^{-1} \\textbf{G}`
        
        Args:
            * rcond        : Cutoff for the pseudo-inverse
        
        Returns:
            * array         : Resolution matrix
        '''
        G = self.G
        Cm = self.Cm
        Cd = self.Cd
        
        # Invert
        A = G @ Cm @ G.T + Cd
        if rcond is None:
            iA = np.linalg.inv(A)
        else:
            iA = np.linalg.pinv(A, rcond=rcond)
        
        # Calculate the resolution matrix
        R = Cm @ G.T @ iA @ G
        
        # All done
        return R

    def describeParams(self, redo=True):
        '''
        Print the parameter description.

        Returns:
            * None
        '''

        # Create the parameter description
        if redo:
            self.makeParamDescription()

        # Get the sources
        sources = self.sources

        if self.verbose:
            print('Parameter Description ----------------------------------')

        # Loop over the param description
        for source in sources:
            
            if source.name not in self.paramDescription.keys() and source.type != 'MultiBlock':
                continue

            # Fault
            if source.type == 'Fault':
                
                description = self.paramDescription[source.name]

                #Prepare the table
                if self.verbose:
                    print('-----------------')
                    print('{:30s}||{:12s}||{:12s}||{:12s}||{:12s}||{:12s}'.format('Fault Name', 'Strike Slip', 'Dip Slip', 'Tensile Slip', 'Coupling', 'Extra Parms'))

                # Get info
                if 'Strike Slip' in description:
                    ss = description['Strike Slip']
                else:
                    ss = 'None'
                if 'Dip Slip' in description:
                    ds = description['Dip Slip']
                else:
                    ds = 'None'
                if 'Tensile Slip' in description:
                    ts = description['Tensile Slip']
                else:
                    ts = 'None'
                if 'Coupling' in description:
                    cp = description['Coupling']
                else:
                    cp = 'None'
                if 'Extra Parameters' in description:
                    op = description['Extra Parameters']
                else:
                    op = 'None'

                # print things
                if self.verbose:
                    print('{:30s}||{:12s}||{:12s}||{:12s}||{:12s}||{:12s}'.format(source.name, ss, ds, ts, cp, op))

            # Pressure
            elif source.type == 'Pressure':
                
                description = self.paramDescription[source.name]

                #Prepare the table
                if self.verbose:
                    print('-----------------')
                    print('{:30s}||{:12s}||{:12s}'.format('Pressure Name', 'Pressure', 'Extra Parms'))

                # Get info
                if 'Pressure' in description:
                    dp = description['Pressure']
                else:
                    dp = 'None'
                if 'Extra Parameters' in description:
                    op = description['Extra Parameters']
                else:
                    op = 'None'

                # print things
                if self.verbose:
                    print('{:30s}||{:12s}||{:12s}'.format(source.name, dp, op))
            
            # Block 
            elif source.type == 'Block':
                
                description = self.paramDescription[source.name]
                
                #Prepare the table
                if self.verbose:
                    print('-----------------')
                    print('{:30s}||{:12s}||{:12s}||{:12s}'.format('Block Name', 'Rotation', 'Intra def.', 'Extra Parms'))

                # Get info
                if 'Rotation' in description:
                    prot = description['Rotation']
                else:
                    prot = 'None'
                if 'Intra def.' in description:
                    pint = description['Intra def.']
                else:
                    pint = 'None'
                if 'Extra Parameters' in description:
                    pext = description['Extra Parameters']
                else:
                    pext = 'None'

                # print things
                if self.verbose: print('{:30s}||{:12s}||{:12s}||{:12s}'.format(source.name, prot, pint, pext))
            
            # MultiBlock
            elif source.type == 'MultiBlock':
                
                for source_block in source.blocks:
                    
                    if source_block.name not in self.paramDescription.keys():
                        continue
                
                    description = self.paramDescription[source_block.name]
                    
                    #Prepare the table
                    if self.verbose:
                        print('-----------------')
                        print('{:30s}||{:12s}||{:12s}||{:12s}'.format('Block Name', 'Rotation', 'Intra def.', 'Extra Parms'))
                    
                    # Get info
                    if 'Rotation' in description:
                        prot = description['Rotation']
                    else:
                        prot = 'None'
                    if 'Intra def.' in description:
                        pint = description['Intra def.']
                    else:
                        pint = 'None'
                    if 'Extra Parameters' in description:
                        pext = description['Extra Parameters']
                    else:
                        pext = 'None'

                    # print things
                    if self.verbose: print('{:30s}||{:12s}||{:12s}||{:12s}'.format(source_block.name, prot, pint, pext))
    
            # SurfaceMotion
            elif source.type == 'Surface':
                
                description = self.paramDescription[source.name]
                
                #Prepare the table
                if self.verbose:
                    print('-----------------')
                    print('{:30s}||{:12s}||{:12s}'.format('Surface Name', 'Surface', 'Extra Parms'))

                # Get the size
                dp = description['Surface']

                # print things
                if self.verbose: print('{:30s}||{:12s}||{:12s}'.format(source.name, dp, 'None'))
            
            # Transformation
            elif source.type == 'transformation':
                
                description = self.paramDescription[source.name]
                
                #Prepare the table
                if self.verbose:
                    print('-----------------')
                    print('{:30s}||{:12s}||{:12s}'.format('Transformation Name', 'Transfo', 'Extra Parms'))

                if 'Transfo' in description:
                    op = description['Transfo']
                elif 'Extra Parameters' in description:
                    op = description['Extra Parameters']
                else:
                    op = 'None'

                # print things
                if self.verbose:
                    print('{:30s}||{:12s}||{:12s}'.format(source.name, op, 'None'))

        if 'Equalized' in self.paramDescription:
            for case in self.paramDescription['Equalized']:
                new,old = case
                if self.verbose:
                    print('-----------------') 
                    print('Equalized parameter indexes: {} --> {}'.format(old,new))

        # all done
        return

    def makeParamDescription(self):
        '''
        Store what parameters mean

        Returns:
            * None
        '''

        sources = self.sources

        # initialize the counters
        ns = 0
        ne = 0
        nSlip = 0

        # Store that somewhere
        self.paramDescription = {}

        # Make a list of parameter types
        self.paramTypes = np.array([None for i in range(self.Np)])

        # Loop over the sources
        for source in sources:

            # Where does this source starts
            nfs = copy.deepcopy(ns)

            if source.type == "Fault":
                # Initialize the values
                ss = 'None'
                ds = 'None'
                ts = 'None'
                cp = 'None'

                # Conditions on slip
                if 's' in source.slipdir:
                    ne += source.slip.shape[0]
                    ss = '{:12s}'.format('{:4d} - {:4d}'.format(ns,ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Strike Slip')
                    ns += source.slip.shape[0]
                if 'd' in source.slipdir:
                    ne += source.slip.shape[0]
                    ds = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Dip Slip')
                    ns += source.slip.shape[0]
                if 't' in source.slipdir:
                    ne += source.slip.shape[0]
                    ts = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Tensile Slip')
                    ns += source.slip.shape[0]
                if 'c' in source.slipdir:
                    ne += source.slip.shape[0]
                    cp = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Coupling')
                    ns += source.slip.shape[0]

                # How many slip parameters
                if ne>nSlip:
                    nSlip = ne

                # conditions on orbits (the rest is orbits)
                npo = ne - nfs
                no = source.Gassembled.shape[1] - npo
                if no>0:
                    ne += no
                    op = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Extra Parameters')
                    ns += no
                else:
                    op = 'None'

                # Store
                self.paramDescription[source.name] = {}
                self.paramDescription[source.name]['Strike Slip'] = ss
                self.paramDescription[source.name]['Dip Slip'] = ds
                self.paramDescription[source.name]['Tensile Slip'] = ts
                self.paramDescription[source.name]['Coupling'] = cp
                self.paramDescription[source.name]['Extra Parameters'] = op
            
            elif source.type == "transformation":
                
                # conditions on orbits (the rest is orbits)
                no = source.Gassembled.shape[1]
                
                if no>0:
                    ne += no
                    op = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Extra Parameters')
                    ns += no
                else:
                    op = 'None'

                # Store
                self.paramDescription[source.name] = {}
                self.paramDescription[source.name]['Transfo'] = op

            elif source.type == "Pressure":

                # Initialize the values
                dp = 'None'
                if source.source=="pCDM":
                    ne += 3
                    dp = '{:12s}'.format('{:4d} - {:4d}'.format(ns,ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Pressure')
                    ns += 3 #fault.slip.shape[0]
                else:
                    ne += 1
                    dp = '{:12s}'.format('{:4d} - {:4d}'.format(ns,ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Pressure')
                    ns += 1 #fault.slip.shape[0]

                # How many slip parameters
                if ne>nSlip:
                    nSlip = ne

                # conditions on orbits (the rest is orbits)
                npo = ne - nfs
                no = source.Gassembled.shape[1] - npo
                if no>0:
                    ne += no
                    op = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Extra Parameters')
                    ns += no
                else:
                    op = 'None'

                # Store
                self.paramDescription[source.name] = {}
                self.paramDescription[source.name]['Pressure'] = dp
                self.paramDescription[source.name]['Extra Parameters'] = op
            
            elif source.type == "Block":
                
                self.paramDescription[source.name] = {}
                
                # Block rotation
                prot = 'None'
                if 'rotation' in source.G[source.datanames[0]] and 'rotation' in source.blockcomponent:
                    ne += 3
                    for i in range(ns, ne):
                        self.paramTypes[i] = (source.name, 'Rotation')
                    prot = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    self.paramDescription[source.name]['Rotation'] = prot
                    ns += 3
                
                # Intrablock deformation
                pint = 'None'
                if 'intradef' in source.G[source.datanames[0]] and 'intradef' in source.blockcomponent:
                    ne += 3
                    for i in range(ns, ne):
                        self.paramTypes[i] = (source.name, 'Intra def.')
                    pint = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                    self.paramDescription[source.name]['Intra def.'] = pint
                    ns += 3
            
            elif source.type == 'MultiBlock':
                
                for block_ in source.blocks:
                    
                    self.paramDescription[block_.name] = {}
                
                    # Block rotation
                    prot = 'None'
                    if ('rotation' in source.G[source.datanames[0]] and 'rotation' in source.blockcomponent[block_.name]) or ('boundary' in source.G[source.datanames[0]] and 'boundary' in source.blockcomponent[block_.name]):
                        ne += 3
                        for i in range(ns, ne):
                            self.paramTypes[i] = (block_.name, 'Rotation')
                        prot = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                        self.paramDescription[block_.name]['Rotation'] = prot
                        ns += 3
                    
                    # Intrablock deformation
                    pint = 'None'
                    if 'intradef' in source.G[source.datanames[0]] and 'intradef' in source.blockcomponent[block_.name]:
                        ne += 3
                        for i in range(ns, ne):
                            self.paramTypes[i] = (block_.name, 'Intra def.')
                        pint = '{:12s}'.format('{:4d} - {:4d}'.format(ns, ne))
                        self.paramDescription[block_.name]['Intra def.'] = pint
                        ns += 3
    
            elif source.type == 'Surface':
                
                # Get how long the GFs are
                ne += source.Gassembled.shape[1]
                for i in range(ns,ne): self.paramTypes[i] = (source.name, 'Surface')
                dp = '{:12s}'.format('{:4d} - {:4d}'.format(ns,ne))
                self.paramDescription[source.name] = {}
                self.paramDescription[source.name]['Surface'] = dp
                ns += source.Gassembled.shape[1]

        # Store the number of slip parameters
        self.nSlip = nSlip

        # all done
        return

    def assembleCm(self):
        '''
        Assembles the Model Covariance Matrix for the concerned sources.

        Returns:
            * None
        '''

        # Get the sources
        sources = self.sources

        # Get the size of Cm
        Np = 0
        st = []
        se = []
        
        if self.source_indexes is None:
            self.source_indexes = {}
            
        for source in sources:
            st.append(Np)
            Np += source.Gassembled.shape[1]
            se.append(Np)
            self.source_indexes[source.name] = [st[-1], se[-1]]

        # Allocate Cm
        self.Cm = np.zeros((Np, Np))

        # Store the guys
        for source in sources:
            st = self.source_indexes[source.name][0]
            se = self.source_indexes[source.name][1]
            self.Cm[st:se, st:se] = source.Cm

        # Store the number of parameters
        self.Np = Np

        # All done
        return

    def affectFaultIndexParameters(self, fault):
        '''
        Build the index parameter for a fault.

        Args:
            * fault : instance of a fault

        Returns:
            * None
        '''

        # Get indexes
        st = self.source_indexes[fault.name][0]
        se = self.source_indexes[fault.name][1]

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
        After computing the m_post model, this routine distributes the m parameters to the sources.

        Kwargs:
            * verbose   : talk to me

        Returns:
            * None
        '''

        # Get the sources
        sources = self.sources

        # Loop over the sources
        for source in sources:

            if verbose:
                print ("---------------------------------")
                print ("---------------------------------")
                print("Distribute the parameters to source {}".format(source.name))

            # Store the mpost
            st = self.source_indexes[source.name][0]
            se = self.source_indexes[source.name][1]
            source.mpost = self.mpost[st:se]

            # Transformation object
            if source.type == 'transformation':
                
                # Distribute simply
                source.distributem()

            # Fault object
            elif source.type == "Fault":

                # Affect the indexes
                self.affectFaultIndexParameters(source)

                # put the slip values in slip
                st = 0
                if 's' in source.slipdir:
                    se = st + source.slip.shape[0]
                    source.slip[:,0] = source.mpost[st:se]
                    st += source.slip.shape[0]
                if 'd' in source.slipdir:
                    se = st + source.slip.shape[0]
                    source.slip[:,1] = source.mpost[st:se]
                    st += source.slip.shape[0]
                if 't' in source.slipdir:
                    se = st + source.slip.shape[0]
                    source.slip[:,2] = source.mpost[st:se]
                    st += source.slip.shape[0]
                if 'c' in source.slipdir:
                    se = st + source.slip.shape[0]
                    source.coupling = source.mpost[st:se]
                    st += source.slip.shape[0]

                # check
                if hasattr(source, 'NumberCustom'):
                    source.custom = {} # Initialize dictionnary
                    # Get custom params for each dataset
                    for dset in source.datanames:
                        if 'custom' in source.G[dset].keys():
                            nc = source.G[dset]['custom'].shape[1] # Get number of param for this dset
                            se = st + nc
                            source.custom[dset] = source.mpost[st:se]
                            st += nc
            
            # Pressure object
            elif source.type == "Pressure":

                st = 0
                if source.source in {"Mogi", "Yang"}:
                    se = st + 1
                    source.deltapressure = source.mpost[st:se].item()
                    st += 1
                
                elif source.source=="pCDM":
                    se = st + 1
                    source.DVx = source.mpost[st:se].item()
                    st += 1
                    se = st + 1
                    source.DVy = source.mpost[st:se].item()
                    st += 1
                    se = st + 1
                    source.DVz = source.mpost[st:se].item()
                    st += 1

                    if source.DVtot is None:
                        source.computeTotalpotency()
                
                elif source.source=="CDM":
                    se = st + 1
                    source.deltaopening = source.mpost[st:se].item()
                    st += 1

            elif source.type == 'Block':
                
                st = 0
                
                # Block rotation
                if 'rotation' in source.G[source.datanames[0]] and 'rotation' in source.blockcomponent:
                    
                    se = st + 1
                    source.omega_x = source.mpost[st:se].item() * MAS2RAD
                    st += 1
                    se = st + 1
                    source.omega_y = source.mpost[st:se].item() * MAS2RAD
                    st += 1
                    se = st + 1
                    source.omega_z = source.mpost[st:se].item() * MAS2RAD
                    st += 1
                    
                    source.rotation2pole()

                # Intrablock deformation
                if 'intradef' in source.G[source.datanames[0]] and 'intradef' in source.blockcomponent:
                    
                    se = st + 1
                    source.eps_lonlon = source.mpost[st:se].item()
                    st += 1
                    se = st + 1
                    source.eps_lonlat = source.mpost[st:se].item()
                    st += 1
                    se = st + 1
                    source.eps_latlat = source.mpost[st:se].item()
                    st += 1
            
            elif source.type == 'MultiBlock':
                
                st = 0
                
                # Block rotation
                for block_ in source.blocks:

                    if ('rotation' in source.G[source.datanames[0]] and 'rotation' in source.blockcomponent[block_.name]) or ('boundary' in source.G[source.datanames[0]] and 'boundary' in source.blockcomponent[block_.name]):

                        se = st + 1
                        block_.omega_x = source.mpost[st:se].item() * MAS2RAD
                        st += 1
                        se = st + 1
                        block_.omega_y = source.mpost[st:se].item() * MAS2RAD
                        st += 1
                        se = st + 1
                        block_.omega_z = source.mpost[st:se].item() * MAS2RAD
                        st += 1

                        block_.rotation2pole()

                # Intrablock deformation
                for block_ in source.blocks:

                    if 'intradef' in source.G[source.datanames[0]] and 'intradef' in source.blockcomponent[block_.name]:

                        se = st + 1
                        block_.eps_lonlon = source.mpost[st:se].item()
                        st += 1
                        se = st + 1
                        block_.eps_lonlat = source.mpost[st:se].item()
                        st += 1
                        se = st + 1
                        block_.eps_latlat = source.mpost[st:se].item()
                        st += 1

            elif source.type == 'Surface':
                
                directions = [source.direction[data] for data in source.direction][0]
                st = 0
                if 'e' in directions:
                    se = st + source.motion.shape[0]
                    source.motion[:,0] = source.mpost[st:se]
                    st += source.motion.shape[0]
                if 'n' in directions:
                    se = st + source.motion.shape[0]
                    source.motion[:,1] = source.mpost[st:se]
                    st += source.motion.shape[0]
                if 'u' in directions:
                    se = st + source.motion.shape[0]
                    source.motion[:,2] = source.mpost[st:se]

            # Get the polynomial/orbital/helmert values if they exist
            if source.type in ('Fault', 'Pressure', 'Block', 'MultiBlock'):
                source.polysol = {}
                source.polysolindex = {}
                for dset in source.datanames:
                    if dset in source.poly.keys():
                        if (source.poly[dset] is None):
                            source.polysol[dset] = None
                        else:

                            if (source.poly[dset].__class__ is not str) and (source.poly[dset].__class__ is not list):
                                if (source.poly[dset] > 0):
                                    se = st + source.poly[dset]
                                    source.polysol[dset] = source.mpost[st:se]
                                    source.polysolindex[dset] = range(st,se)
                                    st += source.poly[dset]
                            elif (source.poly[dset].__class__ is str):
                                if source.poly[dset]=='full':
                                    nh = source.helmert[dset]
                                    se = st + nh
                                    source.polysol[dset] = source.mpost[st:se]
                                    source.polysolindex[dset] = range(st,se)
                                    st += nh
                                if source.poly[dset] in ('strain', 'strainnorotation', 'strainonly', 'strainnotranslation', 'translation', 'translationrotation'):
                                    nh = source.strain[dset]
                                    se = st + nh
                                    source.polysol[dset] = source.mpost[st:se]
                                    source.polysolindex[dset] = range(st,se)
                                    st += nh
                            elif (source.poly[dset].__class__ is list):
                                nh = source.transformation[dset]
                                se = st + nh
                                source.polysol[dset] = source.mpost[st:se]
                                source.polysolindex[dset] = range(st,se)
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
        if self.verbose: print ("Computing the inverse of the data covariance")
        iCd = scilin.inv(Cd)

        # Construct mprior
        if mprior is None:
            mprior = np.zeros((Nm,))

        # Compute mpost
        if self.verbose: print ("Computing m_post")
        One = scilin.inv(np.dot(  np.dot(G.T, iCd), G ) )
        Res = d - np.dot( G, mprior )
        Two = np.dot( np.dot( G.T, iCd ), Res )
        mpost = mprior + np.dot( One, Two )

        # Store m_post
        self.mpost = mpost

        # All done
        return

    def GeneralizedLeastSquareSoln(self, mprior=None, rcond=None, useCm=True, mw=False):
        '''
        Solves the generalized least-square problem using the following formula (Tarantola, 2005,         Inverse Problem Theory, SIAM):

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
            if self.verbose: print ("Computing the inverse of the model covariance")
            iCm = scilin.inv(Cm)
        else:
            iCm = np.zeros(Cm.shape)

        # Check If Cm is symmetric and positive definite
        if (Cd.transpose() != Cd).all():
            print("Cd is not symmetric, Return...")
            return

        # Get the inverse of Cd
        if self.verbose: print ("Computing the inverse of the data covariance")
        
        # Invert
        if hasattr(self, 'iCd'):
            iCd = self.iCd
        else:
            if rcond is None:
                iCd = np.linalg.inv(Cd)
            else:
                iCd = np.linalg.pinv(Cd, rcond=rcond)
            self.iCd = iCd

        # Construct mprior
        if mprior is None:
            mprior = np.zeros((Nm,))

        # Compute mpost
        if self.verbose: print ("Computing m_post")
        One = scilin.inv(np.dot(  np.dot(G.T, iCd), G ) + iCm )
        Res = d - np.dot( G, mprior )
        Two = np.dot( np.dot( G.T, iCd ), Res )
        mpost = mprior + np.dot( One, Two )
        Err = d - np.dot( G, mpost )
        # Store m_post
        self.mpost = mpost
        
        mu = 22.5e9
        Mw_thresh = 10
        if mw:
            computeMwDiff(self.mpost, Mw_thresh, self.patchAreas*1.e6, mu)

        # All done£
        return

    def computeCmPostGeneral(self, rcond=None):
        """
        Computes the general posterior covariance matrix. See Tarantola 2005.
        Result is stored in self.Cmpost
        """
        
        # Import things
        import scipy.linalg as scilin

        # Get things
        G = self.G
        iCm = scilin.inv(self.Cm)

        Cd = self.Cd
        
        # Invert
        if hasattr(self, 'iCd'):
            iCd = self.iCd
        else:
            if rcond is None:
                iCd = np.linalg.inv(Cd)
            else:
                iCd = np.linalg.pinv(Cd, rcond=rcond)
            self.iCd = iCd
    
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

        # Assert
        assert G.shape[0]==d.shape[0], "Green's functions and data not compatible: {} / {}".format(G.shape, d.shape)
        assert G.shape[1]==Cm.shape[1], "Green's functions and model covariance not compatible: {} / {}".format(G.shape, Cm.shape)
        if self.verbose: print('Final data space size: {}'.format(d.shape[0]))
        if self.verbose: print('Final model space size: {}'.format(Cm.shape[0]))

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
        
        # Invert
        if hasattr(self, 'iCd'):
            iCd = self.iCd
        else:
            if rcond is None:
                iCd = np.linalg.inv(Cd)
            else:
                iCd = np.linalg.pinv(Cd, rcond=rcond)
            self.iCd = iCd

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
                print("Performing constrained minimization")
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
            if function=='Gaussian':
                center = params[0]
                tau = params[1]
                p = pymc.Gaussian(name, center, tau, value=init)
            elif function=='Uniform':
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

        # keep track of the file
        self.mpostfile = outfile

        # Print stuff
        if self.verbose: print("Writing Posterior Model to file {}".format(outfile))
        if self.verbose: print("Posterior Model matrix size: {}".format(self.mpost.shape[0]))

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
        
        # Print stuff
        if self.verbose: print("Writing Posterior Model to file {}".format(outfile))
        if self.verbose: print("Posterior Model matrix size: {}".format(self.mpost.shape[0]))

        # keep track of the file
        self.mpostfile = outfile

        # all done
        return

    def writeMpost2H5File(self, outfile, name='static.initialModel'):
        '''
        Writes the solution to a binary file.

        Args:
            * outfile       : Output file name

        
        Kwargs:
            *name           : Name of the dataset

        Returns:
            * None
        '''

        # Do it
        with h5py.File(outfile, 'w') as fout:
            fout.create_dataset(name, data=self.mpost)
        
        # keep track of the file
        self.mpostfile = outfile
        
        # Print stuff
        if self.verbose: print("Writing Posterior Model to file {}".format(outfile))
        if self.verbose: print("Posterior Model matrix size: {}".format(self.mpost.shape[0]))

        # all done
        return

    def writeGFs2H5File(self, outfile, name='static.gf'):
        '''
        Writes the assembled GFs to the file outfile.

        Args:
            * outfile       : Name of the output file.

        Kwargs:
            * name          : Name of the dataset in the file

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # Write to file
        with h5py.File(outfile, 'w') as fout:
            fout.create_dataset(name, data=self.G)

        # Keep track of the file
        self.Gfile = outfile

        # Print stuff
        if self.verbose: print("Writing Green's functions to file {}".format(outfile))
        if self.verbose: print("Green's functions matrix size: {} ; {}".format(self.G.shape[0], self.G.shape[1]))

        # All done
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
            dtype = float

        # Convert the data
        G = self.G.astype(dtype)

        # Write to file
        G.tofile(outfile)

        # Keep track of the file
        self.Gfile = outfile

        # Print stuff
        if self.verbose: print("Writing Green's functions to file {}".format(outfile))
        if self.verbose: print("Green's functions matrix size: {} ; {}".format(G.shape[0], G.shape[1]))

        # All done
        return

    def writeData2H5File(self, outfile, name='static.data'):
        '''
        Writes the assembled data vector to an output file.

        Args:
            * outfile       : Name of the output file.

        Kwargs:
            * name          : name of the dataset in the file

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the GFs'

        # Write to file
        with h5py.File(outfile, 'w') as fout:
            fout.create_dataset(name, data=self.d)

        # Keep track of the file
        self.dfile = outfile

        # Print stuff
        if self.verbose: print("Writing Data to file {}".format(outfile))
        if self.verbose: print("Data vector size: {}".format(self.d.shape[0]))

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
            dtype = float

        # Convert the data
        d = self.d.astype(dtype)

        # Write to file
        d.tofile(outfile)

        # Keep track of the file
        self.dfile = outfile

        # Print stuff
        if self.verbose: print("Writing Data to file {}".format(outfile))
        if self.verbose: print("Data vector size: {}".format(d.shape[0]))

        # All done
        return

    def writeCd2H5File(self, outfile, name='static.Cd', scale=1.):
        '''
        Writes the assembled Data Covariance matrix to a hdf5 file

        Args:
            * outfile       : Name of the output file.

        Kwargs:
            * scale         : Multiply the data covariance.
            * name          : name of the dataset in the file

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the Cd'

        # Write to file
        with h5py.File(outfile, 'w') as fout:
            fout.create_dataset(name, data=self.Cd*scale)

        # keep track of the file
        self.Cdfile = outfile

        # print stuff
        if self.verbose: print("Writing Data Covariance to file {}".format(outfile))
        if self.verbose: print("Data Covariance Size: {} ; {}".format(self.Cd.shape[0], self.Cd.shape[1]))

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
        assert self.ready, 'You need to assemble the Cd'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = float

        # Convert the data
        Cd = self.Cd.astype(dtype) * scale

        # Write to file
        Cd.tofile(outfile)

        # keep track of the file
        self.Cdfile = outfile

        # print stuff
        if self.verbose: print("Writing Data Covariance to file {}".format(outfile))
        if self.verbose: print("Data Covariance Size: {} ; {}".format(Cd.shape[0], Cd.shape[1]))

        # All done
        return

    def writeCm2H5File(self, outfile, name='static.Cm', scale=1.):
        '''
        Writes the assembled Model Prior Covariance matrix to a hdf5 file

        Args:
            * outfile       : Name of the output file.

        Kwargs:
            * scale         : Multiply the model prior covariance.
            * name          : name of the dataset in the file

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the Cm'

        # Write to file
        with h5py.File(outfile, 'w') as fout:
            fout.create_dataset(name, data=self.Cm*scale)

        # keep track of the file
        self.Cmfile = outfile

        # print stuff
        if self.verbose: print("Writing Model Prior Covariance to file {}".format(outfile))
        if self.verbose: print("Model Prior Covariance Size: {} ; {}".format(self.Cm.shape[0], self.Cm.shape[1]))

        # All done
        return

    def writeCm2BinaryFile(self, outfile='Cm.dat', dtype='f', scale=1.):
        '''
        Writes the assembled Model Prior Covariance matrix to a binary file.

        Args:
            * outfile       : Name of the output file.
            * scale         : Multiply the model prior covariance.
            * dtype         : Type of data to write. Can be 'f', 'float', 'd' or 'double'.

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the Cm'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = float

        # Convert the data
        Cm = self.Cm.astype(dtype) * scale

        # Write to file
        Cm.tofile(outfile)

        # keep track of the file
        self.Cmfile = outfile

        # print stuff
        if self.verbose: print("Writing Model Prior Covariance to file {}".format(outfile))
        if self.verbose: print("Model Prior Covariance Size: {} ; {}".format(Cm.shape[0], Cm.shape[1]))

        # All done
        return

    def writeCmpost2H5File(self, outfile, name='static.Cmpost', scale=1.):
        '''
        Writes the assembled Model Prior Covariance matrix to a hdf5 file

        Args:
            * outfile       : Name of the output file.

        Kwargs:
            * scale         : Multiply the model posterior covariance.
            * name          : name of the dataset in the file

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the Cm'

        # Write to file
        with h5py.File(outfile, 'w') as fout:
            fout.create_dataset(name, data=self.Cmpost*scale)

        # keep track of the file
        self.Cmpostfile = outfile

        # print stuff
        if self.verbose: print("Writing Model Posterior Covariance to file {}".format(outfile))
        if self.verbose: print("Model Posterior Covariance Size: {} ; {}".format(self.Cmpost.shape[0], self.Cmpost.shape[1]))

        # All done
        return

    def writeCmpost2BinaryFile(self, outfile='Cmpost.dat', dtype='f', scale=1.):
        '''
        Writes the assembled Model Posterior Covariance matrix to a binary file.

        Args:
            * outfile       : Name of the output file.
            * scale         : Multiply the model posterior covariance.
            * dtype         : Type of data to write. Can be 'f', 'float', 'd' or 'double'.

        Returns:
            * None
        '''

        # Assert
        assert self.ready, 'You need to assemble the Cm'

        # data type
        if dtype in ('f', 'float'):
            dtype = np.float32
        elif dtype in ('d', 'double'):
            dtype = float

        # Convert the data
        Cmpost = self.Cmpost.astype(dtype) * scale

        # Write to file
        Cmpost.tofile(outfile)

        # keep track of the file
        self.Cmpostfile = outfile

        # print stuff
        if self.verbose: print("Writing Model Posterior Covariance to file {}".format(outfile))
        if self.verbose: print("Model Posterior Covariance Size: {} ; {}".format(self.Cmpost.shape[0], self.Cmpost.shape[1]))

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

        # Obsolet
        assert False, 'This is totally obsolete'

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

        assert False, 'This is totally obsolete...'

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
        for source in self.sources:
            for slipmode in ['Strike Slip', 'Dip Slip', 'Tensile Slip', 'Extra Parameters']:
                if parDescr[source.name][slipmode].replace(' ','') != 'None':
                    fout.write("; {} parameters of {} fault segment\n".format(slipmode, source.name))
                    fout.write("[ init_prior_{} ]\n".format(k))

                    ind = parDescr[source.name][slipmode].replace(' ','').partition('-')
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
        for source in self.sources:
            for slipmode in ['Strike Slip', 'Dip Slip', 'Tensile Slip', 'Extra Parameters']:
                if parDescr[source.name][slipmode].replace(' ','') != 'None':
                    fout.write("; {} parameters of {} fault segment\n".format(slipmode, source.name))
                    fout.write("[ run_prior_{} ]\n".format(k))

                    ind = parDescr[source.name][slipmode].replace(' ','').partition('-')
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
