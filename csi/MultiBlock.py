'''
A class that deals with an ensemble of blocks and blocks boundaries.

Written by E. Denise, April 2025
'''

# Import external packages
import os
import numpy as np
import copy
import sys
import scipy.spatial.distance as scidis
import multiprocessing as mp

# Import internal packages
from .Block import Block
from .Fault import Fault
from .SourceInv import SourceInv
from .eulerPoleUtils import ERADIUS, MASYR2DEGMYR, MAS2RAD, llh2xyz, xyz2llh
from .geodeticplot import geodeticplot as geoplot
from .EDKSmp import dropSourcesInPatches as Patches2Sources
from .EDKSmp import sum_layered
from .EDKSmp import interpolateEDKS

# Funnctions

def _edks_chunk_worker(args):

    (
        chunk_id,
        ids_chunk,
        xs_chunk,
        ys_chunk,
        zs_chunk,
        strike_chunk,
        dip_chunk,
        rake_chunk,
        slip_chunk,
        area_chunk,
        xr,
        yr,
        stratKernels,
        prefix,
        unique_ids,
        cleanUp,
        verbose,
        tensile
    ) = args

    local_prefix = f"{prefix}_chunk{chunk_id}"

    iG = np.array(
        sum_layered(
            xs_chunk,
            ys_chunk,
            zs_chunk,
            strike_chunk,
            dip_chunk,
            rake_chunk,
            slip_chunk,
            np.sqrt(area_chunk),
            np.sqrt(area_chunk),
            1,
            1,
            xr,
            yr,
            stratKernels,
            local_prefix,
            BIN_EDKS='EDKS_BIN',
            cleanUp=cleanUp,
            verbose=verbose,
            tensile=tensile
        )
    )

    nrec = iG.shape[1]
    npatch = len(unique_ids)

    G_partial = np.zeros(
        (3, nrec, npatch),
        dtype=np.float32
    )

    id_to_col = {
        gid: i
        for i, gid in enumerate(unique_ids)
    }

    for gid in np.unique(ids_chunk):

        idx = np.flatnonzero(ids_chunk == gid)

        G_partial[:, :, id_to_col[gid]] += np.sum(
            iG[:, :, idx],
            axis=2
        )

    return G_partial


def _parallel_edks_sources(
        Ids,
        xs,
        ys,
        zs,
        strike,
        dip,
        rake,
        slip,
        Areas,
        xr,
        yr,
        stratKernels,
        prefix,
        cleanUp,
        verbose,
        tensile=False,
        nworkers=16,
        chunk_size=10000):

    unique_ids = np.unique(Ids)

    tasks = []

    for i0 in range(0, len(xs), chunk_size):

        i1 = min(i0 + chunk_size, len(xs))

        tasks.append(
            (
                i0,
                Ids[i0:i1],
                xs[i0:i1],
                ys[i0:i1],
                zs[i0:i1],
                strike[i0:i1],
                dip[i0:i1],
                rake[i0:i1],
                slip[i0:i1],
                Areas[i0:i1],
                xr,
                yr,
                stratKernels,
                prefix,
                unique_ids,
                cleanUp,
                verbose,
                tensile
            )
        )

    with mp.Pool(nworkers) as pool:
        partials = pool.map(
            _edks_chunk_worker,
            tasks
        )

    G = np.sum(partials, axis=0)

    return G

# Class Block
class MultiBlock(SourceInv):

    '''
        Class implementing an ensemble of Block objects and block boundaries,
        based on the approach of B.J. Meade and J.P. Loveless (2009).

        You can specify either an official utm zone number or provide
        longitude and latitude for a custom zone (may be wrong for
        large blocks, but not used for block rotation and internal
        block deformation anyway).

        Args:
            * name          : Name of the block.
            * utmzone       : UTM zone  (optional, default=None)
            * lon0          : Longitude defining the center of the custom utm zone
            * lat0          : Latitude defining the center of the custom utm zone
            * ellps         : ellipsoid (optional, default='WGS84')
    '''

    # ----------------------------------------------------------------------
    # Initialize class
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Base class init
        super(MultiBlock, self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)

        # Initialize the block
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing multi block {}".format(self.name))
        self.verbose = verbose
        
        self.type = "MultiBlock"

        # Allocate blocks and boundaries attributes
        self.blocks = []
        self.Nblocks = 0
        self.blockboundaries = []
        
        # A priori covariance matrix
        self.Cm = None

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
    # Returns a copy of the multi block
    def duplicateMultiBlock(self):
        '''
        Returns a full copy (copy.deepcopy) of the multi block object.

        Return:
            * multiblock    : multi block object
        '''

        # All done
        return copy.deepcopy(self)
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the blocks objects
    def setBlocks(self, blocks=None, verbose=True):
        '''
        Set the list of blocks in the multi block object (must be csi Block objects).
        The blocks are stored in self.blocks.

        Args:
            * blocks        : list containing csi Block objects

        Returns:
            * None
        '''
        
        # Speak
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Setting blocks for multi block {}".format(self.name))
        
        # Reinitialize the blocks
        self.blocks = []
        
        # Check
        if not isinstance(blocks, list) or len(blocks) < 2 or not all(isinstance(b, Block) for b in blocks):
            raise ValueError("Please provide a list of Block objects with at least 2 blocks.")

        # Set  list of blocks
        for block_ in blocks:
            
            self.blocks.append(block_)
            
            if verbose:
                print("   Setting Block {}".format(block_.name))
        
        self.Nblocks = len(self.blocks)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the block fault boundaries objects
    def setBlockFaultBoundaries(self, faults=[], discretized=False, coordinates='xy', verbose=True, custom=None,
                                dist_step=1e-3, Nmax_iter=1e9):
        '''
        Giviving a list of Fault objects which are block boundaries, associate them to their bounding blocks.
        In details, for each patch or tent composing the fault, find the two blocks on each side of the fault.
        These faults must be block boundaries because this function will find the two bounding blocks for each fault patch.
        
        The boundaries are stored in self.blockboundaries and the blocks in fault.patchblock.

        Kwargs:
            * faults        : list containing csi Fault objects which are block boundaries.
            * discretized   : If True, will use the discretized block boundaries.
            * coordinates   : 'xy' (or 'utm') or 'lonlat' (or 'll'). Coordinates system to use to find the blocks.
            * verbose       : If True, will print out information
            * custom        : If you want to manually provide the block pairs for each fault patch, a dictionary 
            where keys are faults object and values are lists of [blockA, blockB] for each patch of the fault.
            * dist_step     : Distance step to use when finding blocks near fault patches (default is 1e-3 km).
            Warning: if the blocks are very small, you may need to decrease this value.
            * Nmax_iter     : Maximum number of iterations to find blocks near fault patches (default is 1e9).
            
        Returns:
            * None
        '''
        
        # Speak
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Setting block fault boundaries for multi block {}".format(self.name))
        
        # Check if the list of blocks is initialized
        if self.Nblocks == 0:
            raise ValueError("Please set the block list before setting the block fault boundaries.")

        # Reinitialize the block boundaries
        self.blockboundaries = []
        
        if custom is not None:
            
            for fault in custom:
                
                if verbose:
                    print ("   Setting Fault {} with custom block pairs".format(fault.name))
                
                # Check if the fault is a csi Fault object with rectangular or triangular patches
                if not isinstance(fault, Fault) or fault.patchType not in ('rectangle', 'triangle', 'triangletent'):
                    raise ValueError("Please provide csi Fault objects with rectangular, triangular or triangular tent patches.")
                
                self.blockboundaries.append(fault)
                fault.multiblock = self
                
                # Check the custom input size for patches
                if fault.patchType in ('rectangle', 'triangle'):
                    if len(custom[fault]) != len(fault.patch):
                        raise ValueError("Please provide a list of block pairs for each patch of the fault {}.".format(fault.name))
                    
                    # Associate the blocks
                    fault.patchblock = custom[fault]
                
                # Check the custom input size for tents
                if fault.patchType == 'triangletent':
                    if len(custom[fault]) != len(fault.tent):
                        raise ValueError("Please provide a list of block pairs for each tent of the fault {}.".format(fault.name))
                    # Associate the blocks
                    fault.tentblock = custom[fault]
        
        else:
            
            # Check
            if not isinstance(faults, list):
                faults = [faults]

            # Loop over the faults
            for fault in faults:
                
                # Check if the fault is a csi Fault object with rectangular or triangular patches
                if not isinstance(fault, Fault) or fault.patchType not in ('rectangle', 'triangle', 'triangletent'):
                    raise ValueError("Please provide csi Fault objects with rectangular, triangular or triangular tent patches.")

                self.blockboundaries.append(fault)
                fault.multiblock = self
                
                ################################################
                # 1st loop over the patches/tents of the fault to establish the association between each patch/tent and the two blocks it separates
                ################################################
                
                if fault.patchType in ('rectangle', 'triangle'):
                    
                    # Find the two bounding blocks in the block list for each patch of the fault
                    fault.patchblock = []
                    
                    for idx_patch in range(len(fault.patch)):
                        
                        if verbose:
                            sys.stdout.write('\r   Setting Fault {}: {} / {} patches'.format(fault.name, idx_patch+1,len(fault.patch)))
                            sys.stdout.flush()
                            if idx_patch == len(fault.patch)-1:
                                print('\r   Setting Fault {}: {} / {} patches'.format(fault.name, idx_patch+1,len(fault.patch)))
                        
                        # Get the center of the fault patch
                        x_center, y_center, z_center = fault.getcenter(idx_patch)
                        
                        # Get the coordinates of the projection of the fault patch on the fault trace
                        
                        if discretized:
                            xf = fault.xi
                            yf = fault.yi
                        else:
                            xf = fault.xf
                            yf = fault.yf
                        zf = np.zeros_like(xf)
                        
                        d = scidis.cdist([[x_center, y_center, z_center]], [[x_, y_, z_] for x_, y_, z_ in zip(xf, yf, zf)])[0]
                        imin = d.argmin()
                        x_patch_trace, y_patch_trace = xf[imin], yf[imin]
                        
                        # Get strike of the fault patch
                        strike = fault.getpatchstrike(idx_patch)
                        
                        # Offset the point in the direction normal to the fault strike
                        x_offset = -np.sin(np.pi/2 - strike)
                        y_offset = np.cos(np.pi/2 - strike)
                        
                        # Find the two blocks on each side of the fault patch
                        # We go in each direction along the normal to the fault strike until we find two different blocks
                        # Warning: due to the approximation of the geometry of the patches, the blocks found may not be accurate ...
                        
                        # Flag to indicate if we found the blocks
                        blocks_found = False
                        
                        # Iteration counter
                        i_iter = 1
                        
                        list_block = {}
                        
                        while not blocks_found and i_iter < Nmax_iter:
                            
                            x_left = x_patch_trace + x_offset * dist_step * i_iter
                            y_left = y_patch_trace + y_offset * dist_step * i_iter
                            x_right = x_patch_trace - x_offset * dist_step * i_iter
                            y_right = y_patch_trace - y_offset * dist_step * i_iter
                            
                            if coordinates in ('lonlat', 'll'):
                                x_left, y_left = self.xy2ll(x_left, y_left)
                                x_right, y_right = self.xy2ll(x_right, y_right)

                            blockleft = np.array([block_.PointInBlock(x_left, y_left, coord=coordinates) for block_ in self.blocks])
                            blockright = np.array([block_.PointInBlock(x_right, y_right, coord=coordinates) for block_ in self.blocks])
                            
                            if True in blockleft:
                                
                                blockleft = self.blocks[np.argwhere(blockleft)[0][0]]
                                if blockleft not in list_block.keys():
                                    list_block[blockleft] = [i_iter, 'left']
                            
                            if True in blockright:
                            
                                blockright = self.blocks[np.argwhere(blockright)[0][0]]
                                if blockright not in list_block.keys():
                                    list_block[blockright] = [i_iter, 'right']
                            
                            if len(list_block.keys()) >= 2:
                                blocks_found = True
                            
                            i_iter += 1
                        
                        list_block = [[b, list_block[b][0], list_block[b][1]] for b in list_block]

                        # Check but should not happen ...
                        if len(list_block) < 2:
                            print(f"Less than two blocks found on either side of the fault patch {idx_patch}/{len(fault.patch)} for fault {fault.name}!")
                            fault.patchblock.append([None, None])
                        
                        elif len(list_block) > 2:
                            print(f"More than two blocks found on either side of the fault patch {idx_patch}/{len(fault.patch)} for fault {fault.name}!")
                            fault.patchblock.append([None, None])
                        
                        else:
                            # Associate the block in good order: blockA on the left, blockB on the right
                            if list_block[0][2] == 'left' and list_block[1][2] == 'right':
                                blockA = list_block[1][0]
                                blockB = list_block[0][0]
                            
                            elif list_block[0][2] == 'right' and list_block[1][2] == 'left':
                                blockA = list_block[0][0]
                                blockB = list_block[1][0]
                            
                            elif list_block[0][2] == 'left' and list_block[0][1] > list_block[1][1]:
                                blockA = list_block[1][0]
                                blockB = list_block[0][0]
                            
                            elif list_block[0][2] == 'left' and list_block[1][1] > list_block[0][1]:
                                blockA = list_block[0][0]
                                blockB = list_block[1][0]
                            
                            elif list_block[0][2] == 'right' and list_block[0][1] > list_block[1][1]:
                                blockA = list_block[0][0]
                                blockB = list_block[1][0]
                            
                            elif list_block[0][2] == 'right' and list_block[1][1] > list_block[0][1]:
                                blockA = list_block[1][0]
                                blockB = list_block[0][0]
                            
                            fault.patchblock.append([blockA, blockB])
                
                elif fault.patchType == 'triangletent':
                    
                    # Find the two bounding blocks in the block list for each tent of the fault
                    fault.tentblock = []

                    for idx_tent in range(fault.numtent):
                        
                        if verbose:
                            sys.stdout.write('\r   Setting Fault {}: {} / {} tents'.format(fault.name, idx_tent+1, fault.numtent))
                            sys.stdout.flush()
                            if idx_tent == fault.numtent-1:
                                print('\r   Setting Fault {}: {} / {} tents'.format(fault.name, idx_tent+1, fault.numtent))
                        
                        # Get the coordinate of the node
                        x_tent, y_tent, z_tent = fault.tent[idx_tent]
                        
                        # Get the coordinates of the projection of the fault tent on the fault trace
                        if discretized:
                            xf = fault.xi
                            yf = fault.yi
                        else:
                            xf = fault.xf
                            yf = fault.yf
                        zf = np.zeros_like(xf)
                        
                        d = scidis.cdist([[x_tent, y_tent, z_tent]], [[x_, y_, z_] for x_, y_, z_ in zip(xf, yf, zf)])[0]
                        imin = d.argmin()
                        x_tent_trace, y_tent_trace = xf[imin], yf[imin]
                        
                        # Get strike of the fault tent
                        strike = fault.gettentstrike(idx_tent)
                        
                        # Offset the point in the direction normal to the fault strike
                        x_offset = -np.sin(np.pi/2 - strike)
                        y_offset = np.cos(np.pi/2 - strike)
                        
                        # Find the two blocks on each side of the fault tent
                        # We go in each direction along the normal to the fault strike until we find two different blocks
                        # Warning: due to the approximation of the geometry of the patches, the blocks found may not be accurate ...
                        
                        # Flag to indicate if we found the blocks
                        blocks_found = False
                        
                        # Iteration counter
                        i_iter = 1
                        
                        list_block = {}
                        
                        while not blocks_found and i_iter < Nmax_iter:
                            
                            x_left = x_tent_trace + x_offset * dist_step * i_iter
                            y_left = y_tent_trace + y_offset * dist_step * i_iter
                            x_right = x_tent_trace - x_offset * dist_step * i_iter
                            y_right = y_tent_trace - y_offset * dist_step * i_iter
                            
                            if coordinates in ('lonlat', 'll'):
                                x_left, y_left = self.xy2ll(x_left, y_left)
                                x_right, y_right = self.xy2ll(x_right, y_right)

                            blockleft = np.array([block_.PointInBlock(x_left, y_left, coord=coordinates) for block_ in self.blocks])
                            blockright = np.array([block_.PointInBlock(x_right, y_right, coord=coordinates) for block_ in self.blocks])
                            
                            if True in blockleft:
                                
                                blockleft = self.blocks[np.argwhere(blockleft)[0][0]]
                                if blockleft not in list_block.keys():
                                    list_block[blockleft] = [i_iter, 'left']
                            
                            if True in blockright:
                            
                                blockright = self.blocks[np.argwhere(blockright)[0][0]]
                                if blockright not in list_block.keys():
                                    list_block[blockright] = [i_iter, 'right']
                            
                            if len(list_block.keys()) >= 2:
                                blocks_found = True
                            
                            i_iter += 1
                        
                        list_block = [[b, list_block[b][0], list_block[b][1]] for b in list_block]

                        # Check but should not happen ...
                        if len(list_block) < 2:
                            print(f"Less than two blocks found on either side of the fault tent {idx_tent}/{len(fault.tent)} for fault {fault.name}!")
                            fault.patchblock.append([None, None])
                        
                        elif len(list_block) > 2:
                            print(f"More than two blocks found on either side of the fault tent {idx_tent}/{len(fault.tent)} for fault {fault.name}!")
                            fault.patchblock.append([None, None])
                        
                        else:
                            # Associate the block in good order: blockA on the left, blockB on the right
                            if list_block[0][2] == 'left' and list_block[1][2] == 'right':
                                blockA = list_block[1][0]
                                blockB = list_block[0][0]
                            
                            elif list_block[0][2] == 'right' and list_block[1][2] == 'left':
                                blockA = list_block[0][0]
                                blockB = list_block[1][0]
                            
                            elif list_block[0][2] == 'left' and list_block[0][1] > list_block[1][1]:
                                blockA = list_block[1][0]
                                blockB = list_block[0][0]
                            
                            elif list_block[0][2] == 'left' and list_block[1][1] > list_block[0][1]:
                                blockA = list_block[0][0]
                                blockB = list_block[1][0]
                            
                            elif list_block[0][2] == 'right' and list_block[0][1] > list_block[1][1]:
                                blockA = list_block[0][0]
                                blockB = list_block[1][0]
                            
                            elif list_block[0][2] == 'right' and list_block[1][1] > list_block[0][1]:
                                blockA = list_block[1][0]
                                blockB = list_block[0][0]
                            
                            fault.tentblock.append([blockA, blockB])
                
                ################################################
                # 2nd loop over the patches/tents of the fault to check if some patches/tents are inconsistent with their neighbours
                ################################################
                
                print('   Checking inconsistencies...')
                
                if fault.patchType in ('rectangle', 'triangle'):
                
                    # Compute the adjacency map of the fault patches
                    fault.buildAdjacencyMap(method='vertex', verbose=False)
                    adjacencyMapI = copy.deepcopy(fault.adjacencyMap)
                    
                    adjacencyMapII = []
                    for ipatch, patch in enumerate(adjacencyMapI):
                        adjacencyMapII.append([])
                        for iadjacent in patch:
                            adjacencyMapII[ipatch] += adjacencyMapI[iadjacent]
                        adjacencyMapII[ipatch] = np.unique(adjacencyMapII[ipatch]).tolist()
                        adjacencyMapII[ipatch].remove(ipatch)
            
                    # Removing inconsistent patches
                    isInconsistent = True
                    
                    while isInconsistent:
                        
                        isInconsistent = False
                    
                        for idx_patch in range(len(fault.patch)):
                            
                            blockA, blockB = fault.patchblock[idx_patch]
                            block_neigh = [[fault.patchblock[idx_patch_neigh][0], fault.patchblock[idx_patch_neigh][1]] for idx_patch_neigh in adjacencyMapII[idx_patch] if fault.patchblock[idx_patch_neigh][0] is not None]
                    
                            # If blockA and blockB are not None but most of the neighbors have the same bounding blocks
                            # but different from this blockA and blockB, we can infer that there is an inconsistency and
                            # we set blockA and blockB to this most common block pair among the neighbors
                            if blockA is not None and len(block_neigh) > 0:
                                block_neigh_consistent = [b for b in block_neigh if b[0] == blockA and b[1] == blockB]
                                if len(block_neigh_consistent) < .5 * len(block_neigh):
                                    blockA, blockB = None, None
                                    isInconsistent = True

                            fault.patchblock[idx_patch] = [blockA, blockB]
                    
                    # We correct until there is no None left in the patchblock
                    isCorrected = True
                    
                    while isCorrected:
                        
                        isCorrected = False
                    
                        for idx_patch in range(len(fault.patch)):
                            
                            blockA, blockB = fault.patchblock[idx_patch]
                            block_neigh = [[fault.patchblock[idx_patch_neigh][0], fault.patchblock[idx_patch_neigh][1]] for idx_patch_neigh in adjacencyMapII[idx_patch] if fault.patchblock[idx_patch_neigh][0] is not None]
                            
                            # If blockA and blockB are None but most of the neighbors have the same bounding blocks,
                            # we can infer that there is an inconsistency and we set blockA and blockB to this most
                            # common block pair among the neighbors
                            if blockA is None:
                                if len(block_neigh) > 0:
                                    blockA, blockB = max(set(tuple(b) for b in block_neigh), key=block_neigh.count)
                                    isCorrected = True

                            fault.patchblock[idx_patch] = [blockA, blockB]
                
                elif fault.patchType == 'triangletent':
                    
                    # Compute the adjacency map of the fault patches
                    fault.buildTentAdjacencyMap(verbose=False)
                    adjacencyMapI = copy.deepcopy(fault.adjacentTents)

                    adjacencyMapII = []
                    for itent, tent in enumerate(adjacencyMapI):
                        adjacencyMapII.append([])
                        for iadjacent in tent:
                            adjacencyMapII[itent] += adjacencyMapI[iadjacent]
                        adjacencyMapII[itent] = np.unique(adjacencyMapII[itent]).tolist()
                        adjacencyMapII[itent].remove(itent)
                    
                    # Removing inconsistent patches
                    isInconsistent = True

                    while isInconsistent:
                        
                        isInconsistent = False

                        for idx_tent in range(fault.numtent):
                            
                            blockA, blockB = fault.tentblock[idx_tent]
                            block_neigh = [[fault.tentblock[idx_tent_neigh][0], fault.tentblock[idx_tent_neigh][1]] for idx_tent_neigh in adjacencyMapII[idx_tent] if fault.tentblock[idx_tent_neigh][0] is not None]

                            # If blockA and blockB are not None but most of the neighbors have the same bounding blocks
                            # but different from this blockA and blockB, we can infer that there is an inconsistency and
                            # we set blockA and blockB to this most common block pair among the neighbors
                            if blockA is not None and len(block_neigh) > 0:
                                block_neigh_consistent = [b for b in block_neigh if b[0] == blockA and b[1] == blockB]
                                if len(block_neigh_consistent) < .5 * len(block_neigh):
                                    blockA, blockB = None, None
                                    isInconsistent = True

                            fault.tentblock[idx_tent] = [blockA, blockB]

                    # We correct until there is no None left in the tentblock
                    isCorrected = True

                    while isCorrected:
                        
                        isCorrected = False

                        for idx_tent in range(fault.numtent):
                            
                            blockA, blockB = fault.tentblock[idx_tent]
                            block_neigh = [[fault.tentblock[idx_tent_neigh][0], fault.tentblock[idx_tent_neigh][1]] for idx_tent_neigh in adjacencyMapII[idx_tent] if fault.tentblock[idx_tent_neigh][0] is not None]
                            
                            # If blockA and blockB are None but most of the neighbors have the same bounding blocks,
                            # we can infer that there is an inconsistency and we set blockA and blockB to this most
                            # common block pair among the neighbors
                            if blockA is None:
                                if len(block_neigh) > 0:
                                    blockA, blockB = max(set(tuple(b) for b in block_neigh), key=block_neigh.count)
                                    isCorrected = True

                            fault.tentblock[idx_tent] = [blockA, blockB]
        
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Save the block fault boundaries association in a text file
    def saveBlockFaultBoundaries(self, outDir='./'):
        '''
        Save the block fault boundaries association in a text file.
        
        Returns:
            * None
        '''

        # Check if the list of blocks and block boundaries are initialized
        if self.Nblocks == 0:
            raise ValueError("Please set the block list before setting the block fault boundaries.")
        if len(self.blockboundaries) == 0:
            raise ValueError("Please set the block fault boundaries before saving the association.")

        # Create a directory to save the association
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        # Loop over the faults
        for fault in self.blockboundaries:

            # Save the association in a text file
            with open(os.path.join(outDir, '{}_{}_block_boundaries.txt'.format(self.name.replace(' ', '_'), fault.name.replace(' ', '_'))), 'w') as f:
                
                if fault.patchType in ('rectangle', 'triangle'):
                    
                    f.write('Patch index\tBlock A\tBlock B\n')
                    
                    for idx_patch, patch in enumerate(fault.patch):
                        blockA, blockB = fault.patchblock[idx_patch]
                        f.write('{}\t{}\t{}\n'.format(idx_patch, blockA.name if blockA is not None else 'None', blockB.name if blockB is not None else 'None'))
                        
                elif fault.patchType == 'triangletent':
                    
                    f.write('Tent index\tBlock A\tBlock B\n')

                    for idx_tent, tent in enumerate(fault.tent):
                        blockA, blockB = fault.tentblock[idx_tent]
                        f.write('{}\t{}\t{}\n'.format(idx_tent, blockA.name if blockA is not None else 'None', blockB.name if blockB is not None else 'None'))

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Read the block fault boundaries association from a text file
    def readBlockFaultBoundaries(self, faults=[], inDir='./'):
        '''
        Read the block fault boundaries association from a text file and set the patchblock or tentblock
        attribute of the faults in self.blockboundaries accordingly. The format of the filnename is the one
        used in the saveBlockFaultBoundaries function.
        
        Returns:
            * None
        '''

        # Check if the list of blocks and block boundaries are initialized
        if self.Nblocks == 0:
            raise ValueError("Please set the block list before setting the block fault boundaries.")
        
        custom_block_fault_boundaries = {}
        
        # Loop over the faults
        for fault in faults:

            # Read the association from a text file
            with open(os.path.join(inDir, '{}_{}_block_boundaries.txt'.format(self.name.replace(' ', '_'), fault.name.replace(' ', '_'))), 'r') as f:
                
                lines = f.readlines()
                
                if fault.patchType in ('rectangle', 'triangle'):
                    
                    patchblock = []
                    
                    for line in lines[1:]:
                        idx_patch, blockA_name, blockB_name = line.strip().split('\t')
                        blockA = next((b for b in self.blocks if b.name == blockA_name), None)
                        blockB = next((b for b in self.blocks if b.name == blockB_name), None)
                        patchblock.append([blockA, blockB])
                    
                    custom_block_fault_boundaries[fault] = patchblock
                
                elif fault.patchType == 'triangletent':
                    
                    tentblock = []

                    for line in lines[1:]:
                        idx_tent, blockA_name, blockB_name = line.strip().split('\t')
                        blockA = next((b for b in self.blocks if b.name == blockA_name), None)
                        blockB = next((b for b in self.blocks if b.name == blockB_name), None)
                        tentblock.append([blockA, blockB])
                    
                    custom_block_fault_boundaries[fault] = tentblock
        
        # Set the block fault boundaries association using the custom input
        self.setBlockFaultBoundaries(faults=faults,
                                     custom=custom_block_fault_boundaries)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Plot the blocks and block fault boundaries
    def plot(self, figure=134, slip='total', equiv=False, show=True, Map=True, Fault=True,
             norm=None, linewidth=1.0, elinewidth=1., plotPatches2d=True, box=None, 
             shadedtopo=None, view=None, alpha=1.0, shape=(1., 1., 1.), cmap='jet',
             colorbar=True, cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='',
             drawCoastlines=True, expand=0.2, savefig=False, figsize=(None, None), plotBlockBoundaries2d=True,
             npoints=None, edgecolor='k', markersize=5):
        '''
        Plot the available elements of the multiblocks objects.

        Kwargs:
            * figure        : Number of the figure.
            * slip          : What slip to plot
            * equiv         : useless. For consistency between fault objects
            * show          : Show me
            * Norm          : colorbar min and max values
            * linewidth     : Line width in points
            * plotPatches2d : Plot the 3d fault patches on a 2d map as well
            * drawCoastline : Self-explanatory argument...
            * expand        : Expand the map by {expand} degree around the edges
                              of the fault.
            * savefig       : Save figures as eps.
            * plotBlockBoundaries2d : If True, will plot the block boundaries on the 2d map as well (if Map=True). If False, will only plot them on the 3d view.

        Returns:
            * None
        '''

        # Get lon/lat extents
        lonmin = np.min(np.concatenate([block.lon for block in self.blocks])) - expand
        lonmax = np.max(np.concatenate([block.lon for block in self.blocks])) + expand
        latmin = np.min(np.concatenate([block.lat for block in self.blocks])) - expand
        latmax = np.max(np.concatenate([block.lat for block in self.blocks])) + expand

        # Override lon/lat extents if box is provided
        if box is not None:
            assert len(box)==4, 'box must be 4 floats: box = {}'.format(tuple(box))
            lonmin, lonmax, latmin, latmax = box

        # Create a figure
        fig = geoplot(figure=figure,
                      lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax,
                      figsize=figsize, Map=Map, Fault=Fault)

        # Shaded topo
        if shadedtopo is not None:
            fig.shadedTopography(**shadedtopo)

        # Draw the coastlines
        if drawCoastlines:
            fig.drawCoastlines(parallels=None, meridians=None, drawOnFault=True)
        
        # Draw the block fault boundaries and patches
        for fault in self.blockboundaries:

            if Fault or (not Fault and Map and plotPatches2d):

                if fault.patchType in ('rectangle', 'triangle'):
            
                    fig.faultpatches(fault, slip=slip, norm=norm, colorbar=colorbar, alpha=alpha,
                                     cbaxis=cbaxis, cborientation=cborientation, cblabel=cblabel,
                                     plot_on_2d=plotPatches2d, linewidth=elinewidth, cmap=cmap)
                
                elif fault.patchType == 'triangletent':
                    
                    fig.faultTents(fault, slip=slip, norm=norm, colorbar=colorbar, alpha=alpha,
                                   cbaxis=cbaxis, cborientation=cborientation, cblabel=cblabel,
                                   plot_on_2d=plotPatches2d, linewidth=elinewidth, cmap=cmap,
                                   npoints=npoints, edgecolor=edgecolor, markersize=markersize)
            
            if Map and plotBlockBoundaries2d:
                
                color = fault.color if hasattr(fault, 'color') else 'k'
                linewidth = fault.linewidth if hasattr(fault, 'linewidth') else 1.
                linestyle = fault.linestyle if hasattr(fault, 'linestyle') else 'solid'
                    
                fig.faulttrace(fault, color=color, linewidth=linewidth, linestyle=linestyle,
                            zorder=np.sum([fault.N_slip for fault in self.blockboundaries])+10)

        # Draw the blocks
        for block in self.blocks:
            
            if Map and plotBlockBoundaries2d:
            
                color = block.color if hasattr(block, 'color') else 'k'
                linewidth = block.linewidth if hasattr(block, 'linewidth') else 1.
                linestyle = block.linestyle if hasattr(block, 'linestyle') else 'solid'
                fig.blockboundary(block, color=color, linewidth=linewidth, linestyle=linestyle,
                                  zorder=np.sum([fault.N_slip for fault in self.blockboundaries])+9)

        # Savefigs?
        if savefig:
            prefix = self.name.replace(' ','_')
            fig.savefig(prefix+'_{}'.format(slip), ftype='eps')

        # View?
        if view is not None:
            fig.set_view(**view, shape=shape)

        # show
        if show:
            showFig = []
            if Map:
                showFig.append('map')
            if Fault:
                showFig.append('fault')
            fig.show(showFig=showFig)

        # Save the figure
        self.fig = fig

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Save data to file
    def saveData(self, dtype='d', outputDir='.'):
        '''
        Saves the Data in binary files.

        Kwargs:
            * dtype       : Format of the binary data saved. 'd' for double. 'f' for np.float32
            * outputDir   : Directory to save binary data

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing data to file for multiblock {}'.format(self.name))

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
    # Save the Green's functions
    def saveGFs(self, dtype='d', outputDir='.', suffix_rot='rot', suffix_intradef='intradef', suffix_boundary='bound'):
        '''
        Saves the Green's functions in different files.

        Kwargs:
            * dtype           : Format of the binary data saved
                                'd' for double
                                'f' for np.float32
            * outputDir       : Directory to save binary data.
            * suffix_rot      : Suffix for the rotation Green's functions
            * suffix_intradef : Suffix for the internal deformation Green's functions
            * suffix_boundary  : Suffix for the boundary Green's functions
            

        Returns:
            * None
        '''

        # Print stuff
        if self.verbose:
            print('Writing Greens functions to file for multiblock {}'.format(self.name))

        # Loop over the keys in self.G
        for data in self.G.keys():

            # Get the Green's function
            G = self.G[data]

            # Create one file for the rotation
            if 'rotation' in G.keys():
                g = G['rotation'].flatten()
                n = self.name.replace(' ', '_')
                d = data.replace(' ', '_')
                filename = '{}_{}_{}.gf'.format(n, d, suffix_rot)
                g = g.astype(dtype)
                g.tofile(os.path.join(outputDir, filename))
            
            # Create one file for the internal deformation
            if 'intradef' in G.keys():
                g = G['intradef'].flatten()
                n = self.name.replace(' ', '_')
                d = data.replace(' ', '_')
                filename = '{}_{}_{}.gf'.format(n, d, suffix_intradef)
                g = g.astype(dtype)
                g.tofile(os.path.join(outputDir, filename))
            
            # Create one file for the boundary
            if 'boundary' in G.keys():
                g = G['boundary'].flatten()
                n = self.name.replace(' ', '_')
                d = data.replace(' ', '_')
                filename = '{}_{}_{}.gf'.format(n, d, suffix_boundary)
                g = g.astype(dtype)
                g.tofile(os.path.join(outputDir, filename))

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the Green's functions
    def buildGFs(self, data, method='rotation', blockslipdir='sdt', vertical=True, verbose=True, threshold_vertical=1e-3, edksmethod='fortran', Nworkers=None, chunksize=None):
        '''
        Builds the Green's function matrix.

        The Green's function matrix is stored in a dictionary.
        Each entry of the dictionary is named after the corresponding dataset.

        Args:
            * data          : data object (gps, insar, surfaceslip)
        
        Kwargs:
            * method        : method to compute the Green's functions
                                (combination of 'rotation', 'intradef', 'emptyrotation', 'emptyintradef', 'boundary', 'emptyboundary')
                                ex: 'rotation+emptyintradef+boundary'
            * blockslipdir  : Direction of the slip along the patches at the fault boundaries for coseismic slip deficit. Can be any combination of s (strikeslip), d (dipslip), t (tensile).
                                The default is 'sdt' (strikeslip, dipslip, tensile). Might be dangerous to set if you have a mix of vertical and dipping faults.
                                Direction of the slip along the patches at the fault boundaries for coseismic slip deficit.
                                By default, it is set to 'sdt', but only strike and tensile component is used for a vertical fault and only strike and dip component for a dipping fault.
                                You should not touch this except if you really know what you are doing.
            * vertical      : If True, will produce Green's functions for the vertical displacements in a gps object.
            * verbose       : If True, will print out information
            * threshold_vertical : Threshold in degrees to consider a fault patch as vertical.
            * Nworkers      : Number of workers to use for parallel computing. If None, will use all available cores. Only works for python edks
            * edksmethod    : Method to use for the edks computation: "python" or "fortran".

        Returns:
            * None
        '''

        # Data type check
        if data.dtype not in ('gps', 'insar', 'surfaceslip'):
            raise NotImplementedError("Only GNSS, InSAR and surfaceslip data are supported.")

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
        
        # Rotation
        Grot = None
        if 'rotation' in method and not 'emptyrotation' in method:
            Grot = self.rotationGFs(data, verbose=verbose)
        elif 'emptyrotation' in method:
            Grot = self.emptyrotationGFs(data, verbose=verbose)
        
        # Internal deformation
        Gint = None
        if 'intradef' in method and not 'emptyintradef' in method:
            Gint = self.intradefGFs(data, verbose=verbose)
        elif 'emptyintradef' in method:
            Gint = self.emptyintradefGFs(data, verbose=verbose)
            
        # Coseismic slip deficit due to fault boundaries
        Gbound = None
        if 'boundary' in method and not 'emptyboundary' in method:
            Gbound = self.boundaryGFs(data, blockslipdir=blockslipdir, verbose=verbose, threshold_vertical=threshold_vertical, Nworkers=Nworkers, chunksize=chunksize, edksmethod=edksmethod)
        elif 'emptyboundary' in method:
            Gbound = self.emptyboundaryGFs(data, verbose=verbose)
        
        # Check
        if Grot is None and Gint is None and Gbound is None:
            raise NotImplementedError("Method {} not supported".format(method))

        # Build the dictionary
        G = self._buildGFsdict(data, Grot=Grot, Gint=Gint, Gbound=Gbound, vertical=vertical)

        # Separate the Green's functions for each type of data set
        data.setGFsInSource(self, G, vertical=vertical)

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build zero GFs for block rotation
    def emptyrotationGFs(self, data, verbose=True):
        '''
        Build zero rotation GFs.

        Args:
            * data          : Data object (gps, insar, surfaceslip)

        Kwargs:
            * verbose       : If True, will print out information.

        Returns:
            * G             : Dictionnary of GFs
        '''
        
        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building empty blocks rotation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))

        # Create the GF matrices
        if data.dtype in ('insar', 'gps', 'surfaceslip'):
            
            nd = len(data.lon)
            Grot = np.zeros((3, nd, 3*self.Nblocks))
        
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))

        # All done
        return Grot
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the rotation GFs
    def rotationGFs(self, data, verbose=True):
        '''
        Build the Green's functions for blocks rotation.

        Args:
            * data          : Data object (gps or insar)

        Kwargs:
            * verbose       : If True, will print out information.

        Returns:
            * G             : Dictionnary of GFs
        '''
        
        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building blocks rotation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))

        # Create the GF matrices
        if data.dtype in ('insar', 'gps'):
            
            # Initialize the Green's functions
            Grot = np.zeros((3, len(data.lon), 3*self.Nblocks))
            
            for idx, block_ in enumerate(self.blocks):
                
                Grot_ = block_.rotationGFs(data, verbose=False)
                Grot[:, :, 3*idx:3*(idx+1)] = Grot_
        
        elif data.dtype == 'surfaceslip':
            
            # Initialize the Green's functions
            Grot = np.zeros((3, len(data.lon), 3*self.Nblocks))
            
            # If the surfaceslip data are not at block boundaries
            if not hasattr(data, 'blockboundary'):
            
                for idx, block_ in enumerate(self.blocks):
                    
                    Grot_ = block_.rotationGFs(data, verbose=False)
                    Grot[:, :, 3*idx:3*(idx+1)] = Grot_
            
            # If the surfaceslip data are at block boundaries
            else:

                for idx, block_ in enumerate(self.blocks):
                    
                    Grot_ = block_.rotationGFs(data, verbose=False, override_inblock=True)
                    mask_left = [block_.name == b for b in np.array(data.block_list)[:, 0]]
                    mask_right = [block_.name == b for b in np.array(data.block_list)[:, 1]]
                    
                    Grot[:, mask_left, 3*idx:3*(idx+1)] = Grot_[:, mask_left, :]
                    Grot[:, mask_right, 3*idx:3*(idx+1)] = -Grot_[:, mask_right, :]
                
                # Convention for SurfaceSlip orientation
                Grot *= -1
    
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))

        # All done
        return Grot
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build zero GFs for intrablock deformation
    def emptyintradefGFs(self, data, verbose=True):
        ''' 
        Build zero intrablocks deformation GFs.

        Args:
            * data          : Data object (gps, insar, surfaceslip)

        Kwargs:
            * verbose       : If True, will print out information.

        Returns:
            * G             : Dictionnary of GFs
        '''
        
        # Print
        if verbose:           
            print('---------------------------------')
            print('---------------------------------')
            print("Building empty internal blocks deformation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))

        # Create the GF matrices
        if data.dtype in ('insar', 'gps', 'surfaceslip'):
            
            nd = len(data.lon)
            Gint = np.zeros((3, nd, 3*self.Nblocks))
        
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))
    
        # All done
        return Gint
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the internal deformation GFs
    def intradefGFs(self, data, verbose=True):
        '''
        Build the Green's functions for a block internal deformation following the approach of
        B.J. Meade and J.P. Loveless (2009).

        Args:
            * data          : Data object (gps, insar, surfaceslip)

        Kwargs:
            * verbose       : If True, will print out information.

        Returns:
            * G             : GFs
        '''
        
        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building internal blocks deformation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))
        
        # Create the GF matrices
        if data.dtype in ('insar', 'gps'):
            
            # Initialize the Green's functions
            Gint = np.zeros((3, len(data.lon), 3*self.Nblocks))
            
            for idx, block_ in enumerate(self.blocks):
                
                Gint_ = block_.intradefGFs(data, verbose=False)
                Gint[:, :, 3*idx:3*(idx+1)] = Gint_
        
        elif data.dtype == 'surfaceslip':
            
            # Initialize the Green's functions
            Gint = np.zeros((3, len(data.lon), 3*self.Nblocks))
            
            # If the surfaceslip data are not at block boundaries
            if not hasattr(data, 'blockboundary'):
            
                for idx, block_ in enumerate(self.blocks):
                    
                    Gint_ = block_.intradefGFs(data, verbose=False)
                    Gint[:, :, 3*idx:3*(idx+1)] = Gint_
            
            # If the surfaceslip data are at block boundaries
            else:
                
                for idx, block_ in enumerate(self.blocks):
                    
                    Gint_ = block_.intradefGFs(data, verbose=False, override_inblock=True)
                    mask_left = [block_.name == b for b in np.array(data.block_list)[:, 0]]
                    mask_right = [block_.name == b for b in np.array(data.block_list)[:, 1]]
                    
                    Gint[:, mask_left, 3*idx:3*(idx+1)] = Gint_[:, mask_left, :]
                    Gint[:, mask_right, 3*idx:3*(idx+1)] = -Gint_[:, mask_right, :]
                
                 # Convention for SurfaceSlip orientation
                Gint *= -1
        
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))
    
        # All done
        return Gint
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build zero GFs for boundary coseismic slip dficit
    def emptyboundaryGFs(self, data, verbose=True):
        '''
        Build zero GFs for coseismic slip deficit due to locking of fault block boundaries.

        Args:
            * data          : Data object (gps, insar, surfaceslip)

        Kwargs:
            * verbose       : If True, will print out information.

        Returns:
            * G             : Dictionnary of GFs
        '''
        
        # Print
        if verbose:           
            print('---------------------------------')
            print('---------------------------------')
            print("Building empty internal blocks deformation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))

        # Create the GF matrices
        if data.dtype in ('insar', 'gps', 'surfaceslip'):
            
            nd = len(data.lon)
            Gbound = np.zeros((3, nd, 3*self.Nblocks))
        
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))
    
        # All done
        return Gbound
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the GFs for coseismic slip deficit at block boundaries
    def boundaryGFs(self, data, blockslipdir='sdt', verbose=True, threshold_vertical=1e-3, Nworkers=None, chunksize=None, edksmethod='fortran'):
        '''
        Build the Green's functions for the coseismic slip deficit at fault block boundaries
        following the approach of B.J. Meade and J.P. Loveless (2009).

        Args:
            * data          : Data object (gps, insar, surfaceslip)
            * blockslipdir  : Direction of the slip along the patches at the fault boundaries for coseismic slip deficit.
            Can be any combination of s (strikeslip), d (dipslip), t (tensile). By default, it is set to 'sdt',
            but only strike and tensile component is used for a vertical fault and only strike and dip component for a dipping fault.
            You should not touch this except if you really know what you are doing.

        Kwargs:
            * verbose       : If True, will print out information.
            * threshold_vertical : Threshold dip (in degrees) below which a fault patch is considered vertical.
            * Nworkers      : Number of workers to use for parallel computing. If None, will use all available cores. Only works for python edks
            * edksmethod    : Method to use for the edks computation: "python" or "fortran".

        Returns:
            * G             : GFs
        '''
        
        # Print
        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print("Building boundary blocks deformation Green's functions")
            print("for the data set {} of type {}".format(data.name, data.dtype))
        
        # Get block list
        if not hasattr(data, 'block_list'):
            data.getBlockList(self.blocks)
        
        # Create the GF matrices
        if data.dtype in ('insar', 'gps'):
            
            # Initialize the Green's functions
            Gbound = np.zeros((3, len(data.lon), 3*self.Nblocks))
            
            # Loop over the fault boundaries
            for fault in self.blockboundaries:
                
                if fault.patchType in ('rectangle', 'triangle'):
                
                    for idx_patch, fault_patch in enumerate(fault.patchll):
                        
                        if verbose:
                            sys.stdout.write('\r Building boundary {}: patch {} / {} '.format(fault.name, idx_patch+1,len(fault.patchll)))
                            sys.stdout.flush()
                            if idx_patch == len(fault.patchll)-1:
                                print('\r Building boundary {}: patch {} / {} '.format(fault.name, idx_patch+1,len(fault.patchll)))

                        # Get the two bounding blocks
                        blockleft = fault.patchblock[idx_patch][0]
                        blockright = fault.patchblock[idx_patch][1]

                        idx_blockleft = np.argwhere(np.array(self.blocks) == blockleft)[0][0]
                        idx_blockright = np.argwhere(np.array(self.blocks) == blockright)[0][0]
                        
                        # We only consider the data points on the two bounding blocks
                        mask_blocks = np.logical_not(np.isin(data.block_list, [self.blocks[idx_blockleft].name,
                                                                            self.blocks[idx_blockright].name]))
                        
                        ######################
                        # Patch geometry info.
                        ######################
                        
                        # Strike of the fault patch
                        strike = fault.getpatchstrike(idx_patch, method='xy')
                        
                        # Dip of the fault patch
                        dip = fault.getpatchdip(idx_patch)
                        
                        # Center of the fault patch
                        lon_center, lat_center, depth_center = fault.getcenter(fault_patch)
                        lon_center = lon_center if lon_center < 180 else lon_center - 360
                        
                        ######################
                        # Compute
                        ######################

                        # Compute matrix Pf
                        if np.abs(dip - np.pi/2) < np.deg2rad(threshold_vertical):
                            Pf = np.array([[np.sin(strike), np.cos(strike), 0.],
                                        [0., 0., 0.],
                                        [np.cos(strike), -np.sin(strike), 0.]])
                            
                        else:
                            Pf = np.array([[np.sin(strike), np.cos(strike), 0.],
                                            [np.cos(strike) / np.cos(dip), -np.sin(strike) / np.cos(dip), 0.],
                                            [0., 0., 0.]])
                        
                        # Compute the rotation GFs
                        # to compute the differential velocity on the patch predicted by
                        # the relative rotation of the two bounding blocks
                
                        x_, y_, z_ = llh2xyz(lon=lon_center, lat=lat_center, height=-depth_center)
                        lon_ = np.deg2rad(lon_center)
                        lat_ = np.deg2rad(lat_center)
                        
                        Pv = np.array([[-np.sin(lon_), np.cos(lon_), 0.],
                                        [-np.sin(lat_) * np.cos(lon_), -np.sin(lat_) * np.sin(lon_), np.cos(lat_)],
                                        [np.cos(lat_) * np.cos(lon_), np.cos(lat_) * np.sin(lon_), np.sin(lat_)]])
                        
                        Gb = np.array([[0., z_, -y_],
                                    [-z_, 0., x_],
                                    [y_, -x_, 0.]])
                        
                        Gb *= MAS2RAD
                        
                        Gdv = np.zeros((3, 3*self.Nblocks))
                        Gdv[:, 3*idx_blockleft:3*(idx_blockleft+1)] = Gb
                        Gdv[:, 3*idx_blockright:3*(idx_blockright+1)] = -Gb
                        
                        # Surface displacement for a given patch at the data location
                        slip = [1. if 's' in blockslipdir else 0.,
                                1. if 'd' in blockslipdir else 0.,
                                1. if 't' in blockslipdir else 0.]
                        ss, ds, ts = fault.slip2dis(data=data, patch=idx_patch, slip=slip)
                        G0 = np.stack((ss, ds, ts), axis=2)
                        
                        # Compute Green's Functions
                        Gbound_ = G0 @ Pf @ Pv @ Gdv
                        
                        # Reshape for consistency with the other GFs in csi
                        Gbound_ = np.transpose(Gbound_, (1, 0, 2))
                        
                        # Mask the data points that are not on the two bounding blocks
                        Gbound_[:, mask_blocks, :] = 0.

                        # Add the contribution of this patch
                        Gbound += Gbound_
                
                elif fault.patchType == 'triangletent':
                    
                    # Check if the GF were computed for this fault already
                    # assert hasattr(fault, 'G') and fault.G is not None, f"The Green's functions for the boundary {fault.name} were not computed. Please compute them first."
                    # assert data.name in fault.G.keys(), f"The Green's functions for the boundary {fault.name} were not computed for the data set {data.name}. Please compute them first."
                    # assert 'strikeslip' in fault.G[data.name].keys() and 'dipslip' in fault.G[data.name].keys() and 'tensile' in fault.G[data.name].keys(), f"The Green's functions for the boundary {fault.name} and data set {data.name} should have keys 'strikeslip', 'dipslip' and 'tensileslip'. Please compute them first."
                    
                    ######################
                    # Build the fault Green's functions
                    ######################
                    
                    # Check if we can find kernels
                    if not hasattr(fault, 'kernelsEDKS'):
                        if verbose:
                            print('---------------------------------')
                            print('---------------------------------')
                            print(' WARNING WARNING WARNING WARNING ')
                            print('   Kernels for computation of')
                            print('stratified Greens functions not ')
                            print('    set in {}.kernelsEDKS'.format(fault.name))
                            print('   Looking for default kernels')
                            print('---------------------------------')
                            print('---------------------------------')
                        fault.kernelsEDKS = 'kernels.edks'
                    stratKernels = fault.kernelsEDKS
                    assert os.path.isfile(stratKernels), 'Kernels for EDKS not found...: {}'.format(stratKernels)

                    # Show me
                    if verbose:
                        print('Kernels used: {}'.format(stratKernels))

                    # Check if we can find mention of the spacing between points
                    if not hasattr(fault, 'sourceSpacing') and not hasattr(fault, 'sourceNumber')\
                            and not hasattr(fault, 'sourceArea'):
                        print('---------------------------------')
                        print('---------------------------------')
                        print(' WARNING WARNING WARNING WARNING ')
                        print('  Cannot find sourceSpacing nor  ')
                        print('   sourceNumber nor sourceArea   ')
                        print('         for stratified          ')
                        print('   Greens function computation   ')
                        print('           computation           ')
                        print('          Dying here...          ')
                        print('              Arg...             ')
                        sys.exit(1)
                    
                    # Receivers to meters
                    xr = data.x * 1000.
                    yr = data.y * 1000.
                    
                    # Prefix for the files
                    prefix = '{}_{}'.format(fault.name.replace(' ','-'), data.name.replace(' ','-'))
                    
                    # Check something
                    if not hasattr(fault, 'keepTrackOfSources'):
                        fault.keepTrackOfSources = True
                        
                    # If we have already done that step
                    if fault.keepTrackOfSources and hasattr(fault, 'edksSources'):
                        if verbose:
                            print('Get sources from saved sources')
                        Ids, xs, ys, zs, strike, dip, Areas = fault.edksSources[:7]
                    # Else, drop sources in the patches
                    else:
                        if verbose:
                            print('Subdividing patches into point sources')
                        Ids, xs, ys, zs, strike, dip, Areas = Patches2Sources(fault, verbose=verbose, Nworkers=Nworkers)
                        # All these guys need to be in meters
                        xs *= 1000.
                        ys *= 1000.
                        zs *= 1000.
                        Areas *= 1e6
                        # Strike and dip in degrees
                        strike = strike*180./np.pi
                        dip = dip*180./np.pi
                        # Keep track?
                        fault.edksSources = [Ids, xs, ys, zs, strike, dip, Areas]

                    # Get the slip vector
                    # If saved, good
                    if fault.keepTrackOfSources and hasattr(fault, 'edksSources') and (len(fault.edksSources)>7):
                        slip = fault.edksSources[7]
                    # Else, we have to re-organize the Ids from facet to nodes
                    else:
                        if hasattr(fault, 'homogeneousStrike'):
                            homS = fault.homogeneousStrike
                        else:
                            homS = False
                        if hasattr(fault, 'homogeneousDip'):
                            homD = fault.homogeneousDip
                        else:
                            homD = False
                        fault.Facet2Nodes(homogeneousStrike=homS, homogeneousDip=homD)
                        Ids, xs, ys, zs, strike, dip, Areas, slip = fault.edksSources
                    
                    # Informations
                    if verbose:
                        print('{} sources for {} patches and {} data points'.format(len(Ids), len(fault.patch), len(xr)))
                    
                    # Create interpolation class if python method is selected
                    if edksmethod in ('python'):
                        inter = interpolateEDKS(stratKernels, verbose=verbose)
                        inter.readHeader()
                        inter.readKernel()
                    
                    # For summing subsources
                    source_map = {
                        Id: np.flatnonzero(Ids == Id) for Id in np.unique(Ids)
                        }
                    
                    # Strike slip
                    if verbose:
                        print('Running Strike Slip component for data set {}'.format(data.name))
                    
                    if edksmethod in ('fortran'):
                        Gss = _parallel_edks_sources(
                    Ids,
                    xs,
                    ys,
                    zs,
                    strike,
                    dip,
                    np.zeros_like(strike),
                    slip,
                    Areas,
                    xr,
                    yr,
                    stratKernels,
                    prefix + "_SS",
                    fault.cleanUp,
                    verbose,
                    nworkers=Nworkers,
                    chunk_size=chunksize,
                )
                    elif edksmethod in ('python'):
                        iGss = np.array(inter.interpolate(xs, ys, zs,
                                                        strike*np.pi/180., dip*np.pi/180., np.zeros(dip.shape),
                                                        Areas, slip, 
                                                        xr, yr, method='linear', tensile=False, Nworkers=Nworkers))    
                    
                        if verbose:
                            print('Summing sub-sources...')
                        Gss = np.zeros((3, iGss.shape[1], np.unique(Ids).shape[0]))
                        for Id, idx in source_map.items():
                            Gss[:, :, Id] = np.sum(iGss[:, :, idx], axis=2)
                            Gss[:,:,Id] = np.sum(iGss[:, :, np.flatnonzero(Ids==Id)], axis=2)
                        del iGss
                    
                    # Dip slip
                    if verbose:
                        print('Running Dip Slip component for data set {}'.format(data.name))
                        
                    if edksmethod in ('fortran'):
                        Gds = _parallel_edks_sources(
                            Ids,
                            xs,
                            ys,
                            zs,
                            strike,
                            dip,
                            np.ones_like(strike) * 90.,
                            slip,
                            Areas,
                            xr,
                            yr,
                            stratKernels,
                            prefix + "_DS",
                            fault.cleanUp,
                            verbose,
                            nworkers=Nworkers,
                            chunk_size=chunksize,
                        )
                    elif edksmethod in ('python'):
                        iGds = np.array(inter.interpolate(xs, ys, zs,
                                                        strike*np.pi/180., dip*np.pi/180., np.ones(dip.shape)*np.pi/2.,
                                                        Areas, slip, 
                                                        xr, yr, method='linear', tensile=False, Nworkers=Nworkers))
                
                        if verbose:
                            print('Summing sub-sources...')
                        Gds = np.zeros((3, iGds.shape[1], np.unique(Ids).shape[0]))
                        for Id, idx in source_map.items():
                            Gds[:,:,Id] = np.sum(iGds[:,:,idx], axis=2)
                        del iGds
                    
                    # Tensile slip
                    if verbose:
                        print('Running Tensile component for data set {}'.format(data.name))
                    if edksmethod in ('fortran'):
                        Gts = _parallel_edks_sources(
                            Ids,
                            xs,
                            ys,
                            zs,
                            strike,
                            dip,
                            np.ones_like(strike) * 0.,
                            slip,
                            Areas,
                            xr,
                            yr,
                            stratKernels,
                            prefix + "_TS",
                            fault.cleanUp,
                            verbose,
                            nworkers=Nworkers,
                            chunk_size=chunksize,
                            tensile=True
                        )
                    elif edksmethod in ('python'):
                        iGts = np.array(inter.interpolate(xs, ys, zs,
                                                        strike*np.pi/180., dip*np.pi/180., np.zeros(dip.shape),
                                                        Areas, slip, 
                                                        xr, yr, method='linear', tensile=True, Nworkers=Nworkers))
                        if verbose:
                            print('Summing sub-sources...')
                        Gts = np.zeros((3, iGts.shape[1], np.unique(Ids).shape[0]))
                        for Id, idx in source_map.items():
                            Gts[:, :, Id] = np.sum(iGts[:, :, idx], axis=2)
                        del iGts
                    
                    ##############################
                    # Loop over the tents
                    ##############################
                    
                    for idx_tent, fault_tent in enumerate(fault.tentll):
                        
                        if verbose:
                            sys.stdout.write('\r Building boundary {}: tent {} / {} '.format(fault.name, idx_tent+1, fault.numtent))
                            sys.stdout.flush()
                            if idx_tent == len(fault.tent)-1:
                                print('\r Building boundary {}: tent {} / {} '.format(fault.name, idx_tent+1,len(fault.tent)))
                        
                        # Get the two bounding blocks
                        blockleft = fault.tentblock[idx_tent][0]
                        blockright = fault.tentblock[idx_tent][1]

                        idx_blockleft = np.argwhere(np.array(self.blocks) == blockleft)[0][0]
                        idx_blockright = np.argwhere(np.array(self.blocks) == blockright)[0][0]
                        
                        # We only consider the data points on the two bounding blocks
                        mask_blocks = np.logical_not(np.isin(data.block_list, [self.blocks[idx_blockleft].name,
                                                                               self.blocks[idx_blockright].name]))
                    
                        # Strike of the fault tent
                        strike = fault.gettentstrike(idx_tent, method='xy')
                        
                        # Dip of the fault tent
                        dip = fault.gettentdip(idx_tent)
                        
                        # Center of the fault patch
                        lon_tent, lat_tent, depth_tent = fault_tent
                        lon_tent = lon_tent if lon_tent < 180 else lon_tent - 360

                        # Compute matrix Pf
                        if np.abs(dip - np.pi/2) < np.deg2rad(threshold_vertical):
                            Pf = np.array([[np.sin(strike), np.cos(strike), 0.],
                                        [0., 0., 0.],
                                        [np.cos(strike), -np.sin(strike), 0.]])
                            
                        else:
                            Pf = np.array([[np.sin(strike), np.cos(strike), 0.],
                                            [np.cos(strike) / np.cos(dip), -np.sin(strike) / np.cos(dip), 0.],
                                            [0., 0., 0.]])
                        
                        # Compute the rotation GFs
                        # to compute the differential velocity on the tent predicted by
                        # the relative rotation of the two bounding blocks
                
                        x_, y_, z_ = llh2xyz(lon=lon_tent, lat=lat_tent, height=-depth_tent)
                        lon_ = np.deg2rad(lon_tent)
                        lat_ = np.deg2rad(lat_tent)
                        
                        Pv = np.array([[-np.sin(lon_), np.cos(lon_), 0.],
                                        [-np.sin(lat_) * np.cos(lon_), -np.sin(lat_) * np.sin(lon_), np.cos(lat_)],
                                        [np.cos(lat_) * np.cos(lon_), np.cos(lat_) * np.sin(lon_), np.sin(lat_)]])
                        
                        Gb = np.array([[0., z_, -y_],
                                    [-z_, 0., x_],
                                    [y_, -x_, 0.]])
                        
                        Gb *= MAS2RAD
                        
                        Gdv = np.zeros((3, 3*self.Nblocks))
                        Gdv[:, 3*idx_blockleft:3*(idx_blockleft+1)] = Gb
                        Gdv[:, 3*idx_blockright:3*(idx_blockright+1)] = -Gb
                        
                        # Surface displacement for a given tent at the data location
                        if 's' in blockslipdir:
                            ss = Gss[:, :, idx_tent]
                        else:
                            ss = np.zeros((3, len(data.lon)))
                        
                        if 'd' in blockslipdir: 
                            ds = Gds[:, :, idx_tent]
                        else:
                            ds = np.zeros((3, len(data.lon)))
                        
                        if 't' in blockslipdir:
                            ts = Gts[:, :, idx_tent]
                        else:
                            ts = np.zeros((3, len(data.lon)))
                        
                        G0 = np.stack((ss, ds, ts), axis=2)
                        
                        # Compute Green's Functions
                        Gbound_ = G0 @ Pf @ Pv @ Gdv
                        
                        # Reshape for consistency with the other GFs in csi
                        # Gbound_ = np.transpose(Gbound_, (1, 0, 2))
                        
                        # Mask the data points that are not on the two bounding blocks
                        Gbound_[:, mask_blocks, :] = 0.

                        # Add the contribution of this tent
                        Gbound += Gbound_
                           
            # Set backslip
            Gbound *= -1.
        
        elif data.dtype == 'surfaceslip':
            
            # Initialize the Green's functions
            Gbound = np.zeros((3, len(data.lon), 3*self.Nblocks))
            
            # If the surfaceslip data are at block boundaries
            if hasattr(data, 'blockboundary'):
                
                # Speak to me
                if verbose:
                    print("Building boundary {}".format(data.blockboundary.name))
    
                    print('WARNING: Surfaceslip at a block boundary which is a dipping fault is not well defined.')
                    print('         Be sure you know what you are doing when using this feature!')
                
                for idx, block_ in enumerate(self.blocks):
                    
                    Gbound_ = block_.rotationGFs(data, verbose=False, override_inblock=True)
                    mask_left = [block_.name == b for b in np.array(data.block_list)[:, 0]]
                    mask_right = [block_.name == b for b in np.array(data.block_list)[:, 1]]
                    
                    Gbound[:, mask_left, 3*idx:3*(idx+1)] = Gbound_[:, mask_left, :]
                    Gbound[:, mask_right, 3*idx:3*(idx+1)] = -Gbound_[:, mask_right, :]
            
            # Set backslip
            Gbound *= -1.
            
            # Convention for SurfaceSlip orientation
            Gbound *= -1
            
        else:
            
            raise NotImplementedError("Data type {} not supported".format(data.type))
    
        # All done
        return Gbound
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    def GetLongTermSlipRate(self, threshold_vertical=1e-3, redoG=True):
        '''
        Computes the consistent kinematic slip rate on the fault boundaries due to relative block motion.
        For each fault, it is stored in fault.longterm_slip. The Green's functions for the long-term slip rate are stored in fault.Glongterm_slip.
        
        Kwargs:
            * threshold_vertical : Threshold dip (in degrees) below which a fault patch is considered vertical
            * redoG : If True, recompute the Green's functions for the long-term slip rate.
        
        Returns:
            * None
        '''
        
        # Loop over the fault boundaries
        for fault in self.blockboundaries:
            
            # Initialize the longterm slip rate GF
            if hasattr(fault, 'Glongterm_slip') and not redoG:
                
                G = fault.Glongterm_slip
                
            else:
                
                G = np.zeros((fault.N_slip, 3, 3*self.Nblocks))
                
                if fault.patchType in ('rectangle', 'triangle'):
            
                    for idx_patch, fault_patch in enumerate(fault.patchll):
                        
                        # Get the two bounding blocks
                        blockleft = fault.patchblock[idx_patch][0]
                        blockright = fault.patchblock[idx_patch][1]

                        idx_blockleft = np.argwhere(np.array(self.blocks) == blockleft)[0][0]
                        idx_blockright = np.argwhere(np.array(self.blocks) == blockright)[0][0]
                        
                        ######################
                        # Patch geometry info.
                        ######################
                        
                        # Strike of the (rectangular) fault patch
                        strike = fault.getpatchstrike(idx_patch, method='xy')
                        
                        # Dip of the (rectangular) fault patch
                        dip = fault.getpatchdip(idx_patch)
                        
                        # Center of the (rectangular) fault patch
                        lon_center, lat_center, depth_center = fault.getcenter(fault_patch)
                        lon_center = lon_center if lon_center < 180 else lon_center - 360
                        
                        ######################
                        # Compute
                        ######################
                        
                        # Compute matrix Pf
                        if np.abs(dip - np.pi/2) < np.deg2rad(threshold_vertical):
                            Pf = np.array([[np.sin(strike), np.cos(strike), 0.],
                                        [0., 0., 0.],
                                            [np.cos(strike), -np.sin(strike), 0.]])
                        
                        else:
                            Pf = np.array([[np.sin(strike), np.cos(strike), 0.],
                                            [np.cos(strike) / np.cos(dip), -np.sin(strike) / np.cos(dip), 0.],
                                            [0., 0., 0.]])
                    
                        # Compute the rotation GFs
                        # to compute the differential velocity on the patch predicted by
                        # the relative rotation of the two bounding blocks
                
                        x_, y_, z_ = llh2xyz(lon=lon_center, lat=lat_center, height=-depth_center)
                        lon_ = np.deg2rad(lon_center)
                        lat_ = np.deg2rad(lat_center)
                        
                        Pv = np.array([[-np.sin(lon_), np.cos(lon_), 0.],
                                    [-np.sin(lat_) * np.cos(lon_), -np.sin(lat_) * np.sin(lon_), np.cos(lat_)],
                                    [np.cos(lat_) * np.cos(lon_), np.cos(lat_) * np.sin(lon_), np.sin(lat_)]])
                        
                        Gb = np.array([[0., z_, -y_],
                                        [-z_, 0., x_],
                                        [y_, -x_, 0.]])
                        
                        Gdv = np.zeros((3, 3*self.Nblocks))
                        Gdv[:, 3*idx_blockleft:3*(idx_blockleft+1)] = Gb
                        Gdv[:, 3*idx_blockright:3*(idx_blockright+1)] = -Gb

                        G[idx_patch, :, :] = Pf @ Pv @ Gdv
                
                elif fault.patchType == 'triangletent':
                    
                    for idx_tent, fault_tent in enumerate(fault.tentll):
                        
                        # Get the two bounding blocks
                        blockleft = fault.tentblock[idx_tent][0]
                        blockright = fault.tentblock[idx_tent][1]

                        idx_blockleft = np.argwhere(np.array(self.blocks) == blockleft)[0][0]
                        idx_blockright = np.argwhere(np.array(self.blocks) == blockright)[0][0]

                        # Strike of the fault tent
                        strike = fault.gettentstrike(idx_tent, method='xy')
                        
                        # Dip of the fault tent
                        dip = fault.gettentdip(idx_tent)
                        
                        # Center of the fault patch
                        lon_tent, lat_tent, depth_tent = fault_tent
                        lon_tent = lon_tent if lon_tent < 180 else lon_tent - 360

                        # Compute matrix Pf
                        if np.abs(dip - np.pi/2) < np.deg2rad(threshold_vertical):
                            Pf = np.array([[np.sin(strike), np.cos(strike), 0.],
                                        [0., 0., 0.],
                                        [np.cos(strike), -np.sin(strike), 0.]])
                            
                        else:
                            Pf = np.array([[np.sin(strike), np.cos(strike), 0.],
                                            [np.cos(strike) / np.cos(dip), -np.sin(strike) / np.cos(dip), 0.],
                                            [0., 0., 0.]])
                        
                        # Compute the rotation GFs
                        # to compute the differential velocity on the tent predicted by
                        # the relative rotation of the two bounding blocks
                
                        x_, y_, z_ = llh2xyz(lon=lon_tent, lat=lat_tent, height=-depth_tent)
                        lon_ = np.deg2rad(lon_tent)
                        lat_ = np.deg2rad(lat_tent)
                        
                        Pv = np.array([[-np.sin(lon_), np.cos(lon_), 0.],
                                        [-np.sin(lat_) * np.cos(lon_), -np.sin(lat_) * np.sin(lon_), np.cos(lat_)],
                                        [np.cos(lat_) * np.cos(lon_), np.cos(lat_) * np.sin(lon_), np.sin(lat_)]])
                        
                        Gb = np.array([[0., z_, -y_],
                                    [-z_, 0., x_],
                                    [y_, -x_, 0.]])
                        
                        Gdv = np.zeros((3, 3*self.Nblocks))
                        Gdv[:, 3*idx_blockleft:3*(idx_blockleft+1)] = Gb
                        Gdv[:, 3*idx_blockright:3*(idx_blockright+1)] = -Gb
                        
                        G[idx_tent, :, :] = Pf @ Pv @ Gdv
                    
                fault.Glongterm_slip = G
            
            Ω = np.concatenate([[block_.omega_x, block_.omega_y, block_.omega_z] if None not in [block_.omega_x, block_.omega_y, block_.omega_z] else [0., 0., 0.]
                                for block_ in self.blocks])
            fault.longterm_slip = G @ Ω

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Compute the coupling along faults boundaries
    def GetCoupling(self, threshold_vertical=1e-3, redoG=True, slipdir='s'):
        '''
        Computes the coupling along the fault boundaries as the ratio between the long-term slip rate and the plate convergence rate.
        For each fault, it is stored in fault.coupling.

        Kwargs:
            * threshold_vertical : Threshold dip (in degrees) below which a fault patch is considered vertical
            * redoG : If True, recompute the Green's functions for the long-term slip rate.
            * slipdir : direction of the slip used to compute the coupling. Can be 's', 'd' or 't'.

        Returns:
            * None
        '''
        
        # Get the long-term slip rate
        self.GetLongTermSlipRate(threshold_vertical=threshold_vertical, redoG=redoG)
        
        # Loop over the fault boundaries
        for fault in self.blockboundaries:
            
            # Check if there is slip inverted 
            if hasattr(fault, 'slipdir') and slipdir in fault.slipdir:
            
                # Get coupling and save it in the fault structure
                coupling = 1 - (fault.slip / fault.longterm_slip)
                
                if slipdir == 's':
                    fault.coupling = coupling[:, 0]
                elif slipdir == 'd':
                    fault.coupling = coupling[:, 1]
                elif slipdir == 't':
                    fault.coupling = coupling[:, 2]

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Build the Green's functions dictionary
    def _buildGFsdict(self, data, Grot=None, Gint=None, Gbound=None, vertical=True):
        '''
        Some ordering of the Gfs to make the computation routines simpler.

        Args:
            * data          : instance of data
            * Grot          : Block rotation Green's functions
            * Gint          : Block internal deformation Green's functions
            * Gbound        : Block boundary Green's functions (coseismic slip deficit)

        Kwargs:
            * vertical      : If true, assumes verticals are used for the GPS case

        Returns:
            * G             : Dictionary of GFs
        '''

        # Check if vertical
        Ncomp = 3
        if not vertical:
            Ncomp = 2

        # Block rotation
        if Grot is not None:

            # Consider vertical or not
            Grot = Grot[:Ncomp, :, :]
            
            # Size
            Npoints = Grot.shape[1]
            Nparam = Grot.shape[2]
            Ndata = Ncomp * Npoints

            # Check data format
            if data.dtype == 'gps':
                Grot = Grot.reshape((Ndata, Nparam))

            elif data.dtype in ('insar', 'surfaceslip'):
                # Project the Green's functions onto the LOS direction
                Grot_los = []
                for k in range(Npoints):
                    Grot_los.append((data.los[k, :] @ Grot[:, k, :]).tolist())
                Grot = np.array(Grot_los).reshape((Npoints, Nparam))
        
        # Block boundary
        if Gbound is not None:
            
            # Consider vertical or not
            Gbound = Gbound[:Ncomp, :, :]
            
            # Size
            Npoints = Gbound.shape[1]
            Nparam = Gbound.shape[2]
            Ndata = Ncomp * Npoints
            
            # Check data format
            if data.dtype == 'gps':
                Gbound = Gbound.reshape((Ndata, Nparam))
            elif data.dtype in ('insar', 'surfaceslip'):
                # Project the Green's functions onto the LOS direction
                Gbound_los = []
                for k in range(Npoints):
                    Gbound_los.append((data.los[k, :] @ Gbound[:, k, :]).tolist())
                Gbound = np.array(Gbound_los).reshape((Npoints, Nparam))
        
        # Block internal deformation
        if Gint is not None:
            
            # Consider vertical or not
            Gint = Gint[:Ncomp, :, :]
            
            # Size
            Npoints = Gint.shape[1]
            Nparam = Gint.shape[2]
            Ndata = Ncomp * Npoints
            
            # Check data format
            if data.dtype == 'gps':
                Gint = Gint.reshape((Ndata, Nparam))
            elif data.dtype in ('insar', 'surfaceslip'):
                # Project the Green's functions onto the LOS direction
                Gint_los = []
                for k in range(Npoints):
                    Gint_los.append((data.los[k, :] @ Gint[:, k, :]).tolist())
                Gint = np.array(Gint_los).reshape((Npoints, Nparam))
        
        # Create the dictionary
        G = {'rotation': Grot, 'intradef': Gint, 'boundary': Gbound}

        # All done
        return G
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the Green's functions
    def setGFs(self, data,
               rotation=[None, None, None],
               intradef=[None, None, None],
               boundary=[None, None, None],
               vertical=False, synthetic=False):
        '''
        Stores the input Green's functions matrices into the block source structure.

        These GFs are organized in a dictionary structure in self.G
        Entries of self.G are the data set names (data.name).
            Entries of self.G[data.name] are 'rotation'.

        If you provide GPS GFs, those are organised with E, N and U in lines

        If you provide InSAR GFs, these need to be projected onto the
        LOS direction already.

        Args:
            * data          : data structure
            * rotation      : Green's functions for block rotation displacements.
            * intradef      : Green's functions for internal block deformation displacements.
            * boundary      : Green's functions for coseismic slip deficit at block boundaries.
            
        Returns:
            * None
        '''

        # Get the number of data per point
        if data.dtype in ('insar'):
            data.obs_per_station = 1
            
        elif data.dtype in ('gps'):
            data.obs_per_station = 0
            # Check components
            if not np.isnan(data.vel_enu[:, 0]).any():
                data.obs_per_station += 1
            if not np.isnan(data.vel_enu[:, 1]).any():
                data.obs_per_station += 1
            if vertical:
                if np.isnan(data.vel_enu[:, 2]).any():
                    raise ValueError('Vertical can only be true if all stations have vertical components')
                data.obs_per_station += 1

        # Create the storage for that dataset
        if data.name not in self.G.keys():
            self.G[data.name] = {}

        # Initializes the data vector
        if not synthetic:
            if data.dtype in ('insar', 'surfaceslip'):
                self.d[data.name] = data.vel
                vertical = True # Always true for InSAR
            elif data.dtype == 'gps':
                if vertical:
                    self.d[data.name] = data.vel_enu.T.flatten()
                else:
                    self.d[data.name] = data.vel_enu[:, :2].T.flatten()
                self.d[data.name] = self.d[data.name][np.isfinite(self.d[data.name])]

        if len(rotation) == 3: # gnss case
            
            # Block rotation
            E_rot = rotation[0]
            N_rot = rotation[1]
            U_rot = rotation[2]
            rot = []
            nd = 0
            
            if (E_rot is not None) and (N_rot is not None):
                d, m = E_rot.shape
                rot.append(E_rot)
                rot.append(N_rot)
                nd += 2
            if (U_rot is not None):
                d, m = U_rot.shape
                rot.append(U_rot)
                nd += 1
            if nd > 0:
                rot = np.array(rot)
                rot = rot.reshape((nd*d, m))
                self.G[data.name]['rotation'] = rot
            
            # Boundary coseismic slip deficit
            E_bound = boundary[0]
            N_bound = boundary[1]
            U_bound = boundary[2]
            bound = []
            nd = 0
            
            if (E_bound is not None) and (N_bound is not None):
                d, m = E_bound.shape
                bound.append(E_bound)
                bound.append(N_bound)
                nd += 2
            if (U_bound is not None):
                d, m = U_bound.shape
                bound.append(U_bound)
                nd += 1
            if nd > 0:
                bound = np.array(bound)
                bound = bound.reshape((nd*d, m))
                self.G[data.name]['boundary'] = bound
            
            # Internal block deformation
            E_intra = intradef[0]
            N_intra = intradef[1]
            U_intra = intradef[2]
            intra = []
            nd = 0
            
            if (E_intra is not None) and (N_intra is not None):
                d, m = E_intra.shape
                intra.append(E_intra)
                intra.append(N_intra)
                nd += 2
            if (U_intra is not None):
                d, m = U_intra.shape
                intra.append(U_intra)
                nd += 1
            if nd > 0:
                intra = np.array(intra)
                intra = intra.reshape((nd*d, m))
                self.G[data.name]['intradef'] = intra
            
        elif len(rotation) == 1: # insar case
            
            # Block rotation
            rot = rotation[0]
            if rot is not None:
                self.G[data.name]['rotation'] = rot
            
            # Internal block deformation
            intra = intradef[0]
            if intra is not None:
                self.G[data.name]['intradef'] = intra
            
            # Boundary coseismic slip deficit
            bound = boundary[0]
            if bound is not None:
                self.G[data.name]['boundary'] = bound

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set the Green's functions from file
    def setGFsFromFile(self, data, rotation=None, intradef=None, boundary=None,
                       custom=None, vertical=False, dtype='d', inDir='.'):
        '''
        Sets the Green's functions reading binary files. Be carefull, these have to be in the
        good format (i.e. if it is GPS, then GF are E, then N, then U, optional, and
        if insar, GF are projected already). Basically, it will work better if
        you have computed the GFs using csi...

        Args:
            * data          : Data object

        kwargs:
            * rotation      : File containing the Green's functions for block rotation displacements.
            * intradef      : File containing the Green's functions for internal block deformation displacements.
            * boundary      : File containing the Green's functions for coseismic slip deficit at block boundaries.
            * vertical      : Deal with the UP component (gps: default is false, insar: it will be true anyway).
            * dtype         : Type of binary data. 'd' for double/float64. 'f' for np.float32

        Returns:
            * None
        '''
        
        # Check filenames
        if rotation is None:
            if os.path.isfile(os.path.join(inDir,'{}_{}_rot.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                rotation = os.path.join(inDir, '{}_{}_rot.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))
        
        if intradef is None:
            if os.path.isfile(os.path.join(inDir,'{}_{}_intradef.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                intradef = os.path.join(inDir, '{}_{}_intradef.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))
        
        if boundary is None:
            if os.path.isfile(os.path.join(inDir,'{}_{}_bound.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))):
                boundary = os.path.join(inDir, '{}_{}_bound.gf'.format(self.name.replace(' ','_'), 
                                                   data.name.replace(' ','_')))

        # Load the block rotation Green's functions
        Grot = None
        if rotation is not None:
            
            # Talk to me
            if self.verbose:
                print('---------------------------------')
                print('---------------------------------')
                print("Set up Green's functions for block {}".format(self.name))
                print("and data {} from files: ".format(data.name))
                print("     rotation: {}".format(rotation))    
            
            Grot = np.fromfile(rotation, dtype=dtype)
            ndl = int(Grot.shape[0])
            Grot = Grot.reshape((int(ndl/(3*self.Nblocks)), 3*self.Nblocks))
        
        # Load the boundary coseismic slip deficit Green's functions
        Gbound = None
        if boundary is not None:
            
            # Talk to me
            if self.verbose:
                print("     boundary: {}".format(boundary))
            
            Gbound = np.fromfile(boundary, dtype=dtype)
            ndl = int(Gbound.shape[0])
            Gbound = Gbound.reshape((int(ndl/(3*self.Nblocks)), 3*self.Nblocks))
            
        # Load the internal deformation Green's functions
        Gint = None
        if intradef is not None:
            
            # Talk to me
            if self.verbose:
                print("     intradef: {}".format(intradef))
            
            Gint = np.fromfile(intradef, dtype=dtype)
            ndl = int(Gint.shape[0])
            Gint = Gint.reshape((int(ndl/(3*self.Nblocks)), 3*self.Nblocks))

        # Create the dictionary
        G = {'rotation': Grot, 'intradef': Gint, 'boundary': Gbound}

        # The dataset sets the Green's functions itself
        data.setGFsInSource(self, G, vertical=vertical)

        # If custom
        if custom is not None:
            self.setCustomGFs(data, custom)

        # all done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Set custom Green's functions
    def setCustomGFs(self, data, G):
        '''
        Sets a custom Green's Functions matrix in the G dictionary.

        Args:
            * data          : Data concerned by the Green's function
            * G             : Green's function matrix

        Returns:
            * None
        '''

        # Check
        if not hasattr(self, 'G'):
            self.G = {}

        # Check
        if not data.name in self.G.keys():
            self.G[data.name] = {}

        # Set
        self.G[data.name]['custom'] = G

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Assemble the Green's functions
    def assembleGFs(self, datas, polys=None, verbose=True, blockcomponent='rotation+intradef+boundary',
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
            
            * blockcomponent   : string or dictionary
                Which block components to consider in the assembly.
                
                - If blockcomponent is a string, all blocks will have the same components considered in the assembly.
                   Options are 'rotation', 'intradef', 'boundary' or any combination (e.g. 'rotation+intradef').
                   
                - If blockcomponent is a dictionary, each block can have different components considered in the assembly.
                  The dictonary should be of the form {csi.Block.name: 'rotation+intradef+boundary'}
                  For example, if you want to consider only rotation for block A and rotation + intradef for block B, you can set
                  blockcomponent = {'A': 'rotation', 'B': 'rotation+intradef'}.
                  
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
            print("Assembling G for multiblock {}".format(self.name))
        
        # Store block components to consider
        
        if type(blockcomponent) is str:
            self.blockcomponent = {block.name: blockcomponent for block in self.blocks}
        else:
            self.blockcomponent = blockcomponent
            
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
                if (poly.__class__ is not str) and (poly is not None) and (poly.__class__ is not list):
                    self.poly[data.name] = poly*data.obs_per_station
                else:
                    self.poly[data.name] = poly

        # Create the transformation holder
        if not hasattr(self, 'helmert'):
            self.helmert = {}
        if not hasattr(self, 'strain'):
            self.strain = {}
        if not hasattr(self, 'transformation'):
            self.transformation = {}
        
        # Get the number of block parameters
        
        Nps = 0
        for block in self.blocks:
            if ('rotation' in self.G[datas[0].name].keys() and 'rotation' in self.blockcomponent[block.name]) or ('boundary' in self.G[datas[0].name].keys() and 'boundary' in self.blockcomponent[block.name]):
                Nps += 3
            if 'intradef' in self.G[datas[0].name].keys() and 'intradef' in self.blockcomponent[block.name]:
                Nps += 3
        
        # Get the number of transformation parameters
        Npo = 0
        for data in datas:
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
        
        # Custom ?
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
        
        # Loop over the datasets
        el = 0
        custstart = Nps # custom indices
        polstart = Nps + Npc # poly indices
        
        for data in datas:

            # Keep data name
            self.datanames.append(data.name)

            # print
            if verbose:
                print("Dealing with {} of type {}".format(data.name, data.dtype))

            # Rotation Green's functions

            # Get the corresponding G
            Ndlocal = self.d[data.name].shape[0]
                
            # Check if we have rotation GFs
            Npslocal = 0
            for iblock, block in enumerate(self.blocks):
                if 'rotation' in self.G[data.name].keys() and 'rotation' in self.blockcomponent[block.name]:
                    Glocal = self.G[data.name]['rotation'][:, 3*iblock:3*(iblock+1)]
                    G[el:el+Ndlocal, Npslocal:Npslocal+3] += Glocal
                    Npslocal += 3
            
            Npslocal = 0
            # Check if we have boundary coseismic slip deficit GFs
            for iblock, block in enumerate(self.blocks):
                if 'boundary' in self.G[data.name].keys() and 'boundary' in self.blockcomponent[block.name]:
                    Glocal = self.G[data.name]['boundary'][:, 3*iblock:3*(iblock+1)]
                    G[el:el+Ndlocal, Npslocal:Npslocal+3] += Glocal
                    Npslocal += 3
            
            # Check if we have internal block deformation GFs
            for iblock, block in enumerate(self.blocks):
                if 'intradef' in self.G[data.name].keys() and 'intradef' in self.blockcomponent[block.name]:
                    Glocal = self.G[data.name]['intradef'][:, 3*iblock:3*(iblock+1)]
                    G[el:el+Ndlocal, Npslocal:Npslocal+3] += Glocal
            
            # Custom
            if custom:
                # Check if data has custom GFs
                if 'custom' in self.G[data.name].keys():
                    nc = self.G[data.name]['custom'].shape[1] # Nb of custom param
                    custend = custstart + nc
                    G[el:el+Ndlocal, custstart:custend] = self.G[data.name]['custom']
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
    # Assemble the data vector
    def assembled(self, datas, verbose=True):
        '''
        Assembles a data vector for inversion using the list datas
        Assembled vector is stored in self.dassembled

        Args:
            * datas         : list of data objects

        Returns:
            * None
        '''
        
        # Check if the Green's function are ready
        assert self.Gassembled is not None, \
                "You should assemble the Green's function matrix first"

        # Check
        if type(datas) is not list:
            datas = [datas]

        if verbose:
            # print
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling d for multiblock {}".format(self.name))

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
                    print("Dealing with data {} of type {}".format(data.name, data.dtype))

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
    # Assemble the data covariance matrix
    def assembleCd(self, datas, add_prediction=None, verbose=True):
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

        # Talk to me
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Assembling Cd for multiblock {}".format(self.name))
        
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
            
            # talk to me
            if verbose:
                print("Dealing with data {} of type {}".format(data.name, data.dtype))
                
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
    # Build the model covariance matrix
    def buildCmGaussian(self, sigma_rot=None, sigma_intradef=None, verbose=True):
        '''
        Builds a diagonal Cm using user-defined values with sigma_rot and/or sigma_intradef
        values on the diagonal. Model covariance is stored in self.Cm.

        Kwargs:
            * sigma_rot     : standard deviation for the rotation parameters
            * sigma_intradef: standard deviation for the internal block deformation parameters
            * verbose       : Talk to me (overwrites self.verbose)

        Returns:
            * None
        '''
        
        # Check if the Green's function are ready
        assert self.Gassembled is not None, \
                "You should assemble the Green's function matrix first"

        # Talk to me
        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print (f"Assembling the Cm matrix for multiblock {self.name}")
            print (f"sigma rotation = {sigma_rot}")
            print (f"sigma intradef = {sigma_intradef}")
        
        # Check the number of parameters
        Nps = 0
        for iblock, block in enumerate(self.blocks):
            if ('rotation' in self.blockcomponent[block.name] and 'rotation' in self.G[self.datanames[0]].keys()) or ('boundary' in self.blockcomponent[block.name] and 'boundary' in self.G[self.datanames[0]].keys()):
                Nps += 3
            if 'intradef' in self.blockcomponent[block.name] and 'intradef' in self.G[self.datanames[0]].keys():
                Nps += 3
        
        # Creates the Cm matrix
        Cm = np.eye(Nps)
        n = 0
        
        # Fill with rotation
        for iblock, block in enumerate(self.blocks):
             if ('rotation' in self.blockcomponent[block.name] and 'rotation' in self.G[self.datanames[0]].keys()) or ('boundary' in self.blockcomponent[block.name] and 'boundary' in self.G[self.datanames[0]].keys()):
                if sigma_rot is not None:
                    Cm[n:n+3, n:n+3] *= sigma_rot
                n += 3

        # Fill with intradef
        for iblock, block in enumerate(self.blocks):
            if 'intradef' in self.blockcomponent[block.name] and 'intradef' in self.G[self.datanames[0]].keys():
                if sigma_intradef is not None:
                    Cm[n:n+3, n:n+3] *= sigma_intradef
                n += 3
        
        # Store Cm into self
        self.Cm = Cm

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Save Cm to file
    def writeCm2File(self, dtype='d', outputDir='.'):
        '''
        Write the model a priori covariance matrix to a binary file.

        Args:
            * filename      : Name of the file.
        
        Kwargs:
            * dtype         : Data type to use. Default is double ('d'). Can be 'f' for float32.
            * outputDir     : Directory to write the file in.

        Returns:
            * None
        '''

        # Check that Cm exists
        assert hasattr(self, 'Cm'), "No Cm matrix to write"

        # Write to file
        filename = f"{self.name.replace(' ', '_')}.cm"
        self.Cm.astype(dtype).tofile(os.path.join(outputDir, filename))
        print('Writing Cm matrix to file {}'.format(filename))

        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    # Read Cm from file
    def setCmFromFile(self, filename=None, dtype='d', inDir='.'):
        '''
        Read the model a priori covariance matrix from a binary file.

        Args:
            * filename      : Name of the file.
            
        Kwargs:
            * dtype         : Data type to use. Default is double ('d'). Can be 'f' for float32.
            * inDir         : Directory to read the file from.
        
        Returns:
            * None
        '''
        
        # Check name conventions
        if filename is None:
            if os.path.isfile(os.path.join(inDir, f'{self.name.replace(" ","_")}.cm')):
                filename = os.path.join(inDir, f'{self.name.replace(" ","_")}.cm')

        # Read the files and reshape the Cm
        Cm = np.fromfile(filename, dtype=dtype)
        n = int(np.sqrt(Cm.size))
        Cm = Cm.reshape((n, n))
        
        # Store Cm
        self.Cm = Cm
        print('Reading Cm matrix from file {}'.format(filename))

        # All done
        return
    # ----------------------------------------------------------------------
    