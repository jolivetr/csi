'''
A class that deals with triangular patches for kinematic slip models: the surface coordinates
of the fault, the source points of each patch, the hypocenter, the kinematic Green's functions
and the big G and D matrices of AlTar's kinematic model on triangles.

The usual sequence:

    1. the mesh: setMesh with vertices and faces, or buildPlanarFault or buildCurvedFault, which
       also set the surface coordinates of the vertices, {uv}: along strike and down dip, in km,
       along the fault surface; setSurfaceCoordinates projects a mesh from elsewhere onto its
       mean plane instead
    2. the source points of each patch (setSourcePoints), the shear modulus of each patch
       (setMu), and, for a synthetic, the hypocenter (setHypoOnFault)
    3. the kinematic Green's functions (buildKinGFs): the seismograms at each station of each
       patch slipping by a unit slip along each rake, released by a short triangle at time 0,
       from a waveform engine such as csi.pyrockoengine
    4. the big G and big D (buildBigGD): the Green's functions delayed by each rupture time step,
       one column per time step, rake and patch, and the observations in the same order
    5. the static Green's functions (buildStaticGFs), for joint inversions: from the same engine,
       or from triangular dislocations in a homogeneous half space, with cuokada or cutde
    6. the files AlTar reads (saveKinematic)

AlTar computes the rupture times itself, from the hypocenter and the rupture velocities, on a
mesh refined from these patches; here, only the geometry and the Green's functions are built.
'''

# Externals
import numpy as np

# Personals
from .TriangularPatches import TriangularPatches


class TriangularPatchesKin(TriangularPatches):
    '''
    Triangular patches for kinematic inversions

    Args:
        * name      : name of the fault

    Kwargs:
        * utmzone   : UTM zone (optional)
        * ellps     : ellipsoid (default 'WGS84')
        * lon0      : longitude of the center of the UTM zone
        * lat0      : latitude of the center of the UTM zone
        * verbose   : speak to me (default True)
    '''

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Base class init
        super(TriangularPatchesKin, self).__init__(name, utmzone=utmzone, ellps=ellps,
                                                   lon0=lon0, lat0=lat0, verbose=verbose)

        # The surface coordinates of the vertices and the hypocenter
        self.uv = None
        self.hypo_uv = None
        # The source points
        self.sources_uv = None
        self.sources_xyz = None
        self.sources_patch = None
        # Shear modulus of each patch
        self.mu = None
        # Green's functions and big matrices
        self.G = {}
        self.bigG = None
        self.bigD = None

        # All done
        return

    # ----------------------------------------------------------------------
    def setMesh(self, vertices, faces):
        '''
        Set the patches from vertices (km, x, y and depth) and faces, and the geometry csi's
        triangular patches derive from them: the patches in km and in lon/lat, the depth range
        and the area of each patch

        Args:
            * vertices  : array (nvertices x 3), x, y in km, depth in km, positive down
            * faces     : array (npatches x 3) of vertex indices

        Returns:
            * None
        '''

        # The vertices and faces, and the vertices in lon/lat
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)
        self.Vertices = vertices
        self.Faces = faces
        self.vertices2ll()

        # The corners of each patch, in km and in lon/lat
        self.patch = []
        self.patchll = []
        for face in faces:
            p = vertices[face]
            self.patch.append(p.copy())
            pll = np.column_stack(self.xy2ll(p[:, 0], p[:, 1]) + (p[:, 2],))
            self.patchll.append(pll)
        self.numpatch = faces.shape[0]

        # The depth range, the depths csi plots the fault at, the slips and the areas
        self.top = vertices[:, 2].min()
        self.depth = vertices[:, 2].max()
        self.z_patches = np.linspace(self.depth, 0.0, 5)
        self.initializeslip()
        self.computeArea()

        # All done
        return

    # ----------------------------------------------------------------------
    def buildPlanarFault(self, lon, lat, depth, strike, dip, length, width, nstrike, ndip):
        '''
        A planar fault cut into nstrike x ndip rectangles, each into two triangles, the diagonals
        alternating; the surface coordinates of the vertices are their distances along strike and
        down dip from the start of the top edge, exact on a plane

        Args:
            * lon, lat  : the start of the top edge of the fault
            * depth     : the depth of the top edge (km)
            * strike    : strike angle (deg)
            * dip       : dip angle (deg)
            * length    : length along strike (km)
            * width     : width down dip (km)
            * nstrike   : number of rectangles along strike
            * ndip      : number of rectangles down dip

        Returns:
            * None
        '''

        # The unit vectors along strike and down dip (x east, y north, z down)
        x0, y0 = self.ll2xy(lon, lat)
        φ, δ = np.radians(strike), np.radians(dip)
        es = np.array([np.sin(φ), np.cos(φ), 0.0])
        ed = np.array([np.cos(φ) * np.cos(δ), -np.sin(φ) * np.cos(δ), np.sin(δ)])

        # The grid of vertices, in surface coordinates and in space
        u = np.linspace(0.0, length, nstrike + 1)
        v = np.linspace(0.0, width, ndip + 1)
        U, V = np.meshgrid(u, v, indexing='ij')
        uv = np.column_stack([U.ravel(), V.ravel()])
        vertices = np.array([x0, y0, depth]) + uv[:, :1] * es + uv[:, 1:] * ed

        # Two triangles a rectangle, the diagonals alternating, so the mesh has no preferred
        # direction
        index = lambda i, j: i * (ndip + 1) + j
        faces = []
        for i in range(nstrike):
            for j in range(ndip):
                a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
                if (i + j) % 2:
                    faces += [(a, b, c), (a, c, d)]
                else:
                    faces += [(a, b, d), (b, c, d)]

        # Set the mesh, and its surface coordinates
        self.setMesh(vertices, np.array(faces))
        self.uv = uv

        # All done
        return

    # ----------------------------------------------------------------------
    def buildCurvedFault(self, lon, lat, depth, strike, dip, length, width, nstrike, ndip,
                         steps=2000):
        '''
        A fault whose dip changes with depth, along a constant strike, e.g. a subducting slab,
        cut into nstrike x ndip quadrilaterals, each into two triangles, the diagonals
        alternating. The down-dip profile is integrated from the dip function; every
        cross-section along strike is the same, so the surface is developable and its surface
        coordinates (along strike, and down dip along the surface) are exact

        Args:
            * lon, lat  : the start of the top edge of the fault
            * depth     : the depth of the top edge (km)
            * strike    : strike angle (deg)
            * dip       : the dip (deg) as a function of depth (km)
            * length    : length along strike (km)
            * width     : width down dip, along the surface (km)
            * nstrike   : number of cells along strike
            * ndip      : number of cells down dip

        Kwargs:
            * steps     : the integration steps of the profile

        Returns:
            * None
        '''

        # The unit vectors along strike, and horizontal, perpendicular to it, toward the dip
        x0, y0 = self.ll2xy(lon, lat)
        φ = np.radians(strike)
        es = np.array([np.sin(φ), np.cos(φ), 0.0])
        eh = np.array([np.cos(φ), -np.sin(φ), 0.0])

        # The profile down dip: horizontal distance and depth against the distance along the
        # surface, by forward integration of the dip
        w = np.linspace(0.0, width, steps + 1)
        h, z = np.zeros_like(w), np.full_like(w, depth, dtype=float)
        for k in range(steps):
            δ = np.radians(dip(z[k]))
            dw = w[k + 1] - w[k]
            h[k + 1] = h[k] + dw * np.cos(δ)
            z[k + 1] = z[k] + dw * np.sin(δ)

        # The grid of vertices in surface coordinates, then in space through the profile
        u = np.linspace(0.0, length, nstrike + 1)
        v = np.linspace(0.0, width, ndip + 1)
        U, V = np.meshgrid(u, v, indexing='ij')
        uv = np.column_stack([U.ravel(), V.ravel()])
        H, Z = np.interp(uv[:, 1], w, h), np.interp(uv[:, 1], w, z)
        vertices = np.array([x0, y0, 0.0]) + uv[:, :1] * es + H[:, None] * eh
        vertices[:, 2] = Z

        # Two triangles a cell, the diagonals alternating
        index = lambda i, j: i * (ndip + 1) + j
        faces = []
        for i in range(nstrike):
            for j in range(ndip):
                a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
                if (i + j) % 2:
                    faces += [(a, b, c), (a, c, d)]
                else:
                    faces += [(a, b, d), (b, c, d)]

        # Set the mesh, and its surface coordinates
        self.setMesh(vertices, np.array(faces))
        self.uv = uv

        # All done
        return

    # ----------------------------------------------------------------------
    def convergenceRakes(self, azimuth):
        '''
        The rake of each patch for slip along the plate convergence: the hanging wall moving
        against the azimuth of the subducting plate, projected on the patch. The rake is measured
        in the plane of the patch from its strike, positive toward up dip (90 is pure thrust)

        Args:
            * azimuth   : the direction (deg, clockwise from north) the subducting plate moves in,
                          relative to the overriding one

        Returns:
            * rakes     : (deg) one per patch; slip perpendicular to the convergence is 90 more
        '''

        # The horizontal direction the hanging wall moves in
        a = np.radians(azimuth)
        h = -np.array([np.sin(a), np.cos(a), 0.0])

        # Its components along strike and up dip of each patch, as a rake
        rakes = []
        for p in range(len(self.patch)):
            _, _, _, _, _, strike, dip = self.getpatchgeometry(p, center=True)
            es = np.array([np.sin(strike), np.cos(strike), 0.0])
            ed = np.array([np.cos(strike) * np.cos(dip), -np.sin(strike) * np.cos(dip), np.sin(dip)])
            rakes.append(np.degrees(np.arctan2(h @ -ed, h @ es)))

        # All done
        return np.array(rakes)

    # ----------------------------------------------------------------------
    def setSurfaceCoordinates(self):
        '''
        The surface coordinates of the vertices, along strike and down dip, by projection onto
        the mean plane of the fault, from its shallowest along-strike start; exact for a planar
        fault, an approximation for a curved one, for a mesh made elsewhere (e.g. read from a
        file with setMesh)

        Returns:
            * None
        '''

        # The mean normal of the patches, all pointing up
        V = self.Vertices
        normal = np.zeros(3)
        for face in self.Faces:
            p = V[face]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            normal += n if n[2] < 0 else -n
        normal /= np.linalg.norm(normal)

        # Strike: horizontal, normal to the normal; dip: down the plane
        es = np.array([-normal[1], normal[0], 0.0])
        es /= np.linalg.norm(es)
        ed = np.cross(normal, es)
        if ed[2] < 0:
            es, ed = -es, -ed

        # The projections, from the smallest of each
        u, v = V @ es, V @ ed
        self.uv = np.column_stack([u - u.min(), v - v.min()])

        # All done
        return

    # ----------------------------------------------------------------------
    def setHypoOnFault(self, h_strike, h_dip):
        '''
        Set the hypocenter in surface coordinates, as AlTar's kinematic model on triangles takes
        it

        Args:
            * h_strike  : along strike (km)
            * h_dip     : down dip (km)

        Returns:
            * None
        '''

        self.hypo_uv = np.array([h_strike, h_dip], dtype=float)

        # All done
        return

    # ----------------------------------------------------------------------
    def setSourcePoints(self, n):
        '''
        The source points of each patch: the centroids of its n x n sub-triangles, n^2 points
        that tile the patch evenly, in surface and in cartesian coordinates. The Green's functions
        of a patch can sum their point sources (buildKinGFs with points), and AlTar interpolates
        the rupture time at each of them, so a patch radiates over the time the front takes to
        cross it

        Args:
            * n         : the number of divisions of each edge

        Returns:
            * None
        '''

        assert self.uv is not None, 'set the surface coordinates first'

        # The barycentric coordinates of the centroids of the sub-triangles: those pointing up,
        # then, between them, those pointing down
        bary = []
        for i in range(n):
            for j in range(n - i):
                bary.append(((i + 1 / 3) / n, (j + 1 / 3) / n))
                if i + j < n - 1:
                    bary.append(((i + 2 / 3) / n, (j + 2 / 3) / n))
        bary = np.array(bary)
        w = np.column_stack([1 - bary.sum(1), bary])
        self.sources_per_patch = w.shape[0]

        # The points of each patch, patch by patch, in surface and cartesian coordinates
        uv, xyz, patch = [], [], []
        for p, face in enumerate(self.Faces):
            uv.append(w @ self.uv[face])
            xyz.append(w @ self.Vertices[face])
            patch.append(np.full(w.shape[0], p))
        self.sources_uv = np.vstack(uv)
        self.sources_xyz = np.vstack(xyz)
        self.sources_patch = np.concatenate(patch)

        # All done
        return

    # ----------------------------------------------------------------------
    def setMu(self, mu):
        '''
        The shear modulus of each patch (Pa), for the moments of its point sources: a scalar, an
        array, or a function of depth (km), e.g. pyrockoengine.material, so the moments match the
        earth model of the Green's functions

        Args:
            * mu        : scalar, array of length npatches, or callable

        Returns:
            * None
        '''

        if callable(mu):
            # at the depth of the center of each patch
            self.mu = np.array([mu(self.getcenter(p)[2]) for p in self.patch])
        else:
            self.mu = np.ones(len(self.patch)) * np.asarray(mu, dtype=float)

        # All done
        return

    # ----------------------------------------------------------------------
    def buildKinGFs(self, data, engine, Dt, npts, rakes=(0.0, 90.0), slip=1.0, tstart=-40.0,
                    points=False, bandpass=None):
        '''
        The seismograms of each patch for a unit slip along each rake, released by a triangle
        of half-width Dt centered at time 0, from a point source at its centroid, or, with
        {points}, from its source points, sharing its moment

        The seismograms run from {tstart} to (npts - 1) delta, delta the engine's sampling: the
        part before time 0 holds what the engine radiates before the source (some stores have
        precursors), and lets buildBigGD delay them by the rupture time steps without losing
        them. They are kept in {self.G[data.name]}, one array per rake, (patches x stations x
        components x samples).

        Args:
            * data      : a seismic object with the stations (sta_name, lon, lat)
            * engine    : a waveform engine, e.g. csi.pyrockoengine
            * Dt        : the rupture time step (s)
            * npts      : the number of samples of each seismogram from time 0, at the engine's
                          rate

        Kwargs:
            * rakes     : the rake angles (deg) of the slip components, relative to the strike of
                          each patch: a scalar, or an array with one per patch (e.g. from
                          convergenceRakes); the Green's functions are kept by component index
            * slip      : slip amplitude (m)
            * tstart    : the start of the seismograms (s), early enough to hold the whole
                          traces of the engine, which may start before the source
            * points    : the source points (setSourcePoints) instead of the centroids
            * bandpass  : (fmin, fmax[, order]) in Hz, a zero-phase Butterworth bandpass applied to
                          the seismograms, as to the data; none by default

        Returns:
            * None
        '''

        assert self.mu is not None, 'set the shear modulus first (setMu)'
        stations = {'name': data.sta_name, 'lon': data.lon, 'lat': data.lat}
        Np = len(self.patch)

        # The point sources of each patch: its source points, or its centroid
        if points:
            assert self.sources_xyz is not None, 'set the source points first (setSourcePoints)'
            locations = [self.sources_xyz[self.sources_patch == p] for p in range(Np)]
        else:
            locations = [np.array([self.getcenter(q)]) for q in self.patch]

        # A double couple at each point, rake by rake, patch by patch, with the patch's strike and
        # dip, and its share of the moment mu * slip * area
        sources = []
        for rake in rakes:
            for p in range(Np):
                r = float(rake) if np.ndim(rake) == 0 else float(rake[p])
                _, _, _, _, _, strike, dip = self.getpatchgeometry(p, center=True)
                share = self.mu[p] * slip * self.area[p] * 1.0e6 / len(locations[p])
                lon, lat = self.xy2ll(locations[p][:, 0], locations[p][:, 1])
                for k in range(len(locations[p])):
                    sources.append({'lon': lon[k], 'lat': lat[k], 'depth': locations[p][k, 2],
                                    'strike': np.degrees(strike), 'dip': np.degrees(dip),
                                    'rake': r, 'moment': share, 'time': 0.0, 'duration': 2.0 * Dt})

        # Their seismograms, on a time axis from the sample nearest tstart
        pre = int(round(-tstart / engine.delta))
        synth = engine.synthetics(sources, stations, npts + pre, tstart=-pre * engine.delta)

        # The seismograms of the points of each patch, summed: the sources are laid out rake by
        # rake, patch by patch, so the boundaries repeat the point counts once per rake
        counts = [len(l) for l in locations]
        edges = np.concatenate([[0], np.cumsum(counts * len(rakes))])
        summed = np.array([synth[edges[i]:edges[i + 1]].sum(0) for i in range(len(edges) - 1)])

        # The bandpass of the data, forward and backward, so it doesn't shift them
        if bandpass is not None:
            import scipy.signal
            fmin, fmax = bandpass[:2]
            order = bandpass[2] if len(bandpass) > 2 else 2
            sos = scipy.signal.butter(order, [fmin, fmax], btype='bandpass', fs=1.0 / engine.delta,
                                      output='sos')
            summed = scipy.signal.sosfiltfilt(sos, summed, axis=-1)
        self.kin_bandpass = bandpass

        # Keep them by rake, and the time axis they're on
        self.G[data.name] = {i: summed[i * Np:(i + 1) * Np] for i in range(len(rakes))}
        self.kin_delta = engine.delta
        self.kin_npts = npts
        self.kin_pre = pre
        self.kin_components = engine.components

        # All done
        return

    # ----------------------------------------------------------------------
    def buildStaticGFs(self, data, engine=None, rakes=(0.0, 90.0), slip=1.0, duration=300.0,
                       points=True, method='engine', nu=0.25, device=None):
        '''
        The static displacements of each patch for a unit slip along each rake, at the stations of
        {data}, by one of:

            * 'engine': the final values of its seismograms, from the same waveform engine as the
              kinematic Green's functions, so the two share their earth model; the kinematic
              Green's functions of {data}, if any, are kept
            * 'cuokada' or 'cutde': triangular dislocations in a homogeneous elastic half space
              (Nikkhoo and Walter, 2015), with the package of that name, exact for each triangle

        CSI's own methods, e.g. buildGFs(data, method='meade') or 'edks', remain available for
        gps and insar objects

        Args:
            * data      : an object with the stations (sta_name, lon, lat), e.g. a seismic object

        Kwargs:
            * engine    : a waveform engine, e.g. csi.pyrockoengine, for method 'engine'
            * rakes     : rake angles (deg), relative to the strike of each patch: a scalar, or an
                          array with one per patch (e.g. from convergenceRakes)
            * slip      : slip amplitude (m)
            * duration  : 'engine': how long to let the waves pass (s), longer than the farthest
                          arrivals
            * points    : 'engine': the source points (setSourcePoints) instead of the centroids
            * method    : 'engine' (default), 'cuokada' or 'cutde'
            * nu        : 'cuokada' and 'cutde': Poisson's ratio of the half space
            * device    : 'cuokada': 'cuda' to compute on the gpu; default the cpu

        Returns:
            * G         : (stations x components) x (rakes x patches), the stations slowest, the
                          components east, north and up
        '''

        if method in ('cuokada', 'cutde'):
            G = self._halfspaceStaticGFs(data, method, rakes=rakes, slip=slip, nu=nu, device=device)
            self.staticG = G
            return G
        assert method == 'engine', "method is 'engine', 'cuokada' or 'cutde', not {}".format(method)
        assert engine is not None, "method 'engine' needs a waveform engine"

        # Keep the kinematic Green's functions and their sampling, which buildKinGFs replaces
        saved = self.G.get(data.name)
        sampling = {k: getattr(self, k) for k in ['kin_delta', 'kin_npts', 'kin_pre', 'kin_components',
                                                  'kin_bandpass'] if hasattr(self, k)}

        # Seismograms long enough for the waves to pass, unfiltered, and their last samples
        npts = int(round(duration / engine.delta))
        self.buildKinGFs(data, engine, Dt=engine.delta, npts=npts, rakes=rakes, slip=slip,
                         points=points)
        G = np.concatenate([self.G[data.name][i][..., -1].reshape(len(self.patch), -1).T
                            for i in range(len(rakes))], axis=1)

        # Put the kinematic Green's functions and their sampling back
        if saved is None:
            del self.G[data.name]
        else:
            self.G[data.name] = saved
        for k, v in sampling.items():
            setattr(self, k, v)
        self.staticG = G

        # All done
        return G

    # ----------------------------------------------------------------------
    def _halfspaceStaticGFs(self, data, method, rakes=(0.0, 90.0), slip=1.0, nu=0.25, device=None):
        '''
        The static displacements of each patch for a unit slip along each rake in a homogeneous
        half space, from the triangular dislocations of cuokada or cutde

        Both take the slip of a triangle in its own frame, which the order of its vertices sets:
        the normal n by the right-hand rule, the strike s = z x n and the dip d = n x s, the slip
        being the motion of the block on the side of n. Here a rake is first the motion of the
        hanging wall relative to the footwall, as for the waveform engine, cos(rake) along the
        strike of the patch and sin(rake) up its dip, then, in the frame of the triangle, that
        motion or its opposite, as n points into the hanging wall or the footwall; so the order of
        the vertices of a mesh doesn't matter, except for vertical patches, whose strike csi takes
        from it (homogeneizeStrike orients them), for this method as for the waveform engine

        Args:
            * data      : an object with the stations (lon, lat)
            * method    : 'cuokada' or 'cutde'

        Kwargs:
            * rakes     : rake angles (deg), a scalar or one per patch each
            * slip      : slip amplitude (m)
            * nu        : Poisson's ratio
            * device    : cuokada: 'cuda' for the gpu

        Returns:
            * G         : (stations x 3) x (rakes x patches), the stations slowest
        '''

        # The stations at the surface and the triangles, x east, y north, z up, in km
        x, y = self.ll2xy(np.asarray(data.lon, dtype=float), np.asarray(data.lat, dtype=float))
        obs = np.column_stack([x, y, np.zeros(len(x))])
        tris = np.array([np.asarray(p, dtype=float) for p in self.patch])
        tris[:, :, 2] *= -1.0
        Np = len(tris)

        # The displacements of unit slips along the strike, dip and normal of each triangle frame,
        # (stations, 3 components, patches, 3 slips)
        if method == 'cuokada':
            import cuokada
            M = cuokada.tri_disp_matrix(obs, tris, nu, device=device)
        else:
            import cutde.halfspace
            M = cutde.halfspace.disp_matrix(obs, tris, nu)
        M = np.asarray(M, dtype=float)

        # The frame of each triangle, as cuokada and cutde set it
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1)[:, None]
        s = np.cross([0.0, 0.0, 1.0], n)
        flat = np.linalg.norm(s, axis=1) < 1e-12
        s[flat] = np.column_stack([np.zeros(flat.sum()), np.sign(n[flat, 2]), np.zeros(flat.sum())])
        s /= np.linalg.norm(s, axis=1)[:, None]
        d = np.cross(n, s)

        # The strike, the up-dip direction and the normal into the hanging wall of each patch, as
        # csi's geometry has them, z up
        es, eu, nh = np.zeros((Np, 3)), np.zeros((Np, 3)), np.zeros((Np, 3))
        for p in range(Np):
            _, _, _, _, _, strike, dip = self.getpatchgeometry(p, center=True)
            es[p] = [np.sin(strike), np.cos(strike), 0.0]
            down = np.array([np.cos(strike) * np.cos(dip), -np.sin(strike) * np.cos(dip), -np.sin(dip)])
            eu[p] = -down
            nh[p] = np.cross(down, es[p])
        side = np.where((n * nh).sum(1) >= 0, 1.0, -1.0)

        # Each rake: the motion of the hanging wall, in the frame of each triangle, and the
        # displacements it causes
        G = []
        for rake in rakes:
            r = np.radians(np.full(Np, float(rake)) if np.ndim(rake) == 0 else np.asarray(rake, dtype=float))
            b = slip * side[:, None] * (np.cos(r)[:, None] * es + np.sin(r)[:, None] * eu)
            local = np.column_stack([(b * s).sum(1), (b * d).sum(1), (b * n).sum(1)])
            G.append(np.einsum('ocpk,pk->ocp', M, local).reshape(len(obs) * 3, Np))

        # All done
        return np.concatenate(G, axis=1)

    # ----------------------------------------------------------------------
    def buildBigGD(self, data, Nt, Dt, rakes=(0.0, 90.0), observed=None):
        '''
        The big G of AlTar's kinematic model, one column per time step, rake and patch, in that
        order, the patch fastest: the seismograms of the patch, delayed by the time step; and the
        big D, the observations in the same order, station by station, component by component

        AlTar's slip history Mb holds, for each time step k, rake and patch, the slip released by
        a triangle centered at k Dt, so the data are big G times Mb. The rows of big G are the
        samples of the seismograms from time 0, station by station, component by component, as
        buildKinGFs made them; {Dt} must be a multiple of their sampling interval

        Args:
            * data      : the seismic object of buildKinGFs
            * Nt        : the number of rupture time steps
            * Dt        : the rupture time step (s)

        Kwargs:
            * rakes     : unused, the components are those of buildKinGFs
            * observed  : the observations (stations x components x npts), if any

        Returns:
            * None
        '''

        G = self.G[data.name]
        Np = len(self.patch)
        delta, npts, pre = self.kin_delta, self.kin_npts, self.kin_pre
        nsta, ncomp = G[0].shape[1:3]
        bigG = np.zeros((Nt, len(G), Np, nsta, ncomp, npts))
        for k in range(Nt):
            # The seismogram sample that lands on the first data sample, at time 0, for a source
            # released at k Dt
            first = pre - int(round(k * Dt / delta))
            # data samples before the source's first seismogram sample stay zero
            lo = max(-first, 0)
            if lo >= npts:
                break
            for ir in range(len(G)):
                bigG[k, ir, :, :, :, lo:] = G[ir][:, :, :, first + lo:first + npts]

        # One row per data sample, one column per (time step, rake, patch)
        self.bigG = bigG.reshape(Nt * len(G) * Np, -1).T
        if observed is not None:
            self.bigD = np.asarray(observed, dtype=float).ravel()

        # All done
        return

    # ----------------------------------------------------------------------
    def saveKinematic(self, filename, prefix='kinematicG', dtype=np.float64):
        '''
        Save what AlTar's kinematic model on triangles needs: the Green's functions, in
        {prefix}.gf.h5 as (Nt x rakes x patches) x observations, the observations in
        {prefix}.data.h5, if any, and the mesh in {filename}: the vertices (km), their surface
        coordinates, the faces, the source points in surface coordinates and their patches, the
        hypocenter

        Args:
            * filename  : the .h5 file for the mesh

        Kwargs:
            * prefix    : the prefix of the Green's functions and data files
            * dtype     : the type of the saved matrices

        Returns:
            * None
        '''

        import h5py

        # The mesh, its surface coordinates, the source points and the hypocenter
        with h5py.File(filename, 'w') as h5:
            h5['vertices'] = self.Vertices
            h5['uv'] = self.uv
            h5['faces'] = self.Faces
            h5['area'] = np.asarray(self.area)
            if self.sources_uv is not None:
                h5['sources/uv'] = self.sources_uv
                h5['sources/patch'] = self.sources_patch
            if self.hypo_uv is not None:
                h5['hypocenter'] = self.hypo_uv

        # The big G, transposed to AlTar's layout, one row per column of Mb
        if self.bigG is not None:
            with h5py.File('{}.gf.h5'.format(prefix), 'w') as h5:
                h5['{}.gf'.format(prefix)] = self.bigG.T.astype(dtype)

        # The observations
        if self.bigD is not None:
            with h5py.File('{}.data.h5'.format(prefix), 'w') as h5:
                h5['{}.data'.format(prefix)] = self.bigD.astype(dtype)

        # All done
        return

#EOF
