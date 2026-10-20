'''
A waveform engine on pyrocko's Green's function stores: displacement seismograms of point
double-couple sources, on a fixed time axis, for kinematic Green's functions.

A pyrocko store (see https://pyrocko.org, fomosto) tabulates the seismograms of point sources in
a layered (1D) earth model on a grid of source depths and source-receiver distances, and pyrocko
interpolates between the grid nodes. This engine builds the sources and receivers pyrocko needs,
runs the store, and puts every seismogram on one common time axis, decimated if asked to.

Pyrocko is only imported when an engine is made, so csi does not depend on it.
'''

# Externals
import numpy as np


class pyrockoengine(object):
    '''
    Point-source seismograms from a pyrocko Green's function store (e.g., one downloaded with
    ``fomosto download kinherd <store_id>``)

    The store fixes the earth model, the depth and distance ranges of the sources and receivers,
    and the sampling interval; every source and receiver handed to {synthetics} must fall within
    those ranges.

    Args:
        * store_id      : the id of the store
        * store_dir     : the directory holding the store

    Kwargs:
        * components    : the components of each station, among 'E', 'N', 'Z' (default 'ENZ')
        * quantity      : 'displacement' (default), 'velocity' or 'acceleration'
        * interpolation : how pyrocko interpolates between the nodes of the store, 'multilinear'
                          (default) or 'nearest_neighbor'
        * delta         : the sampling interval of the seismograms (s), a multiple of the
                          store's; the traces are low-pass filtered and decimated to it; default
                          the store's
    '''

    def __init__(self, store_id, store_dir, components='ENZ', quantity='displacement',
                 interpolation='multilinear', delta=None):

        # Import pyrocko here, so that csi doesn't depend on it
        from pyrocko import gf

        # Save things
        self.name = 'pyrocko'
        self.store_id = store_id
        self.components = components
        self.quantity = quantity
        self.interpolation = interpolation

        # The engine and its store
        self.engine = gf.LocalEngine(store_superdirs=[store_dir])
        self.store = self.engine.get_store(store_id)

        # The sampling interval: the store's, or a multiple of it, so decimation keeps samples on
        # the store's own grid
        self.store_delta = self.store.config.deltat
        self.delta = self.store_delta if delta is None else float(delta)
        ratio = self.delta / self.store_delta
        assert abs(ratio - round(ratio)) < 1e-9 and ratio >= 1, \
            'delta must be a multiple of the store sampling interval {}'.format(self.store_delta)

        # All done
        return

    def material(self, depth):
        '''
        The shear modulus of the store's earth model at a depth, mu = rho vs^2, e.g. to give the
        patches of a fault the moduli their Green's functions were computed with

        Args:
            * depth : depth in km

        Returns:
            * mu    : shear modulus in Pa
        '''

        # The layered model of the store, in SI units
        mod = self.store.config.earthmodel_1d
        m = mod.material(depth * 1.0e3)

        # All done
        return m.rho * m.vs ** 2

    def synthetics(self, sources, stations, npts, tstart=0.0):
        '''
        The seismograms of point double couples, one set per source, on a common time axis of
        {npts} samples of {delta} from {tstart}

        Each seismogram is placed on the time axis by the time of its first sample; samples before
        the axis are dropped (and reported, see below), and the axis past the end of the trace is
        filled with its last value, the static offset, so seismograms shorter than the axis keep
        their permanent displacement. Some stores have small precursors before the arrivals of a
        source, so {tstart} should be early enough to hold them: the largest dropped value, as a
        fraction of each station's peak, is kept in {self.lost} and reported above 1e-3.

        Args:
            * sources   : list of dicts with 'lon', 'lat', 'depth' (km), 'strike', 'dip', 'rake'
                          (deg), 'moment' (N.m), 'time' (s) and 'duration' (s), the length of a
                          triangular source time function centered on 'time', or starting at it
                          with 'anchor' -1
            * stations  : dict with 'name', 'lon', 'lat' lists
            * npts      : the number of samples of each seismogram

        Kwargs:
            * tstart    : the time of the first sample (s)

        Returns:
            * synth     : array (sources x stations x components x npts), components in the
                          order of {components}
        '''

        from pyrocko import gf
        from pyrocko import moment_tensor as pmt

        # Targets, one per component of each station
        targets = []
        for name, lon, lat in zip(stations['name'], stations['lon'], stations['lat']):
            for c in self.components:
                targets.append(gf.Target(quantity=self.quantity, lat=float(lat), lon=float(lon),
                                         store_id=self.store_id, codes=('', name, '', c),
                                         interpolation=self.interpolation))

        # Sources: point double couples, their moment as a magnitude, with a triangular source
        # time function
        srcs = []
        for s in sources:
            srcs.append(gf.DCSource(lat=float(s['lat']), lon=float(s['lon']),
                                    depth=float(s['depth']) * 1.0e3, strike=float(s['strike']),
                                    dip=float(s['dip']), rake=float(s['rake']),
                                    magnitude=float(pmt.moment_to_magnitude(s['moment'])),
                                    time=float(s['time']),
                                    stf=gf.TriangularSTF(duration=float(s['duration']),
                                                         anchor=float(s.get('anchor', 0.0)))))

        # Compute every source at every target
        response = self.engine.process(srcs, targets)

        # Put each trace on the time axis: zeros before it, its last value after it
        nsta = len(stations['name'])
        ncomp = len(self.components)
        synth = np.zeros((len(srcs), nsta, ncomp, npts))
        # the largest value of each trace that falls before the time axis
        cut = np.zeros((len(srcs), nsta, ncomp))
        for isrc, src in enumerate(srcs):
            for itgt, tr in enumerate(response.results_list[isrc]):
                tr = tr.trace.pyrocko_trace()
                # Decimate to {delta}, if coarser than the store's sampling
                if self.delta > tr.deltat * (1 + 1e-9):
                    # pad with the end values, so the filter doesn't ring at the ends, from a
                    # multiple of delta, so the decimated samples fall on multiples of it without
                    # snapping them, which would shift the trace; and keep the mean, the static
                    # offset
                    start = np.floor(tr.tmin / self.delta - 20) * self.delta
                    tr.extend(tmin=start, tmax=tr.tmax + 40 * self.delta, fillmethod='repeat')
                    tr.downsample_to(self.delta, snap=False, demean=False)
                    assert abs(tr.tmin / self.delta - round(tr.tmin / self.delta)) < 1e-6, \
                        'the decimated trace is off the sampling grid'
                assert abs(tr.deltat - self.delta) < 1e-9 * self.delta
                y = tr.get_ydata()
                # Copy the part of the trace that overlaps the time axis
                i0 = int(round((tr.tmin - tstart) / self.delta))
                out = synth[isrc, itgt // ncomp, itgt % ncomp]
                lo, hi = max(i0, 0), min(i0 + y.size, npts)
                if hi > lo:
                    out[lo:hi] = y[lo - i0:hi - i0]
                # Past the end of the trace, its last value: the static offset
                if i0 + y.size < npts and y.size:
                    out[max(i0 + y.size, 0):] = y[-1]
                # Before the start of the axis: what was dropped
                if i0 < 0 and y.size:
                    cut[isrc, itgt // ncomp, itgt % ncomp] = np.abs(y[:min(-i0, y.size)]).max()

        # What fell before the time axis, against the largest component of each station
        peak = np.abs(synth).max(axis=(2, 3))
        lost = (cut.max(axis=2) / np.where(peak > 0, peak, np.inf)).max()
        if lost > 1e-3:
            print('Warning: up to {:.1e} of a station\'s peak fell before the start of the time '
                  'axis, at {} s'.format(lost, tstart))
        self.lost = lost

        # All done
        return synth

# EOF
