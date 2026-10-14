"""Fluid-property cache (cache variants). scipy's finite-difference Jacobian
perturbs one state at a time, so most streams' states repeat exactly between
calls; their property evaluation (density solves and correlations) is done
once and replayed. bvp_model.update_parameters calls set_properties through
the module-level cache, which model.model clears at the start of every run.
Keys use exact float equality of the state, which holds because the solver
passes the same floats unchanged.
"""


# properties set by set_properties that depend only on fluid and state
# (molar_flow_rate depends on the flow and is recomputed on a hit)
_CACHED_ATTRS = [
    'ortho_density', 'para_density', 'normal_density',
    'molecular_mass', 'molar_density', 'mass_density',
    'viscosity', 'specific_heat_capacity', 'thermal_conductivity',
    'prandtl_number', 'enthalpy', 'entropy',
    'critical_temperature', 'critical_pressure',
]


class PropertyCache:
    """Cache of property bundles keyed on (fluid class, fluid, T, P, xp)."""
    def __init__(self, maxsize=200_000):
        self._cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def clear(self):
        """Empty the cache and reset the hit/miss counters."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def set_properties(self, stream):
        """Drop-in replacement for stream.set_properties(): restore a cached
        property bundle for this exact state, or compute and store it. Must be
        called after update_conditions (and update_composition for hydrogen).
        """
        # keyed on the fluid and its state, never on the object, so runs cannot share entries
        key = (type(stream).__name__, stream.fluid, stream.temperature,
               stream.pressure, getattr(stream, 'xp', None))
        cached = self._cache.get(key)
        if cached is not None:
            for attr, val in cached.items():
                setattr(stream, attr, val)
            stream.molar_flow_rate = stream.mass_flow_rate / stream.molecular_mass
            self.hits += 1
            return

        stream.set_properties()
        # store until the size limit; beyond it, compute without caching
        if len(self._cache) < self.maxsize:
            self._cache[key] = {attr: getattr(stream, attr) for attr in _CACHED_ATTRS
                                if hasattr(stream, attr)}
        self.misses += 1
