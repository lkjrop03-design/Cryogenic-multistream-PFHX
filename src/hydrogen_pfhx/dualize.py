"""Conversion of a fluid stream into a dual-number copy for the analytic
Jacobian: temperature, pressure, para fraction and every property used by
the right-hand side carry their derivatives with respect to the node's
state vector, so evaluating the model on the copy yields one Jacobian block.
Hydrogen property derivatives are analytic (hydrogen_property_derivatives);
other fluids use central differences of their CoolProp properties.
"""

import copy
import numpy as np
from hydrogen_pfhx.dual import Dual
from hydrogen_pfhx.hydrogen_property_derivatives import hydrogen_property_derivatives


# properties whose derivatives the right-hand side needs
_PROPS = ('mass_density', 'viscosity', 'specific_heat_capacity',
          'thermal_conductivity', 'enthalpy')


def _fluidstream_property_derivatives(stream):
    """Derivatives of the properties in _PROPS of a non-hydrogen stream with
    respect to its own T (per K) and P (per Pa), by central differences on a
    scratch copy (the stream itself is untouched). dxp is zero.
    """
    from hydrogen_pfhx.fluids import FluidStream
    T, P = float(stream.temperature), float(stream.pressure)
    # finite-difference steps in T (K) and P (Pa)
    hT = max(1e-3, 1e-5 * T)
    hP = max(1.0, 1e-5 * P)
    scratch = FluidStream(stream.fluid, stream.mass_flow_rate)

    def props(t, p):
        scratch.update_conditions(t, p)
        scratch.set_properties()
        return {k: float(getattr(scratch, k)) for k in _PROPS}

    tp, tm = props(T + hT, P), props(T - hT, P)
    pp, pm = props(T, P + hP), props(T, P - hP)
    return {k: {'dT': (tp[k] - tm[k]) / (2 * hT),
                'dP': (pp[k] - pm[k]) / (2 * hP),
                'dxp': 0.0} for k in _PROPS}


def dualize_stream(stream, n_states, idx_xp, idx_P, idx_T):
    """Shallow copy of `stream` whose state and properties are dual numbers over
    the n_states node state (rows idx_xp, idx_P, idx_T). Properties use the
    chain rule: d(prop)/d(state) = dprop/dT dT + dprop/dP dP + dprop/dxp dxp.
    """
    hc = copy.copy(stream)
    # analytic derivatives for hydrogen, finite differences otherwise
    if hasattr(stream, 'ortho'):
        derivs = hydrogen_property_derivatives(stream)
    else:
        derivs = _fluidstream_property_derivatives(stream)

    der_T = np.zeros(n_states); der_T[idx_T] = 1.0
    # seed vectors: the state's pressure is in kPa, properties are per Pa
    der_P = np.zeros(n_states); der_P[idx_P] = 1000.0
    der_xp = np.zeros(n_states)
    if idx_xp is not None:
        der_xp[idx_xp] = 1.0

    hc.temperature = Dual(stream.temperature, der_T)
    hc.pressure = Dual(stream.pressure, der_P)
    if idx_xp is not None:
        hc.xp = Dual(stream.xp, der_xp)

    def chain(prop):
        d = derivs[prop]
        return d['dT'] * der_T + d['dP'] * der_P + d['dxp'] * der_xp

    # dual-number properties; Prandtl number follows from them
    hc.mass_density = Dual(stream.mass_density, chain('mass_density'))
    hc.molar_density = Dual(stream.molar_density, chain('mass_density') / stream.molecular_mass)
    hc.viscosity = Dual(stream.viscosity, chain('viscosity'))
    hc.specific_heat_capacity = Dual(stream.specific_heat_capacity, chain('specific_heat_capacity'))
    hc.thermal_conductivity = Dual(stream.thermal_conductivity, chain('thermal_conductivity'))
    hc.enthalpy = Dual(stream.enthalpy, chain('enthalpy'))
    hc.prandtl_number = hc.viscosity * hc.specific_heat_capacity / hc.thermal_conductivity
    return hc
