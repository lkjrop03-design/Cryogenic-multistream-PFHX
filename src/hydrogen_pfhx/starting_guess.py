"""Physics-based starting guess for the BVP solver (called from
bvp_model.initialise_solution; the heuristic guess there is the fallback).

1. End temperatures from energy balances with real enthalpies: streams
   leaving at one end settle near the energy-balanced mix of the streams
   entering there, and the other end closes the overall balance. Which end
   this applies to depends on the heat capacities, so both are tried.
2. A linear background temperature between the two ends.
3. A relaxation length per stream from the model's own right-hand side.
4. Profiles: exponential approach from each inlet onto the background;
   para fraction linear to halfway to equilibrium; pressure flat.
Everything is in physical units; normalised variants scale the result.
"""


import numpy as np
from scipy.optimize import brentq
from hydrogen_pfhx import ortho_para_dynamics

# bounds on the relaxation length, as fractions of the exchanger length
MIN_RELAXATION_FRACTION = 1.0 / 200
MAX_RELAXATION_FRACTION = 10.0


def _xp_guess(xp_in, T):
    """Guessed para fraction at temperature T: halfway from the inlet value to equilibrium."""
    xp_eq, _ = ortho_para_dynamics.para_ortho_equilibrium(T)
    return xp_in + 0.5 * (xp_eq - xp_in)


def _inlets(topology, boundary_properties):
    """Inlet (T in K, P in Pa, para fraction or None) of every stream."""
    out = {}
    for label in topology.labels:
        e = topology.index[label]
        out[label] = (boundary_properties[e['T']],
                      boundary_properties[e['P']] * 1e3,
                      boundary_properties[e['xp']] if e['xp'] is not None else None)
    return out


def _enthalpy(stream, T, P_pa, xp):
    """Specific enthalpy (J/kg) of a stream at a trial state, kept above its temperature floor."""
    stream.update_conditions(max(T, stream.temperature_floor(P_pa)), P_pa)
    if xp is not None:
        stream.update_composition(xp)
    stream.set_properties()
    return stream.enthalpy


def _balance_temperature(streams, inlets, labels, fixed=None):
    """Temperature T at which the streams in `labels`, all brought to T, are
    energy-balanced together with the streams in `fixed`, each brought to its
    own given temperature. Includes the conversion heat implied by _xp_guess.
    Solved with brentq between the lowest and highest temperature involved
    (raises ValueError if no balance exists in that range).
    """
    fixed = fixed or {}
    h_in = {l: _enthalpy(streams[l], *inlets[l]) for l in list(labels) + list(fixed)}

    # enthalpy-flow change (W) of one stream brought from its inlet to T
    def contribution(label, T):
        T_in, P, xp_in = inlets[label]
        xp = _xp_guess(xp_in, T) if xp_in is not None else None
        return streams[label].mass_flow_rate * (_enthalpy(streams[label], T, P, xp) - h_in[label])

    # fixed streams contribute a constant; solve imbalance(T) = 0 for the others
    const = sum(contribution(l, T_f) for l, T_f in fixed.items())
    imbalance = lambda T: const + sum(contribution(l, T) for l in labels)
    temps = [inlets[l][0] for l in list(labels) + list(fixed)] + list(fixed.values())
    return brentq(imbalance, min(temps), max(temps), xtol=1e-3)


def end_temperatures(streams, topology, boundary_properties):
    """Background temperatures (T_A at z=0, T_B at z=L) for the guess.
    First assumes the streams entering at z=L dominate (co-current streams
    leave at their mix); if no energy balance exists that way, assumes the
    streams entering at z=0 dominate. One common temperature if all streams
    enter at the same end.
    """
    inlets = _inlets(topology, boundary_properties)
    # streams entering at z=0 (co-current) and at z=L (counter-current)
    enter_A = [l for l in topology.labels if topology.boundary_side(l) == 'ya']
    enter_B = [l for l in topology.labels if topology.boundary_side(l) != 'ya']
    if not enter_A or not enter_B:
        T = _balance_temperature(streams, inlets, topology.labels)
        return T, T

    # energy-balanced mix of the streams entering at one end (their common inlet T if equal)
    def mix(labels):
        temps = [inlets[l][0] for l in labels]
        if max(temps) - min(temps) < 1e-6:
            return temps[0]
        return _balance_temperature(streams, inlets, labels)

    try:
        # z=L end dominated by the streams entering there; T_A closes the overall balance
        T_B = mix(enter_B)
        T_A = _balance_temperature(streams, inlets, enter_B, fixed={l: T_B for l in enter_A})
        return T_A, T_B
    except ValueError:
        # otherwise the z=0 end dominates; T_B closes the overall balance
        T_A = mix(enter_A)
        T_B = _balance_temperature(streams, inlets, enter_A, fixed={l: T_A for l in enter_B})
        return T_A, T_B


def physical_initial_guess(x, streams, topology, boundary_properties, rhs_physical):
    """Starting guess (n_states x len(x), physical units) on mesh x.
    rhs_physical(y) returns dy/dz for one state column; it is used to estimate
    how fast each stream approaches the background from its inlet.
    Returns (guess, (T_A, T_B), relaxation lengths).
    """
    length = x[-1] - x[0]
    inlets = _inlets(topology, boundary_properties)
    T_A, T_B = end_temperatures(streams, topology, boundary_properties)
    # background temperature, linear between the two ends
    background = lambda z: T_A + (T_B - T_A) * (z - x[0]) / length

    # background temperature at the end where a stream leaves
    def outlet_T(label):
        return T_B if topology.boundary_side(label) == 'ya' else T_A

    relax = {}
    # relaxation length per stream: everyone at the background at this stream's inlet,
    # except the stream itself at its inlet state; lambda = |T_in - T_bg| / |dT/dz|
    for label, (T_in, P, xp_in) in inlets.items():
        e = topology.index[label]
        z_in = x[0] if topology.boundary_side(label) == 'ya' else x[-1]
        y = np.zeros(topology.n_states)
        for other, (_, P_o, xp_o) in inlets.items():
            eo = topology.index[other]
            y[eo['T']] = background(z_in)
            y[eo['P']] = P_o / 1e3
            if eo['xp'] is not None:
                y[eo['xp']] = _xp_guess(xp_o, outlet_T(other))
        y[e['T']] = T_in
        if e['xp'] is not None:
            y[e['xp']] = xp_in
        dTdz = rhs_physical(y)[e['T']]
        dT = abs(T_in - background(z_in))
        if dT < 1e-6 or not np.isfinite(dTdz) or abs(dTdz) < 1e-12:
            lam = MAX_RELAXATION_FRACTION * length
        else:
            lam = dT / abs(dTdz)
        # clamp: resolvable on the mesh, and finite for uncoupled streams
        relax[label] = min(max(lam, MIN_RELAXATION_FRACTION * length),
                           MAX_RELAXATION_FRACTION * length)

    # profiles: exponential approach from each inlet onto the background
    sol = np.zeros((topology.n_states, len(x)))
    T_bg = background(x)
    for label, (T_in, P, xp_in) in inlets.items():
        e = topology.index[label]
        co = topology.boundary_side(label) == 'ya'
        # distance from the stream's own inlet
        d = (x - x[0]) if co else (x[-1] - x)
        z_in = x[0] if co else x[-1]
        sol[e['T'], :] = T_bg + (T_in - background(z_in)) * np.exp(-d / relax[label])
        sol[e['P'], :] = P / 1e3
        if e['xp'] is not None:
            # catalysed streams: para fraction linear towards the halfway target; small pressure drop
            frac = d / length
            sol[e['xp'], :] = xp_in + (_xp_guess(xp_in, outlet_T(label)) - xp_in) * frac
            sol[e['P'], :] = P / 1e3 * (1 - 0.003 * frac)
    return sol, (T_A, T_B), relax
