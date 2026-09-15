"""
Generic (topology-driven) BVP model.

Every per-stream equation is assembled by looping over
`topology.labels`; nothing about stream count, stacking order or flow
direction is hard-coded. This replaces the hand-written
IDX_* constants and per-topology energy-balance blocks.

State vector layout comes from `topology.index` (see topology.py):
per stream, xp (packed-bed streams only), then P, then T.

Sign convention - the single rule that replaces every hand-derived
per-stream sign:

    dT/dz = flow_sign * (net heat GAINED by the stream) / (m_dot * cp)

with flow_sign = +1 co-current, -1 counter-current. Verified to
reproduce every hand-built topology's signs exactly:
  co-current + losing heat       -> negative dT/dz
  counter-current + gaining heat -> negative dT/dz
  counter-current + losing heat  -> POSITIVE dT/dz
Duties are returned by hexs as duties[(a, b)] > 0 meaning heat flows
a -> b, so a stream's net gain sums -duty over interfaces where it is
the first member and +duty where it is the second.
"""

import numpy as np
from hydrogen_pfhx import ortho_para_dynamics


# ----------------------------------------------------------------------
# state <-> physical conversion
# ----------------------------------------------------------------------
def build_boundary_properties(topology, streams_config):
    """Flat inlet-condition array laid out per topology.index."""
    bp = np.zeros(topology.n_states)
    for label in topology.labels:
        spec = streams_config[label]
        e = topology.index[label]
        if e['xp'] is not None:
            bp[e['xp']] = spec['x_para']
        bp[e['P']] = spec['pressure']
        bp[e['T']] = spec['temperature']
    return bp


def update_parameters(process_properties, streams, topology, boundary_properties):
    """Pushes one node's state into the fluid objects (physical units).

    Robustness clamps (all three carried over from fixes made to the
    hand-written models, each traced to a real reproduced failure):
      * T_floor - an intermediate Newton iterate reaching T=1e-6 K made
        CoolProp's own EOS solve return nonsense (a genuine singularity
        as T->0). The floor must sit AT OR ABOVE the triple point
        (13.957 K for normal hydrogen), the EOS's documented lower
        bound - 10 K was tried and still failed.
      * T_ceiling - an iterate reaching ~2770 K drove the catalyst
        conductivity correlation (k = 8.39 - 6.63e-3*T, no built-in
        floor) negative, flipping the sign of a log() argument inside
        effective_radial_conductivity and producing a NaN cascade.
      * P floor/ceiling - same principle, guarding the property calls.
    No cross-stream temperature ordering floor is applied: local
    temperature crossover between streams is physically legitimate in a
    multi-stream exchanger, and an earlier ordering floor was found to
    CAUSE a large energy imbalance rather than prevent one.
    """
    nz_limit = 1e-6
    T_floor = 15.0      # K, just above the 13.957 K triple point
    T_ceiling = 500.0   # K
    P_ceiling = 5.0e4   # kPa

    for label in topology.labels:
        e = topology.index[label]
        stream = streams[label]

        P = np.min((P_ceiling, np.max((nz_limit, process_properties[e['P']])))) * 1e3
        T = np.min((T_ceiling, np.max((T_floor, process_properties[e['T']]))))
        stream.update_conditions(T, P)

        if e['xp'] is not None:
            xp = np.max((0.0, np.min((process_properties[e['xp']], 1.0))))
            stream.update_composition(xp)

        stream.set_properties()

    return streams


# ----------------------------------------------------------------------
# right-hand side
# ----------------------------------------------------------------------
def evaluate_transport_behaviour(reactor, catalyst, streams, topology):
    """Velocity / Reynolds / film coefficients for every stream, in
    place. Split out so outputs.post_process can reuse it verbatim to
    recover transport quantities at the converged solution."""
    temperatures = {l: streams[l].temperature for l in topology.labels}

    for label in topology.labels:
        stream = streams[label]
        if topology.is_packed(label):
            stream.calculate_velocity(
                reactor.total_side_area(label) * catalyst.void_fraction)
            stream.calculate_reynolds_number(
                catalyst.particle_diameter / catalyst.solid_fraction)
        else:
            stream.calculate_velocity(reactor.total_side_area(label))
            stream.calculate_reynolds_number(reactor.hydraulic_diameter)

    # second pass: transport behaviour needs every stream's velocity and
    # Reynolds number already set (a packed stream's wall-temperature
    # blend reads its neighbours' temperatures)
    for label in topology.labels:
        reactor.transport_behaviour(label, streams[label], catalyst, temperatures)

    return temperatures


def net_heat_gained(label, duties, topology):
    """Net heat gained by `label` from all its interfaces (W/m).

    duties[(a, b)] > 0 means heat flows a -> b, so it is a loss for a
    and a gain for b.
    """
    total = 0.0
    for (a, b), q in duties.items():
        if a == label:
            total -= q
        elif b == label:
            total += q
    return total


def bvp_function(z, process_properties, additional_parameters):
    reactor, catalyst, streams, topology, boundary_properties = additional_parameters

    delta = np.zeros(process_properties.shape)

    for zi in np.arange(len(z)):
        streams = update_parameters(
            process_properties[:, zi], streams, topology, boundary_properties)
        temperatures = evaluate_transport_behaviour(
            reactor, catalyst, streams, topology)
        duties, _ = reactor.calculate_heat_transfer_duty(temperatures)

        for label in topology.labels:
            e = topology.index[label]
            stream = streams[label]

            heat_gained = net_heat_gained(label, duties, topology)

            # reaction only occurs in packed (catalysed) streams
            if topology.is_packed(label):
                r_dot = ortho_para_dynamics.first_order_kinetics(stream)
                dNpdz = r_dot * reactor.total_side_area(label) * catalyst.solid_fraction
                delta[e['xp'], zi] = dNpdz * stream.molecular_mass / stream.mass_flow_rate
                heat_gained -= stream.get_heat_of_conversion() * dNpdz

            delta[e['T'], zi] = topology.flow_sign(label) * heat_gained / (
                stream.mass_flow_rate * stream.specific_heat_capacity)
            delta[e['P'], zi] = reactor.pressure_drop(
                label, stream, catalyst) / 1e3

    return delta


# ----------------------------------------------------------------------
# boundary conditions
# ----------------------------------------------------------------------
def boundary_condition(inlet_properties, outlet_properties, topology, boundary_parameters):
    """Each stream's inlet conditions are pinned at whichever end it
    enters: z=0 (ya) for co-current, z=L (yb) for counter-current.

    A packed stream pins xp, P and T (3 residuals); a plain-fin stream
    pins P and T (2). Total always equals topology.n_states.
    """
    residuals = []
    for label in topology.labels:
        side = inlet_properties if topology.boundary_side(label) == 'ya' else outlet_properties
        for row in topology.state_rows(label):
            residuals.append(side[row] - boundary_parameters[row])
    return np.array(residuals)


def build_bc_jacobians(topology):
    """Constant selector matrices - the residual is affine, so its
    Jacobian never changes. Built once at setup and reused."""
    n = topology.n_states
    dbc_dya = np.zeros((n, n))
    dbc_dyb = np.zeros((n, n))
    r = 0
    for label in topology.labels:
        target = dbc_dya if topology.boundary_side(label) == 'ya' else dbc_dyb
        for row in topology.state_rows(label):
            target[r, row] = 1.0
            r += 1
    return dbc_dya, dbc_dyb


# ----------------------------------------------------------------------
# initial guess
# ----------------------------------------------------------------------
def initial_guess(x, topology, boundary_properties, outlet_temperatures, length):
    """Linear temperature profiles between each stream's own inlet and a
    guessed outlet, flat pressures, and a linear composition ramp toward
    (just short of) equilibrium for packed streams.

    Co-current streams run inlet(z=0) -> outlet(z=L); counter-current
    streams run outlet(z=0) -> inlet(z=L).
    """
    nodes = len(x)
    sol = np.zeros((topology.n_states, nodes))

    for label in topology.labels:
        e = topology.index[label]
        T_in = boundary_properties[e['T']]
        T_out = outlet_temperatures[label]

        if topology.boundary_side(label) == 'ya':      # co-current
            sol[e['T'], :] = T_in + (T_out - T_in) * x / length
        else:                                           # counter-current
            sol[e['T'], :] = T_out + (T_in - T_out) * x / length

        sol[e['P'], :] = boundary_properties[e['P']] * np.ones(nodes)

        if e['xp'] is not None:
            xp_in = boundary_properties[e['xp']]
            xp_eq, _ = ortho_para_dynamics.para_ortho_equilibrium(T_out)
            xp_target = xp_eq - 0.05
            sol[e['xp'], :] = xp_in + (xp_target - xp_in) * x / length
            # packed streams also see a small pressure drop
            sol[e['P'], :] = boundary_properties[e['P']] * (1 - 0.003 * x / length)

    return sol


def initialise_solution(streams, reactor, catalyst, topology, boundary_properties,
                        simulation_configuration):
    """Builds the mesh and an initial guess.

    Heuristic: each packed/hot stream is guessed to leave near the
    coldest inlet in the exchanger, and each other stream near the
    hottest, offset by delta_t. This is deliberately simpler than the
    duty-balancing heuristic the hand-written topologies used - that one
    had to know which specific streams exchanged with which, which is
    exactly what varies here. It is only a starting point for the
    solver, not an answer.
    """
    nodes = simulation_configuration['nodes']
    delta_T = simulation_configuration['delta_t']
    x = np.linspace(0, reactor.length, nodes)

    inlet_temperatures = {
        l: boundary_properties[topology.index[l]['T']] for l in topology.labels}
    coldest = min(inlet_temperatures.values())
    hottest = max(inlet_temperatures.values())

    outlet_temperatures = {}
    for label in topology.labels:
        T_in = inlet_temperatures[label]
        if T_in > 0.5 * (coldest + hottest):
            # runs hot relative to the exchanger: leaves toward the cold end
            outlet_temperatures[label] = max(coldest + delta_T, coldest)
        else:
            outlet_temperatures[label] = min(hottest - delta_T, hottest)

    sol_init = initial_guess(
        x, topology, boundary_properties, outlet_temperatures, reactor.length)
    return x, sol_init
