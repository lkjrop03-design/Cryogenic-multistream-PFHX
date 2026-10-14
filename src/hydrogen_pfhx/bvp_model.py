"""The boundary-value problem solved by scipy's solve_bvp (set up in model.model).

State vector per stream (layout from topology.Topology): para fraction
(catalysed streams only), pressure (kPa) and temperature (K), as functions
of the axial position z. bvp_function gives dy/dz from the energy, species
and momentum balances; boundary_condition pins every stream's inlet state
at the end where it enters; initialise_solution builds the starting guess.
In the normalised variants the solver works with dimensionless states
(divided by the inlet values, get_state_scales); in the cache variants
property evaluations go through a PropertyCache.
"""


import numpy as np
from hydrogen_pfhx import ortho_para_dynamics, starting_guess


def build_boundary_properties(topology, streams_config):
    """Inlet values of every state row, in solver units (kPa, K, mol/mol)."""
    bp = np.zeros(topology.n_states)
    for label in topology.labels:
        spec = streams_config[label]
        e = topology.index[label]
        if e['xp'] is not None:
            bp[e['xp']] = spec['x_para']
        bp[e['P']] = spec['pressure']
        bp[e['T']] = spec['temperature']
    return bp


def get_state_scales(topology, boundary_properties):
    """Scale factor per state row of the normalised solver: each stream's inlet
    P and inlet T (1 for the para fraction), so the solver's states are of
    order one. physical = dimensionless * scales; used to unscale states in
    update_parameters and to scale derivatives, boundary values and guesses.
    """
    scales = np.ones(topology.n_states)
    for label in topology.labels:
        e = topology.index[label]
        scales[e['P']] = boundary_properties[e['P']]
        scales[e['T']] = boundary_properties[e['T']]
    return scales


# upper pressure clamp for solver iterates and limit for inlets (kPa); hydrogen's melting
# temperature at this pressure (12.7 K) stays below its temperature floor
PRESSURE_CEILING_KPA = 2.0e4


def update_parameters(process_properties, streams, topology, boundary_properties):
    """Set every stream's state from one column of the solver state and evaluate
    its properties. Solver iterates are made safe for the property models: a
    NaN is replaced by the inlet value, and T, P and xp are clamped (T between
    the fluid's solid-phase floor and 500 K, P below PRESSURE_CEILING_KPA).
    """
    # the solver works in dimensionless states: convert this column to physical units
    scales = get_state_scales(topology, boundary_properties)
    process_properties = process_properties * scales

    # clamp limits: minimum pressure (kPa), maximum temperature (K) and pressure
    nz_limit = 1e-6
    T_ceiling = 500.0
    P_ceiling = PRESSURE_CEILING_KPA

    for label in topology.labels:
        e = topology.index[label]
        stream = streams[label]

        P_raw = process_properties[e['P']]
        T_raw = process_properties[e['T']]
        # replace NaN from a wild Newton iterate by the inlet value before clamping
        if np.isnan(P_raw):
            P_raw = boundary_properties[e['P']]
        if np.isnan(T_raw):
            T_raw = boundary_properties[e['T']]
        # clamp P (to Pa) and T (between the fluid's solid-phase floor at P and T_ceiling)
        P = np.min((P_ceiling, np.max((nz_limit, P_raw)))) * 1e3
        T = np.min((T_ceiling, np.max((stream.temperature_floor(P), T_raw))))
        stream.update_conditions(T, P)

        if e['xp'] is not None:
            # para fraction (catalysed streams), kept within [0, 1]
            xp_raw = process_properties[e['xp']]
            if np.isnan(xp_raw):
                xp_raw = boundary_properties[e['xp']]
            xp = np.max((0.0, np.min((xp_raw, 1.0))))
            stream.update_composition(xp)

        # evaluate the stream's properties at its new state
        stream.set_properties()

    return streams


def evaluate_transport_behaviour(reactor, catalyst, streams, topology):
    """Velocities, Reynolds numbers and film coefficients of all streams at the
    current node (needs every stream's state set). Returns the temperatures,
    which calculate_heat_transfer_duty needs. Also used by outputs.post_process.
    """
    temperatures = {l: streams[l].temperature for l in topology.labels}

    for label in topology.labels:
        stream = streams[label]
        if topology.is_packed(label):
            stream.calculate_velocity(
                # catalysed streams flow through the pores: interstitial velocity
                reactor.total_side_area(label) * catalyst.void_fraction)
        else:
            stream.calculate_velocity(reactor.total_side_area(label))
            # fin streams: Reynolds number on the Manglik-Bergles hydraulic diameter
            stream.calculate_reynolds_number(reactor.offset_strip_hydraulic_diameter)

    for label in topology.labels:
        # film coefficients need every stream's velocity and temperature, hence a second pass
        reactor.transport_behaviour(label, streams[label], catalyst, temperatures)

    return temperatures


def net_heat_gained(label, duties, topology):
    """Net heat (W/m) a stream gains from all its interfaces; duties[(a, b)] > 0
    means heat flows from a to b.
    """
    total = 0.0
    for (a, b), q in duties.items():
        if a == label:
            total -= q
        elif b == label:
            total += q
    return total


def bvp_function(z, process_properties, additional_parameters):
    """Right-hand side dy/dz at every mesh node (called by solve_bvp).

    Energy:   m cp dT/dz = sign * (net interface heat - dH_conv * dN_p/dz)
    Species:  dx_p/dz = sign * dN_p/dz * M / m,  dN_p/dz = r * A_packed
    Momentum: dP/dz from the stream's friction correlation
    sign is the flow direction (+1 co-current, -1 counter-current); r is the
    conversion rate per volume of packed duct, A_packed the stream's total
    catalyst-filled cross-section.
    """
    reactor, catalyst, streams, topology, boundary_properties = additional_parameters
    scales = get_state_scales(topology, boundary_properties)

    # evaluate the balances node by node
    delta = np.zeros(process_properties.shape)

    for zi in np.arange(len(z)):
        streams = update_parameters(
            process_properties[:, zi], streams, topology, boundary_properties)
        temperatures = evaluate_transport_behaviour(
            reactor, catalyst, streams, topology)
        # heat flow across every interface at this node
        duties, _ = reactor.calculate_heat_transfer_duty(temperatures)

        for label in topology.labels:
            e = topology.index[label]
            stream = streams[label]

            heat_gained = net_heat_gained(label, duties, topology)

            if topology.is_packed(label):
                # catalysed stream: para production along the flow and the conversion heat it releases
                r_dot = ortho_para_dynamics.first_order_kinetics(stream)
                dNpdz = r_dot * reactor.total_side_area(label)
                delta[e['xp'], zi] = topology.flow_sign(label) * dNpdz * \
                    stream.molecular_mass / stream.mass_flow_rate
                heat_gained -= stream.get_heat_of_conversion() * dNpdz

            # energy balance with frozen-composition cp; momentum balance from the friction correlation
            delta[e['T'], zi] = topology.flow_sign(label) * heat_gained / (
                stream.mass_flow_rate * stream.specific_heat_capacity)
            delta[e['P'], zi] = reactor.pressure_drop(
                label, stream, catalyst) / 1e3
        # derivatives of the dimensionless states
        delta[:, zi] = delta[:, zi] / scales

    return delta


def boundary_condition(inlet_properties, outlet_properties, topology, boundary_parameters):
    """Residuals of the inlet conditions: every stream's state rows are pinned
    at the end where it enters (z=0 for co-current, z=L for counter-current).
    """
    # inlet values in the solver's dimensionless units
    scales = get_state_scales(topology, boundary_parameters)
    boundary_parameters = boundary_parameters / scales
    residuals = []
    for label in topology.labels:
        side = inlet_properties if topology.boundary_side(label) == 'ya' else outlet_properties
        for row in topology.state_rows(label):
            residuals.append(side[row] - boundary_parameters[row])
    return np.array(residuals)


def build_bc_jacobians(topology):
    """Jacobians of boundary_condition with respect to y(0) and y(L): constant
    selector matrices (the residual is affine), built once in model.model.
    """
    n = topology.n_states
    dbc_dya = np.zeros((n, n))
    dbc_dyb = np.zeros((n, n))
    r = 0
    for label in topology.labels:
        # residual r compares state `row` at the stream's inlet end with its inlet value
        target = dbc_dya if topology.boundary_side(label) == 'ya' else dbc_dyb
        for row in topology.state_rows(label):
            target[r, row] = 1.0
            r += 1
    return dbc_dya, dbc_dyb


def initial_guess(x, topology, boundary_properties, outlet_temperatures, length):
    """Heuristic starting profiles (fallback): straight temperature lines from each
    inlet to a guessed outlet, flat pressures, and a para-fraction ramp to just
    below equilibrium, all running from each stream's own inlet.
    """
    nodes = len(x)
    sol = np.zeros((topology.n_states, nodes))

    for label in topology.labels:
        e = topology.index[label]
        T_in = boundary_properties[e['T']]
        T_out = outlet_temperatures[label]

        if topology.boundary_side(label) == 'ya':
            # co-current: inlet at z=0; counter-current: inlet at z=L
            sol[e['T'], :] = T_in + (T_out - T_in) * x / length
        else:
            sol[e['T'], :] = T_out + (T_in - T_out) * x / length

        sol[e['P'], :] = boundary_properties[e['P']] * np.ones(nodes)

        if e['xp'] is not None:
            # catalysed streams: ramp from the inlet to just below equilibrium at the outlet
            # temperature, from the stream's own inlet; small pressure drop
            xp_in = boundary_properties[e['xp']]
            xp_eq, _ = ortho_para_dynamics.para_ortho_equilibrium(T_out)
            xp_target = xp_eq - 0.05
            frac = x / length if topology.boundary_side(label) == 'ya' else 1 - x / length
            sol[e['xp'], :] = xp_in + (xp_target - xp_in) * frac
            sol[e['P'], :] = boundary_properties[e['P']] * (1 - 0.003 * frac)

    # starting profiles in the solver's dimensionless units
    scales = get_state_scales(topology, boundary_properties)
    return sol / scales[:, np.newaxis]


# which starting guess the last initialise_solution call used
LAST_STARTING_GUESS = 'not run'
# default starting guess of this variant: 'physics' (starting_guess.py) or 'heuristic';
# overridden by simulation: starting_guess in the configuration
DEFAULT_STARTING_GUESS = 'physics'


def initialise_solution(streams, reactor, catalyst, topology, boundary_properties,
                        simulation_configuration):
    """Mesh and starting guess for solve_bvp: the physics-based guess of
    starting_guess, or the heuristic guess if that fails. Records which one was
    used in LAST_STARTING_GUESS (reported in the run diagnostics).
    """
    global LAST_STARTING_GUESS
    # heuristic guess when selected (default of the variants without the physics-based guess)
    if simulation_configuration.get('starting_guess', DEFAULT_STARTING_GUESS) == 'heuristic':
        LAST_STARTING_GUESS = 'heuristic'
        return _heuristic_initialise_solution(
            streams, reactor, catalyst, topology, boundary_properties,
            simulation_configuration)

    nodes = simulation_configuration['nodes']
    x = np.linspace(0, reactor.length, nodes)
    # state scales (normalised variants) or ones
    get_scales = globals().get('get_state_scales')
    scales = (get_scales(topology, boundary_properties) if get_scales is not None
              else np.ones(topology.n_states))
    additional_parameters = (reactor, catalyst, streams, topology, boundary_properties)

    # the model's own right-hand side in physical units, used by the starting guess
    def rhs_physical(y_phys):
        y = (np.asarray(y_phys, dtype=float) / scales)[:, np.newaxis]
        return bvp_function(np.array([x[0]]), y, additional_parameters)[:, 0] * scales

    try:
        sol, end_temperatures, relax = starting_guess.physical_initial_guess(
            x, streams, topology, boundary_properties, rhs_physical)
        if not np.all(np.isfinite(sol)):
            raise ValueError('non-finite starting guess')
        LAST_STARTING_GUESS = 'physics-based'
        return x, sol / scales[:, np.newaxis]
    except Exception as err:
        LAST_STARTING_GUESS = 'heuristic fallback (%s)' % err
        # fallback: the heuristic guess (records the reason)
        print('Physics-based starting guess failed (%s); using the heuristic guess.' % err)
        return _heuristic_initialise_solution(
            streams, reactor, catalyst, topology, boundary_properties,
            simulation_configuration)


def _heuristic_initialise_solution(streams, reactor, catalyst, topology, boundary_properties,
                        simulation_configuration):
    """Fallback mesh and guess: hot streams (inlet above the mid temperature) are
    guessed to leave delta_t above the coldest inlet, the others delta_t below
    the hottest inlet; profiles from initial_guess.
    """
    nodes = simulation_configuration['nodes']
    delta_T = simulation_configuration['delta_t']
    x = np.linspace(0, reactor.length, nodes)

    inlet_temperatures = {
        l: boundary_properties[topology.index[l]['T']] for l in topology.labels}
    # guessed outlet temperatures from the extreme inlets
    coldest = min(inlet_temperatures.values())
    hottest = max(inlet_temperatures.values())

    outlet_temperatures = {}
    for label in topology.labels:
        T_in = inlet_temperatures[label]
        if T_in > 0.5 * (coldest + hottest):
            outlet_temperatures[label] = max(coldest + delta_T, coldest)
        else:
            outlet_temperatures[label] = min(hottest - delta_T, hottest)

    sol_init = initial_guess(
        x, topology, boundary_properties, outlet_temperatures, reactor.length)
    return x, sol_init
