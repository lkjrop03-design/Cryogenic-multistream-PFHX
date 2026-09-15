import numpy as np
from scipy.integrate import solve_bvp
from hydrogen_pfhx import (fluids, catalysts, hexs, bvp_model, helium_neon,
                            outputs, utils, diagnostics)
from hydrogen_pfhx import topology as topology_module


def _build_stream(spec):
    """Constructs one fluid stream from its config block. Any stream may
    be any supported fluid; `type` (packed_bed/plain_fin) and
    `direction` are structural and consumed by the Topology, not here."""
    mass_flowrate_kps = utils.tpd_to_kps(spec['mass_flow_rate'])
    fluid_name = spec.get('fluid', 'Hydrogen')

    if fluid_name == 'Hydrogen':
        stream = fluids.Hydrogen(mass_flowrate_kps)
        stream.update_conditions(spec['temperature'], spec['pressure'] * 1e3)
        stream.update_composition(spec['x_para'])
    elif fluid_name == 'HeliumNeon':
        stream = helium_neon.setup_CoolProp(spec, mass_flowrate_kps)
        stream.update_conditions(spec['temperature'], spec['pressure'] * 1e3)
    else:
        raise ValueError("unsupported fluid %r" % (fluid_name,))

    stream.set_properties()
    return stream


def model(configuration):
    # Step 1. Structure comes entirely from layer_order + streams.
    topology = topology_module.from_configuration(configuration)

    streams = {label: _build_stream(configuration['streams'][label])
               for label in topology.labels}

    reactor = hexs.PlateFinHex(configuration['reactor'], topology)
    catalyst = catalysts.Catalyst(configuration['catalyst'])
    catalyst.calculate_catalyst_mass(
        sum(reactor.total_side_area(l) for l in topology.packed_labels())
        * reactor.length)

    # cheap invariant check: each interface's area must agree computed
    # from either side. Catches a topology-derivation error immediately
    # rather than as a silent energy imbalance later.
    reactor.check_area_consistency()

    # Step 2. Initial guess
    boundary_properties = bvp_model.build_boundary_properties(
        topology, configuration['streams'])
    x_mesh, sol_init = bvp_model.initialise_solution(
        streams, reactor, catalyst, topology, boundary_properties,
        configuration['simulation'])

    # Step 3. Solve
    additional_parameters = (reactor, catalyst, streams, topology,
                             boundary_properties)
    dbc_dya, dbc_dyb = bvp_model.build_bc_jacobians(topology)

    solution, run_diagnostics, iteration_log = diagnostics.solve_with_diagnostics(
        solve_bvp,
        lambda x, y: bvp_model.bvp_function(x, y, additional_parameters),
        lambda ya, yb: bvp_model.boundary_condition(
            ya, yb, topology, boundary_properties),
        x_mesh, sol_init,
        bc_jac=lambda ya, yb: (dbc_dya, dbc_dyb),
        tol=configuration['simulation']['tolerance'], max_nodes=1000)

    run_diagnostics['n_nodes_initial'] = len(x_mesh)
    run_diagnostics['tolerance'] = configuration['simulation']['tolerance']
    run_diagnostics['topology'] = '-'.join(topology.layer_order)
    run_diagnostics['n_states'] = topology.n_states

    if not solution.success:
        print('Solver did not converge: %s' % solution.message)

    # Step 4. Post-process
    results = outputs.post_process(
        solution, streams, reactor, catalyst, topology, boundary_properties)
    run_diagnostics.update(
        outputs.check_physical_consistency(results, streams, topology))

    return results, run_diagnostics
