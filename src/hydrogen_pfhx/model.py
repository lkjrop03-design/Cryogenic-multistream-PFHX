"""Entry point of the model: model(configuration) solves one exchanger case.

Builds the topology, fluid streams, exchanger and catalyst from the
configuration, solves the boundary-value problem of bvp_model with scipy's
solve_bvp (with a conversion-heat continuation fallback), post-processes the
solution, and checks it: energy balance, limits and composite curves.
Returns (results table, run diagnostics); test_model.py runs it per case.
"""

import time
import numpy as np
from scipy.integrate import solve_bvp
from hydrogen_pfhx import (fluids, catalysts, hexs, bvp_model,
                            outputs, utils, diagnostics, composite_curves,
                            continuation)
from hydrogen_pfhx import topology as topology_module


def _check_inlet_limits(stream, spec, fluid_name):
    """Reject an inlet the solver could not represent: pressure above
    bvp_model.PRESSURE_CEILING_KPA, or temperature below the fluid's solid-phase
    floor at that pressure. Runs before any state is set, because hydrogen's
    equations of state already fail below the triple point.
    """
    P_pa = spec['pressure'] * 1e3
    if P_pa > bvp_model.PRESSURE_CEILING_KPA * 1e3:
        raise ValueError('%s inlet pressure %.0f kPa is above the model limit of %.0f kPa'
                         % (fluid_name, spec['pressure'], bvp_model.PRESSURE_CEILING_KPA))
    T_floor = stream.temperature_floor(P_pa)
    if spec['temperature'] < T_floor:
        raise ValueError(
            '%s inlet temperature %.2f K is below the model limit of %.2f K at '
            '%.0f kPa (triple point %.2f K, or the melting line at this pressure, '
            'plus %.1f K margin)' % (fluid_name, spec['temperature'], T_floor,
                                     spec['pressure'], stream.triple_point_K,
                                     stream.TEMPERATURE_FLOOR_MARGIN_K))


def _build_stream(spec):
    """Fluid stream from one 'streams' entry, at its inlet state: hydrogen with
    variable ortho/para composition, or any other CoolProp fluid.
    """
    # configuration: mass flow in t/d, pressure in kPa, temperature in K
    mass_flowrate_kps = utils.tpd_to_kps(spec['mass_flow_rate'])
    fluid_name = spec.get('fluid', 'Hydrogen')

    if fluid_name == 'Hydrogen':
        stream = fluids.Hydrogen(mass_flowrate_kps)
        _check_inlet_limits(stream, spec, fluid_name)
        stream.update_conditions(spec['temperature'], spec['pressure'] * 1e3)
        stream.update_composition(spec['x_para'])
    else:
        stream = fluids.FluidStream(fluid_name, mass_flowrate_kps)
        _check_inlet_limits(stream, spec, fluid_name)
        stream.update_conditions(spec['temperature'], spec['pressure'] * 1e3)

    stream.set_properties()
    return stream


def model(configuration):
    """Solve one case. Returns (results, run_diagnostics): the per-node results
    table of outputs.post_process, and a dict with solver diagnostics,
    physical checks, pinch and composite table.
    """
    # wall-clock and processor time of the whole run (set-up, guess, solve, post-processing)
    t_start, c_start = time.perf_counter(), time.process_time()
    topology = topology_module.from_configuration(configuration)

    # cache variants: start every run with an empty property cache
    cache = getattr(bvp_model, '_property_cache', None)
    if cache is not None:
        cache.clear()

    # model objects: one fluid stream per label, the exchanger and the catalyst
    streams = {label: _build_stream(configuration['streams'][label])
               for label in topology.labels}

    reactor = hexs.PlateFinHex(configuration['reactor'], topology)
    catalyst = catalysts.Catalyst(configuration['catalyst'])

    # every interface's contact area must agree from both sides
    reactor.check_area_consistency()

    # inlet values and the starting guess on the initial mesh
    boundary_properties = bvp_model.build_boundary_properties(
        topology, configuration['streams'])
    t_guess, c_guess = time.perf_counter(), time.process_time()
    x_mesh, sol_init = bvp_model.initialise_solution(
        streams, reactor, catalyst, topology, boundary_properties,
        configuration['simulation'])
    # time spent building the starting guess
    guess_runtime_s = time.perf_counter() - t_guess
    guess_cpu_time_s = time.process_time() - c_guess

    additional_parameters = (reactor, catalyst, streams, topology,
                             boundary_properties)
    # constant boundary-condition Jacobians
    dbc_dya, dbc_dyb = bvp_model.build_bc_jacobians(topology)

    # analytic Jacobian in the solver's dimensionless units (analytic_jacobian.py)
    from hydrogen_pfhx.analytic_jacobian import normalized_fun_jac as _normalized_fun_jac

    # one BVP solve from a given mesh and guess; reused by the continuation fallback
    def _solve(x_start, y_start):
        return diagnostics.solve_with_diagnostics(
            solve_bvp,
            lambda x, y: bvp_model.bvp_function(x, y, additional_parameters),
            lambda ya, yb: bvp_model.boundary_condition(
                ya, yb, topology, boundary_properties),
            x_start, y_start,
            fun_jac=lambda x, y: _normalized_fun_jac(
                x, y, additional_parameters, bvp_model.get_state_scales),
            bc_jac=lambda ya, yb: (dbc_dya, dbc_dyb),
            tol=configuration['simulation']['tolerance'], max_nodes=configuration['simulation'].get('max_nodes', 1000))
    # direct solve; if it fails for a case with catalysed streams, retry by stepping
    # the conversion heat up from zero (continuation.py)
    solution, run_diagnostics, iteration_log = _solve(x_mesh, sol_init)
    if not solution.success and topology.packed_labels():
        print('Direct solve failed (%s); retrying with conversion-heat '
              'continuation.' % solution.message)
        first_runtime = run_diagnostics.get('runtime_s', 0.0)
        first_cpu_time = run_diagnostics.get('cpu_time_s', 0.0)
        cache = getattr(bvp_model, '_property_cache', None)
        solution, run_diagnostics, iteration_log = \
            continuation.solve_with_conversion_heat_continuation(
                _solve, streams, x_mesh, sol_init,
                clear_cache=cache.clear if cache is not None else None)
        run_diagnostics['runtime_s'] += first_runtime
        run_diagnostics['cpu_time_s'] += first_cpu_time
    else:
        run_diagnostics['continuation_steps'] = 'not needed'
    # run information for the summary sheet
    run_diagnostics['starting_guess'] = getattr(bvp_model, 'LAST_STARTING_GUESS', 'unknown')

    run_diagnostics['n_nodes_initial'] = len(x_mesh)
    run_diagnostics['tolerance'] = configuration['simulation']['tolerance']
    run_diagnostics['topology'] = '-'.join(topology.layer_order)
    run_diagnostics['n_states'] = topology.n_states

    if not solution.success:
        print('Solver did not converge: %s' % solution.message)

    # per-node results table and physical checks (energy balance, limits)
    results = outputs.post_process(
        solution, streams, reactor, catalyst, topology, boundary_properties)
    run_diagnostics.update(
        outputs.check_physical_consistency(results, streams, topology))

    sim = configuration['simulation']
    # composite curves and pinch; differences above -tolerance_K are treated as solver noise
    tolerance_K = sim.get('crossover_tolerance_K', 0.05)
    composite_table, pinch = composite_curves.check_composite_crossover(
        results, streams, topology, tolerance_K=tolerance_K)
    run_diagnostics.update(pinch)
    run_diagnostics['composite_table'] = composite_table

    # a crossover is reported (summary status CROSSOVER), or raised if requested
    if not pinch['composite_feasible']:
        print('WARNING: composite curves cross (pinch dT = %.3f K at '
              'T_hot = %.3f K, T_cold = %.3f K) - flagged in the summary.'
              % (pinch['pinch_dT_K'], pinch['pinch_T_hot_K'],
                 pinch['pinch_T_cold_K']))
    if not pinch['composite_feasible'] and sim.get('raise_on_crossover', False):
        raise composite_curves.TemperatureCrossoverError(
            'Composite curves cross: at Q = %.4g W from the cold end the '
            'hot composite is at %.3f K and the cold composite at %.3f K '
            '(dT = %.3f K, tolerance %.3f K). Hot streams: %s; cold '
            'streams: %s. The solver converged, but this solution '
            'requires heat to flow from colder to hotter streams.'
            % (pinch['pinch_Q_W'], pinch['pinch_T_hot_K'],
               pinch['pinch_T_cold_K'], pinch['pinch_dT_K'], tolerance_K,
               pinch['composite_hot_streams'],
               pinch['composite_cold_streams']),
            results=results, run_diagnostics=run_diagnostics,
            composite_table=composite_table)

    # timing summary: guess, and the whole run (runtime_s / cpu_time_s cover the BVP solve only)
    run_diagnostics['guess_runtime_s'] = guess_runtime_s
    run_diagnostics['guess_cpu_time_s'] = guess_cpu_time_s
    run_diagnostics['total_runtime_s'] = time.perf_counter() - t_start
    run_diagnostics['total_cpu_time_s'] = time.process_time() - c_start
    return results, run_diagnostics
