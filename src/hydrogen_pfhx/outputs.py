import os
import numpy as np
import pandas as pd
from hydrogen_pfhx import bvp_model, ortho_para_dynamics


def post_process(solution, streams, reactor, catalyst, topology, boundary_properties):
    """Builds the results DataFrame. Column names are generated from the
    stream labels and interface pairs in the topology, so a run's output
    always matches whatever configuration produced it."""
    z = solution.x
    n = len(z)

    columns = {'Z (m)': z}
    per_node = {}

    # transport quantities are not in the state vector - recover them by
    # re-evaluating at the converged solution, exactly as during the solve
    for label in topology.labels:
        per_node[('h', label)] = np.zeros(n)
        per_node[('velocity', label)] = np.zeros(n)
        per_node[('reynolds', label)] = np.zeros(n)
        per_node[('enthalpy', label)] = np.zeros(n)
        per_node[('density', label)] = np.zeros(n)
        per_node[('cp', label)] = np.zeros(n)
    for pair in topology.interfaces:
        per_node[('U', pair)] = np.zeros(n)
        per_node[('Q', pair)] = np.zeros(n)

    xp_equil = np.zeros(n)

    for i in range(n):
        streams = bvp_model.update_parameters(
            solution.y[:, i], streams, topology, boundary_properties)
        temperatures = bvp_model.evaluate_transport_behaviour(
            reactor, catalyst, streams, topology)
        duties, _ = reactor.calculate_heat_transfer_duty(temperatures)

        for label in topology.labels:
            s = streams[label]
            per_node[('h', label)][i] = reactor.h[label]
            per_node[('velocity', label)][i] = s.velocity
            per_node[('reynolds', label)][i] = s.reynolds_number
            per_node[('enthalpy', label)][i] = s.enthalpy
            per_node[('density', label)][i] = s.mass_density
            per_node[('cp', label)][i] = s.specific_heat_capacity

        for pair in topology.interfaces:
            a, b = pair
            wall = (temperatures[a] + temperatures[b]) / 2
            u, _ = reactor.overall_heat_transfer_coefficient(a, b, wall)
            per_node[('U', pair)][i] = u
            per_node[('Q', pair)][i] = duties[pair]

        # equilibrium composition is only meaningful for a reacting stream
        packed = topology.packed_labels()
        if packed:
            xp_eq, _ = ortho_para_dynamics.para_ortho_equilibrium(
                streams[packed[0]].temperature)
            xp_equil[i] = xp_eq

    # state-vector columns
    for label in topology.labels:
        e = topology.index[label]
        columns['%s pressure (kPa)' % label] = solution.y[e['P'], :]
        columns['%s temperature (K)' % label] = solution.y[e['T'], :]
        if e['xp'] is not None:
            columns['%s para fraction (mol/mol)' % label] = solution.y[e['xp'], :]

    if topology.packed_labels():
        columns['Equilibrium para fraction (mol/mol)'] = xp_equil

    # derived columns
    for label in topology.labels:
        columns['%s velocity (m/s)' % label] = per_node[('velocity', label)]
        columns['%s Reynolds number' % label] = per_node[('reynolds', label)]
        columns['%s film coefficient (W/m2K)' % label] = per_node[('h', label)]
        columns['%s density (kg/m3)' % label] = per_node[('density', label)]
        columns['%s cp (J/kgK)' % label] = per_node[('cp', label)]
        columns['%s enthalpy (J/kg)' % label] = per_node[('enthalpy', label)]

    for pair in topology.interfaces:
        a, b = pair
        columns['Overall U %s-%s (W/m2K)' % (a, b)] = per_node[('U', pair)]
        columns['Heat duty %s-%s (W/m)' % (a, b)] = per_node[('Q', pair)]

    return pd.DataFrame(columns)


def check_physical_consistency(results, streams, topology):
    """Energy balance and range checks, generated per stream.

    Each stream's duty uses its own flow direction to pick inlet vs
    outlet: a co-current stream enters at z=0, a counter-current stream
    at z=L, so 'gained' is (outlet - inlet) enthalpy in each case.
    """
    diagnostics = {}
    total_gained = 0.0

    for label in topology.labels:
        h = results['%s enthalpy (J/kg)' % label].values
        mdot = streams[label].mass_flow_rate
        if topology.boundary_side(label) == 'ya':   # enters z=0, leaves z=L
            duty = (h[-1] - h[0]) * mdot
        else:                                        # enters z=L, leaves z=0
            duty = (h[0] - h[-1]) * mdot
        diagnostics['%s_duty_W' % label] = duty
        total_gained += duty

        T = results['%s temperature (K)' % label].values
        diagnostics['%s_T_min_K' % label] = float(np.min(T))
        diagnostics['%s_T_max_K' % label] = float(np.max(T))
        diagnostics['%s_T_below_triple' % label] = bool(np.any(T < 13.957))

    # Sum of all streams' enthalpy changes should be zero (no heat loss
    # to surroundings is modelled). Normalised by the largest single
    # stream duty so the percentage is meaningful.
    largest = max((abs(diagnostics['%s_duty_W' % l]) for l in topology.labels),
                  default=1e-9)
    diagnostics['energy_imbalance_pct'] = 100 * abs(total_gained) / max(largest, 1e-9)

    packed = topology.packed_labels()
    if packed:
        label = packed[0]
        xp = results['%s para fraction (mol/mol)' % label].values
        xp_eq = results['Equilibrium para fraction (mol/mol)'].values
        # the reacting stream's outlet is at whichever end it exits
        outlet_i = -1 if topology.boundary_side(label) == 'ya' else 0
        diagnostics['xp_reactant_inlet'] = float(xp[0 if outlet_i == -1 else -1])
        diagnostics['xp_reactant_outlet'] = float(xp[outlet_i])
        diagnostics['xp_equil_outlet'] = float(xp_eq[outlet_i])
        diagnostics['approach_to_equilibrium'] = float(xp[outlet_i] - xp_eq[outlet_i])
        diagnostics['xp_bound_violation'] = bool(np.any(xp < 0) or np.any(xp > 1))

    dz = np.diff(results['Z (m)'].values)
    diagnostics['min_node_spacing_m'] = float(np.min(dz)) if len(dz) else np.nan
    diagnostics['max_node_spacing_m'] = float(np.max(dz)) if len(dz) else np.nan
    return diagnostics


def save_results(results, file_path='output/results.csv'):
    directory = os.path.dirname(file_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    results.to_csv(file_path, index=False)


_INVALID_SHEET_CHARS = set(':\\/?*[]')


def _sanitize_sheet_name(name, used_names):
    """Excel sheet names: <=31 chars, no : \\ / ? * [ ], unique."""
    clean = ''.join(c for c in str(name) if c not in _INVALID_SHEET_CHARS).strip()
    clean = clean[:31] or 'Sheet'
    base, suffix = clean, 1
    while clean in used_names:
        tag = '_%d' % suffix
        clean = base[:31 - len(tag)] + tag
        suffix += 1
    used_names.add(clean)
    return clean


def save_results_excel(runs, file_path='output/batch_results.xlsx'):
    """One sheet per run plus a leading summary sheet. Summary columns
    are generated from whichever diagnostics each run produced, so
    per-stream duty columns follow the topology automatically."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    summary_rows = []
    for run in runs:
        diag = run.get('run_diagnostics') or {}
        row = {'label': run['label'], 'error': run.get('error')}
        for key in ('success', 'topology', 'n_states', 'n_nodes_final',
                    'runtime_s', 'energy_imbalance_pct', 'xp_reactant_outlet',
                    'xp_equil_outlet', 'approach_to_equilibrium'):
            row[key] = diag.get(key)
        for key, value in diag.items():
            if key.endswith('_duty_W'):
                row[key] = value
        summary_rows.append(row)

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        used = set()
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name=_sanitize_sheet_name('summary', used), index=False)
        for run in runs:
            sheet = _sanitize_sheet_name(run['label'], used)
            if run.get('results') is not None:
                run['results'].to_excel(writer, sheet_name=sheet, index=False)
            else:
                pd.DataFrame({'error': [run.get('error') or 'run failed']}).to_excel(
                    writer, sheet_name=sheet, index=False)
