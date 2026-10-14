"""Enthalpy-based composite curves and the global temperature-crossover check,
run by model.model on every converged solution.

Streams are classified hot or cold from their own inlet and outlet. Each
stream's heat load is Q_i(T) = m_dot (h(T) - h at its cold end), from the
real enthalpy (including the ortho-para conversion heat), and the loads are
summed into a hot and a cold composite. Both are read at equal heat duty Q,
measured from the cold end; a negative temperature difference anywhere
means heat would have to flow from colder to hotter streams. The smallest
difference is the pinch, reported in the run diagnostics.
"""


import numpy as np
import pandas as pd


class TemperatureCrossoverError(ValueError):

    """Raised by model.model on a crossover when simulation.raise_on_crossover
    is true; carries the run's results so a batch runner can still save them.
    """
    def __init__(self, message, results=None, run_diagnostics=None,
                 composite_table=None):
        super().__init__(message)
        self.results = results
        self.run_diagnostics = run_diagnostics
        self.composite_table = composite_table


def _flow_ordered(results, topology, label):
    """Temperature and enthalpy of one stream from its own inlet to its outlet."""
    T = results['%s temperature (K)' % label].values
    h = results['%s enthalpy (J/kg)' % label].values
    if topology.boundary_side(label) == 'ya':
        return T.copy(), h.copy()
    return T[::-1].copy(), h[::-1].copy()


def classify_streams(results, topology, min_delta_T=1e-6):
    """Hot (cools along its flow), cold (warms) or idle (unchanged) streams."""
    hot, cold, idle = [], [], []
    for label in topology.labels:
        T, _ = _flow_ordered(results, topology, label)
        if T[0] - T[-1] > min_delta_T:
            hot.append(label)
        elif T[-1] - T[0] > min_delta_T:
            cold.append(label)
        else:
            idle.append(label)
    return hot, cold, idle


def _stream_heat_curve(T_flow, h_flow, mdot, is_hot):
    """Heat-load curve of one stream: (T increasing, Q(T) non-decreasing,
    largest rewarming excursion in K). Q is measured from the stream's cold end.
    A stream whose temperature reverses is replaced by its monotonic envelope
    (running minimum for hot, maximum for cold streams): heat released while a
    hot stream rewarms is counted at the lowest temperature already reached,
    which can only move the hot curve down (conservative).
    """
    if is_hot:
        # hot stream: running minimum; Q from the outlet (cold end), reordered cold end first
        T_env = np.minimum.accumulate(T_flow)
        excursion = float(np.max(T_flow - T_env))
        Q = mdot * (h_flow - h_flow[-1])
        T_env, Q = T_env[::-1], Q[::-1]
    else:
        # cold stream: running maximum; Q from the inlet (cold end)
        T_env = np.maximum.accumulate(T_flow)
        excursion = float(np.max(T_env - T_flow))
        Q = mdot * (h_flow - h_flow[0])

    # Q never decreases with temperature (smooths numerical wiggles in h)
    Q = np.maximum.accumulate(Q)

    # isothermal stretches: keep the largest Q at a repeated temperature
    T_unique, first = np.unique(T_env, return_index=True)
    Q_unique = np.maximum.reduceat(Q, first)
    return T_unique, Q_unique, excursion


def _composite(curves, T_grid):
    """Summed heat load of a set of streams on a temperature grid."""
    Q_total = np.zeros_like(T_grid)
    for T_s, Q_s in curves:
        # zero below a stream's range, its full duty above it
        Q_total += np.interp(T_grid, T_s, Q_s, left=0.0, right=Q_s[-1])
    return Q_total


def build_composite_curves(results, streams, topology, n_points=1000):
    """Hot and cold composite curves on a common heat-duty axis.
    Returns (table of Q, both composite temperatures and their difference,
    info with the classification and the composite duties).
    """
    hot, cold, idle = classify_streams(results, topology)
    if not hot or not cold:
        raise ValueError(
            'Composite curves need at least one hot and one cold stream '
            '(hot: %s, cold: %s, unchanged: %s).' % (hot, cold, idle))

    hot_curves, cold_curves, excursions = [], [], {}
    # one heat-load curve per stream, collected per side
    for label in hot + cold:
        T_f, h_f = _flow_ordered(results, topology, label)
        T_s, Q_s, exc = _stream_heat_curve(
            T_f, h_f, streams[label].mass_flow_rate, is_hot=(label in hot))
        (hot_curves if label in hot else cold_curves).append((T_s, Q_s))
        excursions[label] = exc

    # each composite on its own temperature grid: every stream breakpoint plus a dense fill
    def own_grid(curves):
        T_all = np.concatenate([c[0] for c in curves])
        return np.unique(np.concatenate([
            T_all, np.linspace(T_all.min(), T_all.max(), n_points)]))

    T_grid_hot, T_grid_cold = own_grid(hot_curves), own_grid(cold_curves)
    Q_hot_T = _composite(hot_curves, T_grid_hot)
    Q_cold_T = _composite(cold_curves, T_grid_cold)

    Q_hot_total, Q_cold_total = Q_hot_T[-1], Q_cold_T[-1]
    # common heat-duty axis, from the cold end up to the smaller composite duty
    Q_max = min(Q_hot_total, Q_cold_total)
    Q_axis = np.linspace(0.0, Q_max, n_points)

    # invert Q(T); where a composite is flat in Q, take the conservative end:
    # lowest T for the hot curve, highest T for the cold curve
    def T_at_Q(Q_of_T, T_grid, lowest):
        Qm = np.maximum.accumulate(Q_of_T)
        if lowest:
            Q_u, idx = np.unique(Qm, return_index=True)
            T_u = T_grid[idx]
        else:
            Q_u, idx = np.unique(Qm[::-1], return_index=True)
            T_u = T_grid[::-1][idx]
        return np.interp(Q_axis, Q_u, T_u)

    T_hot_Q = T_at_Q(Q_hot_T, T_grid_hot, lowest=True)
    T_cold_Q = T_at_Q(Q_cold_T, T_grid_cold, lowest=False)

    table = pd.DataFrame({
        'Q from cold end (W)': Q_axis,
        'Hot composite temperature (K)': T_hot_Q,
        'Cold composite temperature (K)': T_cold_Q,
        'Temperature difference (K)': T_hot_Q - T_cold_Q,
    })
    info = {
        'hot_streams': hot, 'cold_streams': cold, 'idle_streams': idle,
        'hot_composite_duty_W': float(Q_hot_total),
        'cold_composite_duty_W': float(Q_cold_total),
        'rewarming_excursion_K': excursions,
    }
    return table, info


def check_composite_crossover(results, streams, topology,
                              tolerance_K=0.0, n_points=1000):
    """Composite curves plus the pinch (smallest temperature difference) and a
    feasibility flag; differences above -tolerance_K count as feasible
    (solver noise). Returns (table, diagnostics).
    """
    table, info = build_composite_curves(results, streams, topology, n_points)
    dT = table['Temperature difference (K)'].values
    # pinch: location of the smallest hot-minus-cold temperature difference
    i = int(np.argmin(dT))

    diag = {
        'composite_feasible': bool(dT[i] >= -tolerance_K),
        'pinch_dT_K': float(dT[i]),
        'pinch_Q_W': float(table['Q from cold end (W)'].iloc[i]),
        'pinch_T_hot_K': float(table['Hot composite temperature (K)'].iloc[i]),
        'pinch_T_cold_K': float(table['Cold composite temperature (K)'].iloc[i]),
        'composite_hot_streams': ', '.join(info['hot_streams']),
        'composite_cold_streams': ', '.join(info['cold_streams']),
        'hot_composite_duty_W': info['hot_composite_duty_W'],
        'cold_composite_duty_W': info['cold_composite_duty_W'],
        'max_rewarming_excursion_K': float(max(info['rewarming_excursion_K'].values())),
    }
    return table, diag
