"""Continuation fallback for the BVP solve (called by model.model when a direct
solve of a case with catalysed streams fails).

Newton's method only converges from a starting point close enough to the
solution, and the ortho-para conversion heat couples a poor para-fraction
guess into the energy balance. The case is therefore re-solved as a chain:
conversion heat switched off, then at 50 %, then 100 %, each step starting
from the previous solution and its refined mesh. The final step is the
full-physics problem.
"""


from hydrogen_pfhx import ortho_para_dynamics

# fractions of the conversion heat solved in turn
DEFAULT_STEPS = (0.0, 0.5, 1.0)


def solve_with_conversion_heat_continuation(solve, streams, x_start, y_start,
                                            steps=DEFAULT_STEPS, clear_cache=None):
    """Re-solve with the conversion heat stepped through `steps` (fractions).

    solve(x, y) runs one BVP solve and returns (solution, run_diagnostics,
    iteration_log). The conversion heat is scaled through each hydrogen
    stream's para-enthalpy reference, which every enthalpy and heat-of-
    conversion evaluation reads; the references are always restored.
    clear_cache empties the property cache between steps (cached enthalpies
    depend on the reference). Returns the last step's triple, with the steps
    taken ('continuation_steps') and the summed runtime added.
    """
    # conversion-heat scale (~1417 J/mol): removing it makes the conversion heat ~0
    offset = ortho_para_dynamics.ORTHO_PARA_GROUND_STATE_GAP
    # every hydrogen stream's para equation of state, with its current reference
    paras = [s.para for s in streams.values() if hasattr(s, 'para')]
    full_ref = [p.h_ref for p in paras]
    x, y = x_start, y_start
    record, runtime, cpu_time = [], 0.0, 0.0
    try:
        for fraction in steps:
            for p, h_ref in zip(paras, full_ref):
                # shift para enthalpy so only `fraction` of the conversion heat remains
                p.h_ref = h_ref - (1.0 - fraction) * offset
            if clear_cache is not None:
                clear_cache()
            # solve from the previous step's solution; stop at the first failure
            solution, run_diagnostics, iteration_log = solve(x, y)
            runtime += run_diagnostics.get('runtime_s', 0.0)
            cpu_time += run_diagnostics.get('cpu_time_s', 0.0)
            record.append('%.0f%%: %s (%d nodes)' % (
                100 * fraction, 'converged' if solution.success else 'failed',
                len(solution.x)))
            if not solution.success:
                break
            # this step's solution and mesh start the next step
            x, y = solution.x, solution.y
    # restore the true references whatever happened
    finally:
        for p, h_ref in zip(paras, full_ref):
            p.h_ref = h_ref
        if clear_cache is not None:
            clear_cache()
    # stopped before the full conversion heat: not a valid full-physics result
    if record and not record[-1].startswith('100%'):
        solution.success = False
        run_diagnostics['success'] = False
    run_diagnostics['continuation_steps'] = '; '.join(record)
    run_diagnostics['runtime_s'] = runtime
    run_diagnostics['cpu_time_s'] = cpu_time
    return solution, run_diagnostics, iteration_log
