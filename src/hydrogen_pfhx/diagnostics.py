"""Run-time diagnostics of a BVP solve: wall time, convergence status and the
per-iteration log that scipy's solve_bvp prints (verbose=2). model.model
merges these with the physical checks of outputs.check_physical_consistency.
"""

import io
import time
import contextlib
import os
import numpy as np
import pandas as pd
import sys

class _Tee(io.StringIO):
    """Text stream that writes to the real stdout and keeps a copy, so the
    solver's iteration log is both shown and parsed.
    """
    def __init__(self, real_stdout):
        super().__init__()
        self._real_stdout = real_stdout

    def write(self, s):
        self._real_stdout.write(s)
        return super().write(s)

    def flush(self):
        self._real_stdout.flush()

def parse_solve_bvp_log(log_text):
    """Turn solve_bvp's printed iteration table into a DataFrame
    (iteration, residuals, nodes); other printed lines are skipped.
    """
    records = []
    for line in log_text.splitlines():
        parts = line.split()
        # iteration rows have exactly five numeric columns
        if len(parts) != 5:
            continue
        try:
            records.append({
                'iteration': int(parts[0]),
                'max_residual': float(parts[1]),
                'max_bc_residual': float(parts[2]),
                'total_nodes': int(parts[3]),
                'nodes_added': int(parts[4]),
            })
        except ValueError:
            continue
    return pd.DataFrame(records)


def solve_with_diagnostics(solve_fn, *args, **kwargs):
    """Run solve_fn (scipy solve_bvp) with verbose output captured, and return
    (solution, run_diagnostics, iteration_log). run_diagnostics holds status,
    iterations, final node count, largest residual and runtime.
    """
    # verbose=2 makes solve_bvp print the per-iteration table
    kwargs.setdefault('verbose', 2)
    log_stream = _Tee(sys.stdout)
    # wall-clock time and processor time of this process (excludes sleep and waiting)
    t0 = time.perf_counter()
    c0 = time.process_time()
    with contextlib.redirect_stdout(log_stream):
        solution = solve_fn(*args, **kwargs)
    runtime_s = time.perf_counter() - t0
    cpu_time_s = time.process_time() - c0

    iteration_log = parse_solve_bvp_log(log_stream.getvalue())

    # summary of the solve; per-stream physical checks are added by model.model
    run_diagnostics = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'success': solution.success,
        'status': solution.status,
        'message': solution.message,
        'n_iterations': solution.niter,
        'n_nodes_final': len(solution.x),
        'max_rms_residual': float(np.max(solution.rms_residuals)) if len(solution.rms_residuals) else np.nan,
        'runtime_s': runtime_s,
        'cpu_time_s': cpu_time_s,
    }
    return solution, run_diagnostics, iteration_log
