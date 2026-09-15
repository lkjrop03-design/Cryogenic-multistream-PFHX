import io
import time
import contextlib
import os
import numpy as np
import pandas as pd
import sys

class _Tee(io.StringIO):
    """Writes to an internal buffer and to the real stdout simultaneously."""
    def __init__(self, real_stdout):
        super().__init__()
        self._real_stdout = real_stdout

    def write(self, s):
        self._real_stdout.write(s)
        return super().write(s)

    def flush(self):
        self._real_stdout.flush()

def parse_solve_bvp_log(log_text):
    """Parses scipy solve_bvp's verbose=2 stdout into a per-iteration DataFrame."""
    records = []
    for line in log_text.splitlines():
        parts = line.split()
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
            continue  # header row
    return pd.DataFrame(records)


def solve_with_diagnostics(solve_fn, *args, **kwargs): 
    kwargs.setdefault('verbose', 2)
    log_stream = _Tee(sys.stdout)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(log_stream):
        solution = solve_fn(*args, **kwargs)
    runtime_s = time.perf_counter() - t0

    iteration_log = parse_solve_bvp_log(log_stream.getvalue())

    run_diagnostics = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'success': solution.success,
        'status': solution.status,
        'message': solution.message,
        'n_iterations': solution.niter,
        'n_nodes_final': len(solution.x),
        'max_rms_residual': float(np.max(solution.rms_residuals)) if len(solution.rms_residuals) else np.nan,
        'runtime_s': runtime_s,
    }
    return solution, run_diagnostics, iteration_log


def save_diagnostics(run_diagnostics, iteration_log, output_dir='output'):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    summary_path = os.path.join(output_dir, 'run_log.csv')
    row = pd.DataFrame([run_diagnostics])
    if os.path.isfile(summary_path):
        row.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        row.to_csv(summary_path, index=False)

    if not iteration_log.empty:
        iteration_log.to_csv(os.path.join(output_dir, 'iteration_log.csv'), index=False)