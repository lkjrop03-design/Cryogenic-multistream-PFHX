# Batch runner for the generic (topology-driven) model.
# Runs every .yaml in tests/ and writes one Excel workbook.
import sys, os, glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from hydrogen_pfhx import model, outputs, utils

scenario_dir = os.path.join(os.path.dirname(__file__), 'tests')
run_specs = [{'label': os.path.splitext(os.path.basename(p))[0], 'config_path': p}
             for p in sorted(glob.glob(os.path.join(scenario_dir, '*.yaml')))]

runs = []
for spec in run_specs:
    print("\n###### Running '%s' (%s) ######" % (spec['label'], spec['config_path']))
    try:
        configuration = utils.load_config(spec['config_path'])
        results, run_diagnostics = model.model(configuration)
        runs.append({'label': spec['label'], 'results': results,
                     'run_diagnostics': run_diagnostics, 'error': None})
        print("Converged: %s, topology %s, %d nodes, %.2fs, energy imbalance %.2f%%"
              % (run_diagnostics['success'], run_diagnostics['topology'],
                 run_diagnostics['n_nodes_final'], run_diagnostics['runtime_s'],
                 run_diagnostics['energy_imbalance_pct']))
    except Exception as e:
        print("'%s' FAILED: %s" % (spec['label'], e))
        runs.append({'label': spec['label'], 'results': None,
                     'run_diagnostics': None, 'error': str(e)})

output_path = os.path.join(os.path.dirname(__file__), 'output', 'batch_results.xlsx')
outputs.save_results_excel(runs, output_path)
print("\nSaved %d run(s) to %s" % (len(runs), output_path))
