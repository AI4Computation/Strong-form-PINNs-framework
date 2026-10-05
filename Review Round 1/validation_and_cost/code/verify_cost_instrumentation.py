"""Reference-solver controls and numerical checks for cost instrumentation."""
from pathlib import Path
import argparse
import subprocess
import json
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'efficiency_evaluation' / 'C1_reference'
METHODS = ['anchored', 'fourier', 'fourier_half', 'independent_marginal']


def reference(method):
    import benchmark_c1_cost as bench
    import train
    bench.warmup()
    bench.load_reference()
    config = bench.read(bench.OLD / 'runs' / f'C1_{method}_s41' / 'result.json')['configuration']
    config = {**config, 'id': str(OUT / 'preflight_controls' / f'original_{method}_s41'), 'max_iter': 100}
    train.run_one(config)


def check_all():
    # Every reference and instrumented run uses a fresh process; no GPU overlap.
    for method in METHODS:
        identity = f'C1_{method}_s41'
        path = OUT / 'preflight_controls' / f'original_{method}_s41' / 'result.json'
        if not path.exists():
            subprocess.run([sys.executable, str(Path(__file__)), '--reference', method], check=True)
        path = OUT / 'preflight' / identity / 'result.json'
        if not path.exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name('benchmark_c1_cost.py')),
                            '--method', method, '--seed', '41', '--preflight'], check=True)
    import benchmark_c1_cost as bench
    torch = bench.torch
    checks = {}
    for method in METHODS:
        identity = f'C1_{method}_s41'
        pf = OUT / 'preflight' / identity
        old = OUT / 'preflight_controls' / f'original_{method}_s41'
        a, b = bench.read(pf / 'result.json'), bench.read(old / 'result.json')
        x = torch.load(pf / 'model.pt', map_location='cpu', weights_only=True)['state_dict']
        y = torch.load(old / 'step_00100.pt', map_location='cpu', weights_only=True)['state_dict']
        bitwise = all(torch.equal(x[k], y[k]) for k in x)
        common = {r['accepted_step']: r for r in b['metric_trace']}
        trace_diff = max(abs(row[key] - common[row['accepted_step']][key])
                         for row in a['trace'] for key in ['u_vector_pct', 's_vector_pct'])
        loss_equal = a['closure_losses'] == b['closure_losses']
        clock_identity = abs(a['solver_total_s'] + a['common_diagnostics_s'] - a['observed_total_s'])
        monotone = all(x['solver_s'] < y['solver_s'] and x['state_elapsed_s'] < y['state_elapsed_s']
                       for x, y in zip(a['trace'], a['trace'][1:]))
        initial = torch.load(bench.OLD / 'runs' / identity / 'step_00000.pt', map_location='cpu', weights_only=True)['state_dict']
        initial_equal = bench.state_digest(initial) == a['observed_state_sha256']['0']
        checks[method] = dict(endpoint_bitwise_equal_to_current_original=bitwise,
                             initial_bitwise_equal_to_historical=initial_equal,
                             all_closure_losses_equal=loss_equal, max_metric_difference_pp=trace_diff,
                             clock_decomposition_error_s=clock_identity, clock_monotone=monotone,
                             reference_sha256=bench.sha(old / 'result.json'),
                             instrumented_sha256=bench.sha(pf / 'result.json'))
        assert bitwise and initial_equal and loss_equal and trace_diff == 0 and clock_identity < 1e-9 and monotone, checks[method]
    bench.write(OUT / 'instrumentation_checks.json', dict(
        checked_utc=bench.utc(), passed=True, methods=checks,
        historical_trajectory_note='The first strict historical-endpoint check failed at seed 41. All current unchanged-solver controls and instrumented runs agree bitwise. Fresh-process warmup differs from the historical multi-run execution history, so historical accuracy is not used with new timings.',
        preflight_training_runs=8, accepted_steps_per_run=100,
        production_runs_started=0, original_scientific_sources_edited=False))
    print('COST_INSTRUMENTATION_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--reference', choices=METHODS)
    args = p.parse_args()
    if args.reference:
        reference(args.reference)
    else:
        check_all()
