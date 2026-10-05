"""Complete-cohort time-to-accuracy and resource analysis; no model selection."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'efficiency_evaluation' / 'C1_reference'
METHODS = ['anchored', 'fourier', 'fourier_half', 'independent_marginal']
METRICS = {'displacement': ['u_vector_pct'], 'stress': ['s_vector_pct'],
           'both': ['u_vector_pct', 's_vector_pct'], 'area_stress': ['s_area_pct'],
           'u_and_area_stress': ['u_vector_pct', 's_area_pct']}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def summary(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(n=0, mean=None, sd=None, median=None, minimum=None, maximum=None)
    return dict(n=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values)>1 else None,
                median=float(np.median(values)), minimum=float(values.min()), maximum=float(values.max()))


def crossing(trace, keys, target):
    passed = [all(row[k] is not None and row[k] <= target for k in keys) for row in trace]
    first = next((i for i, ok in enumerate(passed) if ok), None)
    stable = next((i for i in range(len(passed)-2) if all(passed[i:i+3])), None)
    result = dict(reached=first is not None, first=None, consecutive_three=None,
                  terminal_reached=passed[-1], pass_to_fail_transitions=sum(a and not b for a,b in zip(passed,passed[1:])),
                  final_observed_solver_s=trace[-1]['solver_s'])
    if first is not None:
        row = trace[first]; previous = trace[first-1] if first else None
        result['first'] = dict(accepted_step=row['accepted_step'],
            previous_observed_step=previous['accepted_step'] if previous else None,
            solver_lower_s=previous['solver_s'] if previous else 0., solver_upper_s=row['solver_s'],
            state_elapsed_lower_s=previous['state_elapsed_s'] if previous else 0.,
            state_elapsed_upper_s=row['state_elapsed_s'], available_elapsed_s=row['available_elapsed_s'],
            solver_cpu_s=row['solver_cpu_s'],
            solver_peak_cuda_allocated_MiB=row['solver_peak_cuda_allocated_MiB'],
            solver_peak_cuda_reserved_MiB=row['solver_peak_cuda_reserved_MiB'])
    if stable is not None:
        start, confirm = trace[stable], trace[stable+2]
        result['consecutive_three'] = dict(start_step=start['accepted_step'], confirmation_step=confirm['accepted_step'],
            start_solver_s=start['solver_s'], confirmation_solver_s=confirm['solver_s'],
            confirmation_available_elapsed_s=confirm['available_elapsed_s'])
    return result


def check_crossing_logic():
    trace = [dict(x=x, accepted_step=i, solver_s=float(i), state_elapsed_s=float(i)+.1,
             available_elapsed_s=float(i)+.2, solver_cpu_s=float(i),
             solver_peak_cuda_allocated_MiB=1., solver_peak_cuda_reserved_MiB=2.)
             for i,x in enumerate([6.,4.,6.,4.,3.,2.])]
    c = crossing(trace, ['x'], 5.)
    assert c['first']['accepted_step'] == 1 and c['first']['solver_lower_s'] == 0
    assert c['consecutive_three']['confirmation_step'] == 5
    assert c['pass_to_fail_transitions'] == 1 and c['terminal_reached']
    c = crossing(trace, ['x'], 1.)
    assert not c['reached'] and c['first'] is None and c['consecutive_three'] is None
    assert crossing(trace[-2:], ['x'], 5.)['consecutive_three'] is None


def paired_ratio(x, y, rng):
    assert len(x) == len(y) == 8
    logratio = np.log(np.asarray(x) / np.asarray(y))
    indices = rng.integers(0, 8, (10000, 8))
    estimates = np.exp(logratio[indices].mean(1))
    return dict(n_pairs=8, geometric_mean=float(np.exp(logratio.mean())),
                bootstrap_percentile_95=[float(z) for z in np.quantile(estimates, [.025, .975])],
                numerator_faster_pairs=int((np.asarray(x)<np.asarray(y)).sum()))


def main():
    check_crossing_logic()
    p, m = read(OUT/'protocol.json'), read(OUT/'manifest.json')
    assert m['status'] == 'complete' and len(m['completed']) == 32 and not m['flagged']
    assert len(set(m['completed'])) == 32
    assert sha(OUT/'protocol.json') == m['protocol_sha256']
    for path,digest in {**p['worker_source_sha256'], **p['protected_inputs_sha256']}.items():
        assert sha(path) == digest, path
    records = {}; telemetry = {}; per_run = {}; audit = {}
    for entry in p['run_order']:
        identity = f"C1_{entry['method']}_s{entry['seed']}"
        path = OUT/'runs'/identity/'result.json'
        assert sha(path) == m['result_sha256'][identity]
        d = read(path); monitor = read(OUT/'telemetry'/f'{identity}_run.json')
        assert d['formal_timing'] and not d['preflight'] and not monitor['flagged']
        assert d['environment']['gpu_count'] == 1 and d['environment']['torch_threads'] == 4
        assert d['environment']['dtype'] == 'float32' and not d['environment']['tf32']
        assert d['trainable_parameters'] == 110705
        assert d['numerical_replay']['bitwise_initial_equal']
        assert d['accepted_steps'] == 2000 and d['stop_reason'] == 'max_iter'
        assert d['protocol_sha256'] == m['protocol_sha256']
        assert abs(d['solver_total_s']+d['common_diagnostics_s']-d['observed_total_s']) < 1e-8
        assert all(a['solver_s']<b['solver_s'] for a,b in zip(d['trace'],d['trace'][1:]))
        assert [r['accepted_step'] for r in d['trace']] == [0,10]+list(range(50,2001,50))
        assert d['trace'][-1]['solver_s'] <= d['solver_total_s']
        assert all(np.isfinite(row[key]) for row in d['trace'] for keys in METRICS.values() for key in keys)
        assert all(s['power']['ac_line_status'] == 1 for s in monitor['samples'])
        result = {}
        for name,keys in METRICS.items():
            result[name] = {str(t): crossing(d['trace'],keys,t) for t in [5,2,1]}
        per_run[identity] = result; records[identity] = d; telemetry[identity] = monitor
        audit[identity] = dict(clock_decomposition_passed=True, monotone_observations=True,
              all_targets_evaluated=True, interference_flagged=False, initial_and_sampling_match=True,
              historical_endpoint_bitwise_equal=d['numerical_replay']['bitwise_endpoint_equal'])
    for seed in range(41,49):
        hashes=[records[f'C1_{method}_s{seed}']['collocation_sha256'] for method in METHODS]
        assert all(h == hashes[0] for h in hashes), ('paired sampling',seed)
    order_positions={method:[sum(row['method']==method for row in p['run_order'][i::4])
                             for i in range(4)] for method in METHODS}
    assert all(counts == [2,2,2,2] for counts in order_positions.values())
    horizon = min(d['trace'][-1]['solver_s'] for d in records.values())
    groups = {}; resources = {}; rng=np.random.default_rng(20261222)
    for method in METHODS:
        ids=[f'C1_{method}_s{s}' for s in range(41,49)]
        values = [records[k] for k in ids]
        method_stats = {}
        for metric in METRICS:
            method_stats[metric] = {}
            for target in [5,2,1]:
                rows=[per_run[k][metric][str(target)] for k in ids]
                good=[r['first'] for r in rows if r['reached']]
                confirmed=[r['consecutive_three'] for r in rows if r['consecutive_three'] is not None]
                capped=[min(r['first']['solver_upper_s'], horizon) if r['reached'] else horizon for r in rows]
                method_stats[metric][str(target)] = dict(
                    reached=sum(r['reached'] for r in rows), total=8,
                    terminal_reached=sum(r['terminal_reached'] for r in rows),
                    three_observations_confirmed=sum(r['consecutive_three'] is not None for r in rows),
                    any_recrossing_runs=sum(r['pass_to_fail_transitions']>0 for r in rows),
                    successful_solver_lower_s=summary([r['solver_lower_s'] for r in good]),
                    successful_solver_upper_s=summary([r['solver_upper_s'] for r in good]),
                    observed_solver_bracket_width_s=summary([r['solver_upper_s']-r['solver_lower_s'] for r in good]),
                    successful_available_elapsed_s=summary([r['available_elapsed_s'] for r in good]),
                    successful_solver_cpu_s=summary([r['solver_cpu_s'] for r in good]),
                    successful_solver_peak_cuda_allocated_MiB=summary([r['solver_peak_cuda_allocated_MiB'] for r in good]),
                    successful_solver_peak_cuda_reserved_MiB=summary([r['solver_peak_cuda_reserved_MiB'] for r in good]),
                    confirmed_solver_s=summary([r['confirmation_solver_s'] for r in confirmed]),
                    confirmed_available_elapsed_s=summary([r['confirmation_available_elapsed_s'] for r in confirmed]),
                    restricted_horizon_solver_s=horizon,
                    observed_hits_by_horizon=sum(r['reached'] and r['first']['solver_upper_s']<=horizon for r in rows),
                    restricted_mean_observation_cost_s=float(np.mean(capped)),
                    restricted_cost_definition='Mean min(first observed successful state time, common horizon); NR contributes horizon without being labelled a success.')
        groups[method] = method_stats
        idle=[read(OUT/'telemetry'/f'{k}_idle.json') for k in ids]
        device_baseline=[float(np.median([s['gpu']['memory.used'] for s in samples[-3:]])) for samples in idle]
        device_peak=[max(s['gpu']['memory.used'] for s in telemetry[k]['samples']) for k in ids]
        resources[method] = dict(
            solver_total_s=summary([d['solver_total_s'] for d in values]),
            observed_total_s=summary([d['observed_total_s'] for d in values]),
            solver_cpu_s=summary([d['solver_cpu_s'] for d in values]),
            common_diagnostics_s=summary([d['common_diagnostics_s'] for d in values]),
            solver_cuda_allocated_MiB=summary([d['peak_by_phase']['solver']['cuda_allocated_MiB'] for d in values]),
            solver_cuda_reserved_MiB=summary([d['peak_by_phase']['solver']['cuda_reserved_MiB'] for d in values]),
            evaluation_cuda_allocated_MiB=summary([d['peak_by_phase']['evaluation']['cuda_allocated_MiB'] for d in values]),
            lifetime_peak_rss_MiB=summary([d['process_lifetime_memory']['peak_rss_lifetime_MiB'] for d in values]),
            lifetime_peak_private_commit_MiB=summary([d['process_lifetime_memory']['peak_private_commit_lifetime_MiB'] for d in values]),
            sampled_total_device_peak_MiB=summary(device_peak),
            sampled_total_device_increment_MiB=summary([p-b for p,b in zip(device_peak,device_baseline)]),
            sampled_device_scope='Whole-device memory sampled every3s, relative to the median last3 idle samples. Includes driver/context and concurrent desktop allocations; neither a per-process attribution nor a guaranteed transient peak.',
            sampled_solver_rss_MiB=summary([d['peak_by_phase']['solver']['sampled_rss_MiB'] for d in values]),
            reference_arrays_MiB=summary([d['held_reference_array_bytes']/2**20 for d in values]),
            construction_to_samples_s=summary([d['construction']['sampling_ready_s'] for d in values]),
            construction_to_optimizer_s=summary([d['construction']['optimizer_ready_s'] for d in values]),
            solution_save_s=summary([d['solution_save_s'] for d in values]),
            cold_import_s=summary([d['cold_start']['import_s'] for d in values]),
            cuda_and_warmup_s=summary([d['cold_start']['cuda_init_and_uniform_warmup_s'] for d in values]),
            reference_load_s=summary([d['cold_start']['reference_load_s'] for d in values]),
            terminal_u_pct=summary([d['metrics']['u_vector_pct'] for d in values]),
            terminal_s_pct=summary([d['metrics']['s_vector_pct'] for d in values]),
            terminal_area_s_pct=summary([d['metrics']['s_area_pct'] for d in values]),
            trainable_parameters=sorted(set(d['trainable_parameters'] for d in values)),
            fixed_coefficients=sorted(set(d['fixed_coefficients'] for d in values)))
    ratios={}
    for competitor in METHODS[1:]:
        for metric in ['both','u_and_area_stress']:
            for target in [5,2,1]:
                a=[per_run[f'C1_anchored_s{s}'][metric][str(target)] for s in range(41,49)]
                b=[per_run[f'C1_{competitor}_s{s}'][metric][str(target)] for s in range(41,49)]
                if all(x['reached'] and y['reached'] for x,y in zip(a,b)):
                    for clock in ['solver_upper_s','available_elapsed_s']:
                        value=paired_ratio([x['first'][clock] for x in a],[y['first'][clock] for y in b],rng)
                        ratios[f'anchored_over_{competitor}__{metric}__{target}__{clock}']=value
    monitor_rows=[s for t in telemetry.values() for s in t['samples']]
    monitoring=dict(sample_count=len(monitor_rows), AC_power_all_samples=True, interference_flagged_runs=0,
        gpu_temperature_C=summary([s['gpu']['temperature.gpu'] for s in monitor_rows]),
        maximum_external_gpu_engine_pct=max(s['engines']['max_external_engine_pct'] for s in monitor_rows),
        maximum_external_cpu_pct=max(s['external_cpu_pct'] for s in monitor_rows),
        external_gpu_samples_above20=sum(s['engines']['max_external_engine_pct']>20 for s in monitor_rows),
        external_cpu_samples_above20=sum(s['external_cpu_pct']>20 for s in monitor_rows),
        limitation='Sampling and the sustained-interference rule do not establish zero background work or exclude all short bursts.')
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(), complete=True, formal_runs=32,
        instrumentation_training_runs=8, methods=METHODS, targets_pct=[5,2,1],
        method_order_position_counts=order_positions,
        common_solver_horizon_s=horizon, groups=groups, resources=resources, per_run=per_run,
        paired_time_ratios=ratios, monitoring=monitoring, audit=audit,
        source_sha256=dict(protocol=sha(OUT/'protocol.json'), analysis_plan=sha(OUT/'analysis_plan.json'),
                          instrumentation_checks=sha(OUT/'instrumentation_checks.json')),
        statistical_scope='Eight development seeds on one laptop/GPU; timing variation includes optimization variation, not replicated timing noise at identical weights.',
        historical_accuracy_used_with_new_times=False, candidate_method_adopted=False)
    write(OUT/'analysis.json', output)
    write(OUT/'checks.json', dict(passed=True, records=32, protected_inputs=len(p['protected_inputs_sha256']),
          crossing_logic_checked=True, input_hashes_unchanged=True, all_measurements_unflagged=True,
          paired_sampling_equal=True, balanced_method_order=True, same_resource_configuration=True,
          analysis_sha256=sha(OUT/'analysis.json')))
    for method in METHODS:
        r=groups[method]['both']['5'];v=resources[method]
        print(json.dumps(dict(method=method, reached=r['reached'], median_solver=r['successful_solver_upper_s']['median'],
               median_observed=r['successful_available_elapsed_s']['median'], restricted=r['restricted_mean_observation_cost_s'],
               peak_cuda=v['solver_cuda_allocated_MiB']['mean'], peak_rss=v['lifetime_peak_rss_MiB']['mean']),ensure_ascii=False))
    print('C1_COST_ANALYSIS_COMPLETE')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
