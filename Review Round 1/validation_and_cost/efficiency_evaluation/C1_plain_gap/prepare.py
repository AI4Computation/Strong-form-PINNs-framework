"""Prepare inputs, verify instrumentation, and freeze before requesting idle time."""
import argparse
import hashlib
import os
import subprocess
import sys
from common import HERE, ROOT, OLD, PROJECT, METHODS, read, write, sha, utc, verify_hashes

def source_files():
    return ([HERE / n for n in ['common.py', 'worker.py', 'prepare.py', 'supervisor.py']]
            + [OLD / 'code' / n for n in ['runtime.py', 'models.py', 'observed_lbfgs.py']]
            + [ROOT / 'code' / n for n in ['cost_runtime.py', 'run_cost_benchmark.py']]
            + [PROJECT / '代码源文件/审稿修改-基线对比/benchmark_suite.py'])

def inputs():
    path = HERE / 'inputs.json'
    if path.exists():
        return read(path)
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import torch
    def digest(state):
        h = hashlib.sha256()
        for key, value in state.items():
            a = value.detach().cpu().contiguous().numpy()
            for data in [key.encode(), str(a.dtype).encode(), str(a.shape).encode(), a.tobytes()]:
                h.update(data)
        return h.hexdigest()
    result = dict(created_utc=utc(), configurations={}, sample_sha256={},
                  initial_state_sha256={}, protected_inputs_sha256={})
    for seed in range(41, 49):
        for method in METHODS:
            identity = f'C1_{method}_s{seed}'
            folder = OLD / 'runs' / identity
            prior = read(folder / 'result.json')
            config = prior['configuration']
            assert config['case'] == 'C1' and config['max_iter'] == 2000
            assert config['method'] == method and config['seed'] == seed
            assert config['geometry'] == 'circle' and config['p_top'] == -5 and config['p_lateral'] == -1
            result['configurations'][identity] = config
            result['sample_sha256'][identity] = prior['collocation_hashes']
            initial = torch.load(folder / 'step_00000.pt', map_location='cpu', weights_only=True)['state_dict']
            result['initial_state_sha256'][identity] = digest(initial)
            for file in [folder / 'result.json', folder / 'step_00000.pt']:
                result['protected_inputs_sha256'][str(file)] = sha(file)
    for file in [OLD / 'config/geometry.json']:
        result['protected_inputs_sha256'][str(file)] = sha(file)
    write(path, result)
    return result

def checks():
    data = inputs()
    assert not (HERE / 'protocol.json').exists(), 'Cannot rerun preparation after freeze.'
    sources = {str(p): sha(p) for p in source_files() if p.exists()}
    for method in METHODS:
        for variant in ['stock', 'bare', 'instrumented']:
            folder = HERE / 'checks' / f'C1_{method}_s42_{variant}'
            if (folder / 'result.json').exists():
                continue
            assert not folder.exists(), 'Partial check needs review.'
            folder.mkdir(parents=True)
            print(f'DEVELOPMENT CHECK {method} {variant}; not formal timing', flush=True)
            with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-B', '-X', 'utf8', str(HERE / 'worker.py'),
                                '--method', method, '--seed', '42', '--variant', variant, '--development'],
                               check=True, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    verification = []
    protected = dict(data['protected_inputs_sha256'])
    for method in METHODS:
        results = {v: read(HERE / 'checks' / f'C1_{method}_s42_{v}' / 'result.json')
                   for v in ['stock', 'bare', 'instrumented']}
        for left, right in [('stock', 'bare'), ('bare', 'instrumented')]:
            a, b = results[left], results[right]
            for field in ['initial_state_sha256', 'endpoint_state_sha256', 'endpoint_gradient_sha256',
                          'sample_sha256', 'closure_evaluations', 'closure_losses', 'optimizer_iterations']:
                verification.append(dict(method=method, comparison=f'{left}/{right}',
                                         field=field, passed=a[field] == b[field]))
        a, b = results['bare'], results['instrumented']
        for field in ['accepted_steps', 'stop_reason', 'accepted_history']:
            verification.append(dict(method=method, comparison='bare/instrumented',
                                     field=field, passed=a[field] == b[field]))
        verification.append(dict(method=method, field='0_50_100_states_bitwise',
            passed=[r['state_sha256'] for r in a['observations']] == [r['state_sha256'] for r in b['observations']]))
        verification.append(dict(method=method, field='100_steps_completed', passed=b['accepted_steps'] == 100))
        verification.append(dict(method=method, field='clock_boundaries_ordered',
            passed=all(0 <= r['optimization_s'] <= r['state_elapsed_s'] <= r['available_elapsed_s']
                       <= b['observed_solution_s'] for r in b['observations'])))
        for obs in b['observations']:
            file = HERE / 'checks' / f'C1_{method}_s42_instrumented' / obs['checkpoint']
            verification.append(dict(method=method, field=obs['checkpoint']+'_hash',
                                     passed=sha(file) == obs['checkpoint_sha256']))
    for file in (HERE / 'checks').rglob('*'):
        if file.is_file():
            protected[str(file)] = sha(file)
    verify_hashes(sources)
    verify_hashes(data['protected_inputs_sha256'])
    result = dict(passed=all(r['passed'] for r in verification), checks=verification,
                  development_runs=9, formal_timing=False, fem_evaluations=0, completed_utc=utc(),
                  source_sha256=sources, protected_inputs_sha256=protected)
    write(HERE / 'instrumentation_verification.json', result)
    print(f'CHECKS {sum(r["passed"] for r in verification)}/{len(verification)}', flush=True)
    assert result['passed'], 'Instrumentation changed numerical results; do not freeze or time.'

def freeze():
    assert not (HERE / 'protocol.json').exists(), 'Frozen protocol cannot be overwritten.'
    v = read(HERE / 'instrumentation_verification.json')
    assert v['passed']
    verify_hashes(v['source_sha256'])
    verify_hashes(v['protected_inputs_sha256'])
    order = []
    # Six permutations, then first two: pairwise order imbalance at most two runs.
    import itertools
    permutations = list(itertools.permutations(METHODS))
    for i, seed in enumerate(range(41, 49)):
        for method in permutations[i % 6]:
            order.append(dict(identity=f'C1_{method}_s{seed}', method=method, seed=seed))
    old_rules = read(ROOT / 'efficiency_evaluation/C1_reference/protocol.json')['interference_rules']
    protocol = dict(frozen_utc=utc(), purpose='R2.5: matched plain versus anchored and strong half-Fourier C1 costs',
        scope='Existing eight paired development seeds; no independent geometry or new-method validation',
        runs=24, run_order=order, parameter_counts=dict(vanilla_matched=110913, anchored=110705, fourier_half=110705),
        science='Frozen original configuration, initialization, collocation, objective, all closures, L-BFGS strong Wolfe; 2000 accepted-step maximum, default 2500 closure cap, history 50, gradient tolerance 1e-7, change tolerance 1e-12.',
        precision='One GPU, float32, TF32 off, deterministic algorithms, four Torch CPU threads, fresh worker per run',
        observation_steps='0, every 50 accepted steps, termination; portable states saved with actual timestamps',
        fem_policy='Timed worker blocks FEM file access. Freeze every timed checkpoint in the whole cohort before post-evaluation; no reference-directed stopping or reruns.',
        accuracy_targets_pct=[5, 2, 1],
        target_weightings=['joint point-weighted vector displacement and stress',
                           'joint area-weighted vector displacement and stress at FEM integration points'],
        statistic='Report first observed success and adjacent observed-state time bracket, three-consecutive-observation confirmation, endpoint and NR counts. A bracket is an observation bracket, not proof against earlier unobserved crossings. No linear time interpolation; no mean-time claim ignoring failures.',
        clock_boundaries=dict(net_optimization='Synchronized optimizer entry to return minus measured observer logging/state export time; includes all closures and line searches.',
            complete_observed='Before model construction through final portable model save; includes sampling, optimizer construction, input hash audits, all state observations and logging.',
            cold_total='Script entry to solution save includes imports, uniform CUDA warmup and preparation. Supervisor launch-to-exit additionally includes queries and postsolve audit/logging; report separately.',
            queries='Fixed geometry-only 4096 float32 coordinates; all five native mixed outputs; 10 warmup batches, 50 device-resident and 50 host-GPU-host batches after solution clock closes.'),
        resources='Solver-window CUDA allocated/reserved allocator peaks before query; lifetime RSS labelled lifetime; observer RSS labelled sampled. No FEM arrays held. Counters are not whole-device memory or a transient phase RSS guarantee.',
        interference_rules=old_rules,
        idle_requirement='Explicit fresh author idle confirmation, bound to this frozen protocol, no more than six hours old; single continuous window, no silent selective retry. Stop and retain flagged attempts before next run.',
        source_sha256={str(p): sha(p) for p in source_files()},
        protected_inputs_sha256={**v['protected_inputs_sha256'], str(HERE/'inputs.json'):sha(HERE/'inputs.json'),
                                 str(HERE/'instrumentation_verification.json'):sha(HERE/'instrumentation_verification.json')})
    write(HERE / 'protocol.json', protocol)
    write(HERE / 'manifest.json', dict(status='awaiting_fresh_idle_confirmation', active=None,
          total_runs=24, completed=[], flagged=[], protocol_sha256=sha(HERE/'protocol.json'), updated_utc=utc()))
    print('PROTOCOL_FROZEN; formal timing not started; fresh idle confirmation required.', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inputs', 'checks', 'freeze'])
    args = parser.parse_args()
    globals()[args.action]()
