"""Serial benchmark supervisor with predeclared interference checks."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import csv
from cost_runtime import GpuEngines, ac_power, system_times, process_cpu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'efficiency_evaluation' / 'C1_reference'
OLD = ROOT.parent / 'controlled_pinn'
NO_WINDOW = subprocess.CREATE_NO_WINDOW


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gpu_status():
    names = ['utilization.gpu', 'memory.used', 'temperature.gpu', 'pstate',
             'clocks.current.sm', 'clocks.current.memory', 'power.draw']
    p = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(names), '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True, creationflags=NO_WINDOW)
    values = next(csv.reader([p.stdout.strip().splitlines()[0]]))
    result = {}
    for name, value in zip(names, values):
        value = value.strip()
        if name == 'pstate':
            result[name] = value
        else:
            try:
                result[name] = float(value)
            except ValueError:
                result[name] = None
    return result


def cpu_use(before, after):
    total = after['total'] - before['total']
    return max(0., 100 * (1 - (after['idle'] - before['idle']) / total)) if total > 0 else 0.


def prepare():
    if (OUT / 'protocol.json').exists():
        return read(OUT / 'protocol.json')
    checks = read(OUT / 'instrumentation_checks.json')
    assert checks['passed']
    design_path = ROOT / 'efficiency_evaluation' / 'design.json'
    design = read(design_path)
    sources = [ROOT / 'code' / n for n in ['cost_runtime.py', 'benchmark_c1_cost.py',
                                             'run_cost_benchmark.py', 'verify_cost_instrumentation.py']]
    sources += [OLD / 'code' / n for n in ['runtime.py', 'models.py', 'metrics.py', 'observed_lbfgs.py', 'train.py']]
    sources += [ROOT.parents[1] / '代码源文件/审稿修改-基线对比/benchmark_suite.py']
    old = read(ROOT / 'residual_localization' / 'manifest.json')
    protected = dict(old['inputs_sha256'])
    for p in [design_path, ROOT / 'efficiency_evaluation' / 'hardware.json',
              ROOT / '计算资源与达标时间评价方案.md',
              OLD / 'fem' / 'references' / 'C1.npz', OLD / 'fem' / 'tr3_circle_sq0p0025_UnitL.npz',
              OLD / 'config' / 'reference_selection.json', OUT / 'instrumentation_checks.json']:
        protected[str(p.resolve())] = sha(p)
    p = dict(frozen_utc=utc(), scope='Fresh-process C1 reference cost comparison. Eight paired development seeds; not independent geometry validation.',
        design_sha256=sha(design_path), methods=design['initial_reference_batch']['methods'],
        run_order=design['initial_reference_batch']['run_order'], runs=32,
        protocol='Same fixed models, initialization, samples, objective and installed observed L-BFGS as the prior C1 study. No scientific-source edits.',
        targets_pct=[5, 2, 1], observation_schedule='0, 10, every 50 accepted updates, termination',
        primary_metrics=['u_vector_pct', 's_vector_pct', 'simultaneous_u_s'],
        secondary_metrics=['s_area_pct', 'simultaneous_u_and_area_s'], consecutive_observations=3,
        random_seed_scope='Existing 41-48 paired seeds; cost and accuracy always taken from the same fresh run.',
        cold_start='Uniform generic matmul/tanh/sin/cos backward warmup; no target-model training. Imports, CUDA setup and FEM-reference loading reported separately.',
        solver_clock='Before model/feature construction and sample generation through final solution save, minus independently timed common evaluation and extra diagnostic logging. Initial optimizer construction, all trial closures and rejected line-search work included.',
        observed_clock='Same origin, includes common offline evaluation and diagnostic logging; threshold available time after the corresponding evaluation.',
        resource_clock='CUDA peaks reset at solver/evaluation phase boundaries and accumulated separately. Process-lifetime peak RAM includes held FEM arrays and diagnostics; phase RSS is only boundary-sampled.',
        system='Windows desktop remains running; sampled external GPU-engine and CPU load screen for competing activity, not proof of exclusive hardware.',
        interference_rules=dict(pre_run_minimum_cooldown_s=5, pre_run_consecutive_good_samples=3,
            pre_run_sample_interval_s=1, pre_run_max_wait_s=120,
            pre_run_gpu_utilization_pct_max=10, pre_run_temperature_C_max=65,
            pre_run_external_gpu_engine_pct_max=15, pre_run_system_cpu_pct_max=15,
            during_run_sample_interval_s=3, during_run_external_gpu_engine_pct_max=20,
            during_run_external_cpu_pct_max=20, during_run_consecutive_bad_samples=3,
            ac_power_required=True, process_wall_timeout_s=600,
            on_interference='Retain run and flag, then stop before next run; no outcome-based removal or automatic selective retry.'),
        user_notice='User notified in this turn before measurement; prior idle-test authorization retained.',
        historical_replay='Current original solver vs instrumented controls agree bitwise at 100 accepted steps for all four methods. Historical full trajectories may differ; their accuracy is not reused for new time comparisons.',
        worker_source_sha256={str(x.resolve()): sha(x) for x in sources},
        protected_inputs_sha256=protected)
    write(OUT / 'protocol.json', p)
    write(OUT / 'manifest.json', dict(status='prepared', protocol_sha256=sha(OUT / 'protocol.json'),
          created_utc=utc(), completed=[], flagged=[], active=None, total_runs=32))
    return p


def update_live(m, phase, identity=None):
    m.update(status=phase, active=identity, updated_utc=utc())
    write(OUT / 'manifest.json', m)
    state = read(ROOT / '进度.json')
    state.update(updated_utc=utc(), milestone='C1_cost_benchmark_' + phase,
        current_jobs=[] if identity is None else [dict(identity=identity, stage=phase)],
        completed_formal_efficiency_replays=len(m['completed']),
        efficiency_measurement_status=phase, new_timing_comparison_performed=bool(m['completed']),
        this_round_new_training_runs=len(m['completed']),
        this_round_instrumentation_check_runs=8)
    write(ROOT / '进度.json', state)


def wait_idle(engines, rules, identity):
    start = time.perf_counter()
    before = system_times(); good = 0; rows = []
    while time.perf_counter() - start < rules['pre_run_max_wait_s']:
        time.sleep(rules['pre_run_sample_interval_s'])
        after = system_times(); cpu = cpu_use(before, after); before = after
        row = dict(utc=utc(), elapsed_s=time.perf_counter() - start, gpu=gpu_status(),
                   system_cpu_pct=cpu, engines=engines.sample(), power=ac_power())
        rows.append(row)
        acceptable = (row['gpu']['utilization.gpu'] <= rules['pre_run_gpu_utilization_pct_max']
            and row['gpu']['temperature.gpu'] <= rules['pre_run_temperature_C_max']
            and cpu <= rules['pre_run_system_cpu_pct_max']
            and row['engines']['max_external_engine_pct'] <= rules['pre_run_external_gpu_engine_pct_max']
            and row['power']['ac_line_status'] == 1)
        good = good + 1 if acceptable else 0
        if good >= rules['pre_run_consecutive_good_samples'] and row['elapsed_s'] >= rules['pre_run_minimum_cooldown_s']:
            write(OUT / 'telemetry' / f'{identity}_idle.json', rows)
            return True
    write(OUT / 'telemetry' / f'{identity}_idle.json', rows)
    return False


def run_batch():
    p = prepare(); m = read(OUT / 'manifest.json')
    assert sha(OUT / 'protocol.json') == m['protocol_sha256']
    for path, digest in {**p['worker_source_sha256'], **p['protected_inputs_sha256']}.items():
        assert sha(path) == digest, path
    assert not m['flagged'], 'Flagged run requires explicit audit, not silent retry.'
    rules = p['interference_rules']
    engines = GpuEngines()
    try:
        for rec in p['run_order']:
            identity = f"C1_{rec['method']}_s{rec['seed']}"
            if identity in m['completed']:
                continue
            if (OUT / 'STOP_REQUESTED').exists():
                update_live(m, 'stopped_by_request'); return
            update_live(m, 'waiting_for_idle', identity)
            if not wait_idle(engines, rules, identity):
                update_live(m, 'idle_guard_wait_expired', None)
                print('IDLE_GUARD_STOP', flush=True); return
            folder = OUT / 'runs' / identity
            folder.mkdir(parents=True, exist_ok=True)
            assert not (folder / 'result.json').exists(), 'Unregistered result requires review.'
            update_live(m, 'running', identity)
            print(f'START {len(m["completed"])+1}/32 {identity}', flush=True)
            launched = time.perf_counter(); system_before = system_times()
            with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, str(ROOT / 'code' / 'benchmark_c1_cost.py'),
                    '--method', rec['method'], '--seed', str(rec['seed'])], stdout=log,
                    stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
                cpu_before = process_cpu(process.pid) or 0.
                samples = []; bad = 0; flagged = False; reason = []
                while process.poll() is None:
                    time.sleep(rules['during_run_sample_interval_s'])
                    system_after = system_times(); own_cpu = process_cpu(process.pid)
                    total_cpu = cpu_use(system_before, system_after)
                    dt = system_after['clock'] - system_before['clock']
                    own_pct = 0. if own_cpu is None else 100 * max(0, own_cpu-cpu_before) / (os.cpu_count() * dt)
                    external_cpu = max(0., total_cpu - own_pct)
                    row = dict(utc=utc(), since_launch_s=time.perf_counter()-launched, gpu=gpu_status(),
                        system_cpu_pct=total_cpu, external_cpu_pct=external_cpu,
                        engines=engines.sample(process.pid), power=ac_power())
                    samples.append(row)
                    competing = (row['engines']['max_external_engine_pct'] > rules['during_run_external_gpu_engine_pct_max']
                                 or external_cpu > rules['during_run_external_cpu_pct_max'])
                    bad = bad + 1 if competing else 0
                    if bad >= rules['during_run_consecutive_bad_samples']:
                        flagged = True
                        if 'sustained_external_activity' not in reason:
                            reason.append('sustained_external_activity')
                    if row['power']['ac_line_status'] != 1:
                        flagged = True
                        if 'AC_power_changed' not in reason:
                            reason.append('AC_power_changed')
                    if row['since_launch_s'] > rules['process_wall_timeout_s']:
                        flagged = True; reason.append('process_wall_timeout')
                        process.terminate()
                    if (OUT / 'STOP_REQUESTED').exists():
                        flagged = True; reason.append('user_stop_requested'); process.terminate()
                    system_before = system_after
                    if own_cpu is not None:
                        cpu_before = own_cpu
                process.wait()
            result_path = folder / 'result.json'
            if process.returncode != 0 or not result_path.exists():
                flagged = True; reason.append('worker_failed_or_incomplete')
            record = dict(identity=identity, worker_pid=process.pid, exit_code=process.returncode,
                process_launch_to_exit_s=time.perf_counter()-launched, flagged=flagged, reasons=reason,
                monitoring_interval_s=rules['during_run_sample_interval_s'], samples=samples)
            write(OUT / 'telemetry' / f'{identity}_run.json', record)
            if result_path.exists():
                m['completed'].append(identity)
                m.setdefault('result_sha256', {})[identity] = sha(result_path)
                result = read(result_path)
                print(json.dumps(dict(finished=len(m['completed']), identity=identity,
                      solver_s=result['solver_total_s'], S=result['metrics']['s_vector_pct'],
                      S_area=result['metrics']['s_area_pct'], flagged=flagged)), flush=True)
            if flagged:
                m['flagged'].append(dict(identity=identity, reasons=reason))
                update_live(m, 'review_required')
                print('BENCHMARK_STOP_FOR_REVIEW', flush=True); return
            update_live(m, 'between_runs')
        for path, digest in p['protected_inputs_sha256'].items():
            assert sha(path) == digest, path
        m['completed_utc'] = utc()
        update_live(m, 'complete')
        print('C1_COST_BENCHMARK_COMPLETE', flush=True)
    finally:
        engines.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        p = prepare()
        print(json.dumps(dict(prepared=True, runs=p['runs'], sources=len(p['worker_source_sha256']), protected=len(p['protected_inputs_sha256']))))
    else:
        run_batch()
