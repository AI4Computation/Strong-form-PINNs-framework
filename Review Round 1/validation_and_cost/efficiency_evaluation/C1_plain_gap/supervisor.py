"""One fresh worker at a time; stop on interruption or predeclared interference."""
import os
import sys
import subprocess
import time
import msvcrt
from common import HERE, ROOT, read, write, sha, utc, verify_hashes, consent
sys.path.insert(0, str(ROOT / 'code'))
from cost_runtime import GpuEngines, ac_power, system_times, process_cpu
from run_cost_benchmark import gpu_status, cpu_use

def live(manifest, status, active=None):
    manifest.update(status=status, active=active, updated_utc=utc())
    write(HERE / 'manifest.json', manifest)
    p = read(ROOT / '进度.json')
    p.update(updated_utc=utc(), milestone='C1_plain_gap_' + status,
             current_jobs=[] if active is None else [active],
             plain_gap_formal_completed=len(manifest['completed']), plain_gap_status=status)
    write(ROOT / '进度.json', p)

def idle(engines, rules, identity):
    start = time.perf_counter()
    before = system_times()
    rows, good = [], 0
    while time.perf_counter() - start < rules['pre_run_max_wait_s']:
        if (HERE / 'STOP_REQUESTED').exists():
            break
        time.sleep(rules['pre_run_sample_interval_s'])
        after = system_times()
        row = dict(utc=utc(), elapsed_s=time.perf_counter()-start, gpu=gpu_status(),
                   system_cpu_pct=cpu_use(before, after), engines=engines.sample(), power=ac_power())
        before = after
        rows.append(row)
        acceptable = (row['gpu']['utilization.gpu'] is not None
            and row['gpu']['temperature.gpu'] is not None
            and row['gpu']['utilization.gpu'] <= rules['pre_run_gpu_utilization_pct_max']
            and row['gpu']['temperature.gpu'] <= rules['pre_run_temperature_C_max']
            and row['system_cpu_pct'] <= rules['pre_run_system_cpu_pct_max']
            and row['engines']['max_external_engine_pct'] <= rules['pre_run_external_gpu_engine_pct_max']
            and row['power']['ac_line_status'] == 1)
        good = good + 1 if acceptable else 0
        if good >= rules['pre_run_consecutive_good_samples'] and row['elapsed_s'] >= rules['pre_run_minimum_cooldown_s']:
            write(HERE / 'telemetry' / f'{identity}_idle.json', rows)
            return True
    write(HERE / 'telemetry' / f'{identity}_idle.json', rows)
    return False

def main():
    # OS lock releases after crash; the retained manifest still requires review of partial runs.
    with (HERE / 'supervisor.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        batch()

def batch():
    p = read(HERE / 'protocol.json')
    m = read(HERE / 'manifest.json')
    assert sha(HERE / 'protocol.json') == m['protocol_sha256']
    consent()
    verify_hashes(p['source_sha256'])
    verify_hashes(p['protected_inputs_sha256'])
    assert not m['flagged'], 'Flagged attempts need review; never silently retry.'
    rules = p['interference_rules']
    engines = GpuEngines()
    process = None
    try:
        for rec in p['run_order']:
            identity = rec['identity']
            if identity in m['completed']:
                continue
            consent()
            if (HERE / 'STOP_REQUESTED').exists():
                live(m, 'stopped_by_request'); return
            live(m, 'waiting_for_idle', identity)
            if not idle(engines, rules, identity):
                live(m, 'idle_guard_stopped'); return
            folder = HERE / 'runs' / identity
            assert not folder.exists(), 'An unregistered or partial run requires review.'
            folder.mkdir(parents=True)
            live(m, 'running', identity)
            print(f'START {len(m["completed"])+1}/24 {identity}', flush=True)
            launched = time.perf_counter()
            before = system_times()
            samples, reasons, bad = [], [], 0
            with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(HERE/'worker.py'),
                    '--method', rec['method'], '--seed', str(rec['seed'])],
                    stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                own_before = process_cpu(process.pid) or 0.
                while True:
                    try:
                        process.wait(timeout=rules['during_run_sample_interval_s'])
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    after = system_times()
                    own = process_cpu(process.pid)
                    total = cpu_use(before, after)
                    own_pct = (0. if own is None else 100*max(0., own-own_before)
                               / (os.cpu_count()*(after['clock']-before['clock'])))
                    row = dict(utc=utc(), since_launch_s=time.perf_counter()-launched,
                        gpu=gpu_status(), system_cpu_pct=total, external_cpu_pct=max(0., total-own_pct),
                        engines=engines.sample(process.pid), power=ac_power())
                    samples.append(row)
                    competing = (row['external_cpu_pct'] > rules['during_run_external_cpu_pct_max']
                        or row['engines']['max_external_engine_pct'] > rules['during_run_external_gpu_engine_pct_max'])
                    bad = bad+1 if competing else 0
                    if bad >= rules['during_run_consecutive_bad_samples'] and 'sustained_external_activity' not in reasons:
                        reasons.append('sustained_external_activity')
                    if row['power']['ac_line_status'] != 1 and 'AC_power_changed' not in reasons:
                        reasons.append('AC_power_changed')
                    if row['since_launch_s'] > rules['process_wall_timeout_s']:
                        reasons.append('process_wall_timeout'); process.terminate()
                    if (HERE / 'STOP_REQUESTED').exists():
                        reasons.append('stop_requested'); process.terminate()
                    before = after
                    if own is not None:
                        own_before = own
                duration = time.perf_counter()-launched
            result_path = folder / 'result.json'
            if process.returncode != 0 or not result_path.exists():
                reasons.append('worker_failed_or_incomplete')
            record = dict(identity=identity, worker_pid=process.pid, exit_code=process.returncode,
                process_launch_to_exit_s=duration, flagged=bool(reasons), reasons=reasons,
                monitoring_interval_s=rules['during_run_sample_interval_s'], samples=samples)
            write(HERE / 'telemetry' / f'{identity}_run.json', record)
            if result_path.exists():
                m['completed'].append(identity)
                m.setdefault('result_sha256', {})[identity] = sha(result_path)
            process = None
            print(f'FINISH {len(m["completed"])}/24 {identity}; flagged={bool(reasons)}', flush=True)
            if reasons:
                m['flagged'].append(dict(identity=identity, reasons=reasons))
                live(m, 'review_required'); return
            live(m, 'between_runs')
        verify_hashes(p['source_sha256'])
        verify_hashes(p['protected_inputs_sha256'])
        files = {str(file): sha(file) for name in ['runs', 'telemetry']
                 for file in (HERE/name).rglob('*') if file.is_file()}
        write(HERE / 'cohort_freeze.json', dict(frozen_utc=utc(), runs=len(m['completed']),
            fem_evaluations=0, protocol_sha256=sha(HERE/'protocol.json'), files_sha256=files))
        live(m, 'timed_models_frozen_awaiting_postevaluation')
        print('24 TIMED MODELS FROZEN; post-evaluation may now start.', flush=True)
    except BaseException as e:
        if process is not None and process.poll() is None:
            process.terminate(); process.wait(timeout=30)
        m['flagged'].append(dict(identity=m.get('active'), reasons=['supervisor_exception'], error=repr(e)))
        live(m, 'review_required')
        raise
    finally:
        engines.close()

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
