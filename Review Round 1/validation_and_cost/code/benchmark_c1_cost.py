"""Fresh-process C1 cost replay using unchanged scientific models and optimizer."""
import time
SCRIPT_STARTED = time.perf_counter()
import os
import sys
from pathlib import Path
import json
import hashlib
import platform
import argparse
import gc
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'controlled_pinn'
OUT = ROOT / 'efficiency_evaluation' / 'C1_reference'
sys.path.insert(0, str(OLD / 'code'))
import runtime
from runtime import torch, sync
from models import build_model, samples_for, objective, array_hash, legacy_path
from metrics import predict, field_metrics, relative
from observed_lbfgs import ObservedLBFGS
from cost_runtime import memory, ac_power, PhasePeaks
import numpy as np
IMPORT_SECONDS = time.perf_counter() - SCRIPT_STARTED


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, x):
    p = Path(p)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    state = model.state_dict() if hasattr(model, 'state_dict') else model
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def warmup():
    start = time.perf_counter()
    torch.cuda.init()
    with torch.random.fork_rng(devices=[0]):
        x = torch.full((4096, 64), .01, device='cuda', requires_grad=True)
        w = torch.full((64, 100), .02, device='cuda', requires_grad=True)
        for _ in range(8):
            z = x @ w
            (z.tanh().square() + z.sin().square() + z.cos().square()).mean().backward()
            x.grad = None; w.grad = None
        sync()
    del x, w, z
    gc.collect(); torch.cuda.empty_cache(); sync()
    return time.perf_counter() - start


def load_reference():
    approval = read(OLD / 'config' / 'reference_selection.json')
    assert approval['circle']['validated']
    with np.load(OLD / 'fem' / 'references' / 'C1.npz') as data:
        ref = {k: data[k] for k in ['xy_u', 'u', 'xy_s', 's']}
    with np.load(OLD / 'fem' / 'tr3_circle_sq0p0025_UnitL.npz') as data:
        assert np.array_equal(data['xy_s'], ref['xy_s'])
        weights = data['volume']
    radius = np.linalg.norm(ref['xy_s'], axis=1)
    near = (radius > .1) & (radius <= .2)
    return ref, weights, near


def evaluate(model, ref, weights, near):
    u = predict(model, ref['xy_u'], 'u', 'circle')
    s = predict(model, ref['xy_s'], 's', 'circle')
    result = {f'u_{k}': v for k, v in field_metrics(ref['u'], u, ['x', 'y']).items()}
    result.update({f's_{k}': v for k, v in field_metrics(ref['s'], s, ['xx', 'yy', 'xy']).items()})
    result['s_near_pct'] = relative(ref['s'][near], s[near])
    delta2 = np.sum((s - ref['s'])**2, axis=1)
    result['s_area_pct'] = float(100 * np.sqrt((weights @ delta2) / (weights @ np.sum(ref['s']**2, axis=1))))
    return result


def run(method, seed, preflight=False):
    identity = f'C1_{method}_s{seed}'
    config = read(OLD / 'runs' / identity / 'result.json')['configuration']
    assert config['method'] == method and config['seed'] == seed
    folder = OUT / ('preflight' if preflight else 'runs') / identity
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'result.json').exists(), 'A completed run cannot be overwritten.'
    if preflight:
        config = {**config, 'max_iter': 100}
        protocol_hash = None
    else:
        protocol = read(OUT / 'protocol.json')
        protocol_hash = sha(OUT / 'protocol.json')
        for path, digest in protocol['worker_source_sha256'].items():
            assert sha(path) == digest, path
        assert any(r['method'] == method and r['seed'] == seed for r in protocol['run_order'])
    power = ac_power()
    assert power['ac_line_status'] == 1, 'AC power is required for this laptop benchmark.'
    warmup_seconds = warmup()
    ref_start = time.perf_counter()
    ref, weights, near = load_reference()
    reference_load_seconds = time.perf_counter() - ref_start
    reference_array_bytes = int(sum(v.nbytes for v in ref.values()) + weights.nbytes + near.nbytes)
    gc.collect(); torch.cuda.empty_cache(); sync()
    before_memory = memory()
    torch.cuda.reset_peak_memory_stats()
    peaks = PhasePeaks(torch)
    trace = []; history = []; closures = []; state_hashes = {}
    diagnostic_seconds = 0.; diagnostic_cpu_seconds = 0.
    last_loss = None; last_grad = None
    started_utc = utc()
    start_cpu = time.process_time(); start = time.perf_counter()
    try:
        # The clock includes construction and sampling, unlike the prior logger.
        model = build_model(method, seed, 'circle')
        sync(); model_ready_seconds = time.perf_counter() - start
        samples = samples_for(seed, 'circle')
        sync(); sampled_seconds = time.perf_counter() - start
        parameters = [p for p in model.parameters() if p.requires_grad]

        def observe(step, evals, loss, gradient_max, final=False):
            nonlocal diagnostic_seconds, diagnostic_cpu_seconds, last_loss, last_grad
            sync(); entered = time.perf_counter(); entered_cpu = time.process_time()
            raw = entered - start
            row = dict(accepted_step=step, closure_evaluations=evals, loss=loss,
                       gradient_max=gradient_max, solver_s=raw - diagnostic_seconds,
                       solver_cpu_s=entered_cpu - start_cpu - diagnostic_cpu_seconds,
                       state_elapsed_s=raw)
            last_loss, last_grad = loss, gradient_max
            rss = peaks.capture('solver')
            row['solver_peak_cuda_allocated_MiB'] = peaks.peaks['solver']['cuda_allocated_MiB']
            row['solver_peak_cuda_reserved_MiB'] = peaks.peaks['solver']['cuda_reserved_MiB']
            if step % 50 == 0 or step == 10 or final:
                row.update(evaluate(model, ref, weights, near))
                sync(); row['available_elapsed_s'] = time.perf_counter() - start
                trace.append(row.copy())
            if step in [0, 10, 100, 500, 2000]:
                state_hashes[str(step)] = state_digest(model)
            if step % 100 == 0 or final:
                write(folder / 'progress.json', dict(status='running', identity=identity,
                      latest=row, power=ac_power(), updated_utc=utc()))
            peaks.capture('evaluation')
            sync()
            diagnostic_seconds += time.perf_counter() - entered
            diagnostic_cpu_seconds += time.process_time() - entered_cpu
            if history and history[-1]['accepted_step'] == step:
                history[-1].update(row)
            else:
                history.append(row)

        optimizer = ObservedLBFGS(parameters, observer=observe, max_iter=config['max_iter'],
                    history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-7,
                    tolerance_change=1e-12)
        sync(); optimizer_ready_seconds = time.perf_counter() - start

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, samples, config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite loss')
            loss.backward(); closures.append(float(loss.detach()))
            return loss

        optimizer.step(closure)
        sync(); optimizer_stop_elapsed = time.perf_counter() - start
        state = optimizer.state[parameters[0]]
        steps = optimizer.accepted_steps
        if trace[-1]['accepted_step'] != steps:
            observe(steps, state['func_evals'], last_loss, last_grad, final=True)
        # One final portable solution is part of the solver's delivered output.
        save_start = time.perf_counter()
        torch.save(dict(configuration=config, accepted_step=steps,
                        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), folder / 'model.pt')
        sync()
        solution_save_seconds = time.perf_counter() - save_start
        elapsed = time.perf_counter() - start
        total_solver_seconds = elapsed - diagnostic_seconds
        solver_cpu_seconds = time.process_time() - start_cpu - diagnostic_cpu_seconds
        lifetime_memory = peaks.capture('solver')
        sample_hashes = {k: array_hash(v) for k, v in samples.items()}
        prior = read(OLD / 'runs' / identity / 'result.json')
        assert sample_hashes == prior['collocation_hashes'], 'Sampling changed.'
        # Audits occur after the timed window; never used to alter the run.
        expected_path = OLD / 'runs' / identity / f'step_{steps:05d}.pt'
        expected = torch.load(expected_path, map_location='cpu', weights_only=True)['state_dict']
        actual = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        bitwise = all(torch.equal(actual[k], expected[k]) for k in expected)
        parameter_difference = max(float((actual[k] - expected[k]).abs().max()) for k in expected)
        previous_trace = {r['accepted_step']: r for r in prior['metric_trace']}
        metric_difference = max(abs(r[k] - previous_trace[r['accepted_step']][k]) for r in trace
                                for k in ['u_vector_pct', 's_vector_pct'])
        initial = torch.load(OLD / 'runs' / identity / 'step_00000.pt', map_location='cpu', weights_only=True)['state_dict']
        initial_equal = state_digest(initial) == state_hashes['0']
        assert initial_equal, 'Initial parameters or fixed features changed.'
        payload = dict(status='complete', identity=identity, configuration=config, preflight=preflight,
            started_utc=started_utc, completed_utc=utc(), protocol_sha256=protocol_hash,
            environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), gpu_count=1, dtype='float32',
                torch_threads=torch.get_num_threads(), mkl_threading=os.environ.get('MKL_THREADING_LAYER'),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                tf32=torch.backends.cuda.matmul.allow_tf32, power_at_start=power),
            cold_start=dict(import_s=IMPORT_SECONDS, cuda_init_and_uniform_warmup_s=warmup_seconds,
                            reference_load_s=reference_load_seconds),
            construction=dict(model_ready_s=model_ready_seconds, sampling_ready_s=sampled_seconds,
                              optimizer_ready_s=optimizer_ready_seconds),
            accepted_steps=steps, closure_evaluations=len(closures), stop_reason=optimizer.stop_reason,
            optimizer_attempted_iterations=state['n_iter'], max_eval=optimizer.param_groups[0]['max_eval'],
            solver_total_s=total_solver_seconds, solver_cpu_s=solver_cpu_seconds,
            observed_total_s=elapsed, optimizer_stop_elapsed_s=optimizer_stop_elapsed,
            common_diagnostics_s=diagnostic_seconds, common_diagnostics_cpu_s=diagnostic_cpu_seconds,
            solution_save_s=solution_save_seconds, peak_by_phase=peaks.peaks,
            process_lifetime_memory=lifetime_memory, process_memory_before_solver=before_memory,
            held_reference_array_bytes=reference_array_bytes,
            memory_scope='Lifetime RAM includes interpreter, CUDA context, held FEM arrays and evaluation. Phase RSS is sampled at observer boundaries, not a guaranteed transient maximum. CUDA peaks are allocator counters, not total-device usage. Reserved solver memory may retain caches from prior evaluation.',
            trainable_parameters=sum(p.numel() for p in parameters),
            fixed_coefficients=sum(p.numel() for p in model.buffers()),
            metrics={k: v for k, v in trace[-1].items() if k.startswith(('u_', 's_'))},
            trace=trace, accepted_history=history, closure_losses=closures,
            observed_state_sha256=state_hashes, collocation_sha256=sample_hashes,
            numerical_replay=dict(bitwise_initial_equal=initial_equal, bitwise_endpoint_equal=bitwise, max_parameter_abs_difference=parameter_difference,
                                  max_trace_primary_metric_difference_pp=metric_difference,
                                  interpretation='Compare logging against the unchanged solver under current fresh-process warmup using preflight_controls. Historical trajectories from a different execution history need not be bitwise identical; never combine old accuracy with new measured time.',
                                  old_result_sha256=sha(OLD / 'runs' / identity / 'result.json')),
            formal_timing=not preflight)
        write(folder / 'result.json', payload)
        write(folder / 'progress.json', dict(status='complete', identity=identity, accepted_steps=steps))
        print(json.dumps(dict(identity=identity, preflight=preflight, solver_s=total_solver_seconds,
              observed_s=elapsed, bitwise=bitwise, metric_diff=metric_difference,
              allocated_MiB=peaks.peaks['solver']['cuda_allocated_MiB']), ensure_ascii=False), flush=True)
    except BaseException as error:
        sync()
        write(folder / 'failure.json', dict(identity=identity, preflight=preflight,
              exception=repr(error), elapsed_since_solver_start_s=time.perf_counter() - start,
              completed_accepted_steps=history[-1]['accepted_step'] if history else None,
              trace=trace, closure_count=len(closures), failed_utc=utc()))
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['anchored', 'fourier', 'fourier_half', 'independent_marginal'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    run(args.method, args.seed, args.preflight)
