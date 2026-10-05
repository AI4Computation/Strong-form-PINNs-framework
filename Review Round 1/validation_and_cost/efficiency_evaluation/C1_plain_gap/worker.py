"""Unchanged C1 mixed PINNs, with FEM-free time-stamped state observations."""
import time
SCRIPT_STARTED = time.perf_counter()
import sys
import os
import gc
import argparse
import hashlib
import platform
from common import HERE, ROOT, OLD, METHODS, read, write, sha, utc, verify_hashes, consent

# A reference-data access is an error in this process, including after training.
def deny_fem(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace('\\', '/').lower()
        if '/fem/' in name or name.endswith('/reference_selection.json'):
            raise RuntimeError('FEM data are forbidden in the timed worker: ' + name)
sys.addaudithook(deny_fem)
sys.path.insert(0, str(OLD / 'code'))
sys.path.insert(0, str(ROOT / 'code'))
from runtime import torch, sync
from models import build_model, samples_for, objective, array_hash
from observed_lbfgs import ObservedLBFGS
from cost_runtime import memory, ac_power
IMPORT_SECONDS = time.perf_counter() - SCRIPT_STARTED

def digest(model):
    h = hashlib.sha256()
    state = model.state_dict() if hasattr(model, 'state_dict') else model
    for key, value in state.items():
        a = value.detach().cpu().contiguous().numpy()
        for data in [key.encode(), str(a.dtype).encode(), str(a.shape).encode(), a.tobytes()]:
            h.update(data)
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
            x.grad = None
            w.grad = None
        sync()
    del x, w, z
    gc.collect()
    torch.cuda.empty_cache()
    sync()
    return time.perf_counter() - start

def query(model):
    # Geometry-only fixed polar sampling in a square with a circular hole.
    # The same 4096 points and all five native mixed outputs for every method.
    import numpy as np
    start = time.perf_counter()
    rng = np.random.default_rng(81042)
    candidates = rng.uniform(-.5, .5, (8192, 2))
    xy = candidates[np.linalg.norm(candidates, axis=1) > .1][:4096].astype('float32')
    assert len(xy) == 4096
    cpu = torch.from_numpy(xy)
    x = cpu.to('cuda')
    sync()
    setup_s = time.perf_counter() - start
    with torch.no_grad():
        for _ in range(10):
            y = model(x)
        sync()
        gpu_s, roundtrip_s = [], []
        for _ in range(50):
            sync(); t = time.perf_counter()
            y = model(x)
            sync(); gpu_s.append(time.perf_counter() - t)
        for _ in range(50):
            sync(); t = time.perf_counter()
            y_cpu = model(cpu.to('cuda')).cpu()
            sync(); roundtrip_s.append(time.perf_counter() - t)
    assert bool(torch.isfinite(y_cpu).all())
    return dict(points=4096, point_sha256=hashlib.sha256(xy.tobytes()).hexdigest(),
                output_columns=['u_x', 'u_y', 's_xx', 's_yy', 's_xy'],
                geometry_only_setup_s=setup_s, warmup_batches=10,
                device_resident_batch_s=gpu_s, host_to_device_to_host_batch_s=roundtrip_s,
                total_query_phase_s=time.perf_counter() - start)

def run(method, seed, variant, development):
    identity = f'C1_{method}_s{seed}'
    inputs = read(HERE / 'inputs.json')
    config = dict(inputs['configurations'][identity])
    protocol_hash = None
    if development:
        config['max_iter'] = 100
        folder = HERE / 'checks' / f'{identity}_{variant}'
    else:
        assert variant == 'instrumented'
        p = read(HERE / 'protocol.json')
        consent()
        verify_hashes(p['source_sha256'])
        assert identity in [r['identity'] for r in p['run_order']]
        protocol_hash = sha(HERE / 'protocol.json')
        folder = HERE / 'runs' / identity
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'result.json').exists(), 'Completed run cannot be overwritten.'
    assert not list(folder.glob('step_*.pt')), 'Partial run requires review, not overwrite.'
    power = ac_power()
    if not development:
        assert power['ac_line_status'] == 1
    warmup_s = warmup()
    gc.collect(); torch.cuda.empty_cache(); sync()
    torch.cuda.reset_peak_memory_stats()
    memory_before = memory()
    rows, observations, closures = [], [], []
    diagnostic_s = 0.
    last_loss = last_gradient = None
    start_utc = utc()
    start = time.perf_counter()
    model = build_model(method, seed, 'circle')
    sync(); model_ready_s = time.perf_counter() - start
    samples = samples_for(seed, 'circle')
    sync(); samples_ready_s = time.perf_counter() - start
    parameters = [p for p in model.parameters() if p.requires_grad]

    def observe(step, evals, loss, gradient, final=False):
        nonlocal diagnostic_s, last_loss, last_gradient
        sync(); entered = time.perf_counter()
        last_loss, last_gradient = loss, gradient
        row = dict(accepted_step=step, closure_evaluations=evals,
                   loss=loss, gradient_max=gradient)
        if not final:
            rows.append(row.copy())
        if step % 50 == 0 or final:
            row.update(state_elapsed_s=entered - start,
                       optimization_s=entered - optimization_start - diagnostic_s)
            row['state_sha256'] = digest(model)
            if variant == 'instrumented':
                name = f'step_{step:05d}.pt'
                torch.save(dict(configuration=config, accepted_step=step,
                                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}),
                           folder / name)
                row['checkpoint'] = name
                row['checkpoint_sha256'] = sha(folder / name)
                row['memory'] = memory()
                sync(); row['available_elapsed_s'] = time.perf_counter() - start
                observations.append(row)
                write(folder / 'progress.json', dict(identity=identity, latest=row, updated_utc=utc()))
            else:
                observations.append(row)
        diagnostic_s += time.perf_counter() - entered

    kwargs = dict(max_iter=config['max_iter'], history_size=50, line_search_fn='strong_wolfe',
                  tolerance_grad=1e-7, tolerance_change=1e-12)
    optimizer = (torch.optim.LBFGS(parameters, **kwargs) if variant == 'stock'
                 else ObservedLBFGS(parameters, observer=observe, **kwargs))
    sync(); optimizer_ready_s = time.perf_counter() - start
    audit_start = time.perf_counter()
    initial_digest = digest(model)
    sample_hashes = {k: array_hash(v) for k, v in samples.items()}
    assert sample_hashes == inputs['sample_sha256'][identity]
    assert initial_digest == inputs['initial_state_sha256'][identity]
    setup_audit_s = time.perf_counter() - audit_start

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, samples, config)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite physical loss')
        loss.backward()
        closures.append(float(loss.detach()))
        return loss

    sync(); optimization_start = time.perf_counter()
    optimizer.step(closure)
    sync(); stopped = time.perf_counter()
    optimization_s = stopped - optimization_start - diagnostic_s
    state = optimizer.state[parameters[0]]
    steps = getattr(optimizer, 'accepted_steps', None)
    if variant != 'stock' and (not observations or observations[-1]['accepted_step'] != steps):
        observe(steps, state['func_evals'], last_loss, last_gradient, final=True)
    save_start = time.perf_counter()
    torch.save(dict(configuration=config, accepted_step=steps,
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), folder / 'model.pt')
    sync(); delivered = time.perf_counter()
    gpu_peaks = dict(allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                     reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)
    solver_memory = memory()
    endpoint_digest = digest(model)
    gradient_digest = digest({str(i): p.grad for i, p in enumerate(parameters)})
    assert all(bool(torch.isfinite(p).all()) for p in parameters)
    assert state['func_evals'] == len(closures)
    query_result = query(model) if variant == 'instrumented' else None
    result = dict(status='complete', identity=identity, variant=variant,
                  formal_timing=not development, configuration=config, started_utc=start_utc,
                  completed_utc=utc(), protocol_sha256=protocol_hash,
                  import_s=IMPORT_SECONDS, uniform_cuda_warmup_s=warmup_s,
                  before_solver_process_elapsed_s=start - SCRIPT_STARTED,
                  script_entry_to_solution_s=delivered - SCRIPT_STARTED,
                  construction=dict(model_ready_s=model_ready_s, samples_ready_s=samples_ready_s,
                                    optimizer_ready_s=optimizer_ready_s, setup_audit_s=setup_audit_s),
                  optimization_s=optimization_s, observed_solution_s=delivered - start,
                  observation_and_logging_s=diagnostic_s, final_solution_save_s=delivered-save_start,
                  accepted_steps=steps, closure_evaluations=len(closures),
                  optimizer_iterations=state['n_iter'], max_eval=optimizer.param_groups[0]['max_eval'],
                  stop_reason=getattr(optimizer, 'stop_reason', 'stock_control'),
                  initial_state_sha256=initial_digest, endpoint_state_sha256=endpoint_digest,
                  endpoint_gradient_sha256=gradient_digest, sample_sha256=sample_hashes,
                  closure_losses=closures, accepted_history=rows, observations=observations,
                  gpu_solver_allocator_peaks=gpu_peaks, process_before_solver=memory_before,
                  process_memory_at_solution=solver_memory, query=query_result,
                  trainable_parameters=sum(p.numel() for p in parameters),
                  fixed_coefficients=sum(p.numel() for p in model.buffers()),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                   gpu_count=1, torch_threads=torch.get_num_threads(), dtype='float32',
                                   tf32=torch.backends.cuda.matmul.allow_tf32,
                                   deterministic=torch.are_deterministic_algorithms_enabled(),
                                   start_power=power, cpu_logical_count=os.cpu_count()),
                  fem_reference_access_forbidden=True)
    write(folder / 'result.json', result)
    print(f'{identity} {variant}: complete, steps={steps}, closures={len(closures)}', flush=True)

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--variant', choices=['stock', 'bare', 'instrumented'], default='instrumented')
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    try:
        run(args.method, args.seed, args.variant, args.development)
    except BaseException as e:
        folder = HERE / ('checks' if args.development else 'runs') / (
            f'C1_{args.method}_s{args.seed}' + (f'_{args.variant}' if args.development else ''))
        write(folder / 'failure.json', dict(error=repr(e), utc=utc(), formal_timing=not args.development))
        raise
