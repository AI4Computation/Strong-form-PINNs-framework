"""Fresh initialization, controlled baseline training, then separate FEM evaluation."""
from common import *
from models import build_model, samples_for, objective, legacy
from observed_lbfgs import ObservedLBFGS
import metrics

def build(c, device):
    legacy.DEVICE = torch.device(device)
    method = c['method']
    model = build_model('fourier' if method=='fourier_quarter' else method, c['seed'], c['geometry'])
    if method == 'fourier_quarter':
        model.B.mul_(.25)
    return model.to(device)

def train(c, out, device):
    out.mkdir(parents=True, exist_ok=False)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    synchronize(device); start = time.perf_counter()
    model = build(c, device)
    samples = samples_for(c['seed'], c['geometry'])
    trace = []; closures = 0
    def observer(step, evaluations, loss, gradient):
        if not np.isfinite([loss,gradient]).all():
            raise FloatingPointError('Nonfinite accepted state.')
        trace.append(dict(accepted_step=step, closure_evaluations=evaluations, loss=loss, max_gradient=gradient))
        if step % 50 == 0 or step == 10:
            save(model, out/f'step_{step:04d}.pt', step)
    opt = ObservedLBFGS(model.parameters(), observer=observer, max_iter=c['max_iter'],
                        history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-12)
    def closure():
        nonlocal closures
        opt.zero_grad(set_to_none=True)
        loss = objective(model, samples, c)
        finite_loss_and_gradients(loss, model)
        closures += 1
        return loss
    opt.step(closure)
    save(model, out/'terminal.pt', opt.accepted_steps)
    synchronize(device)
    write(out/'trace.json', trace)
    write(out/'result.json', dict(configuration=c, accepted_steps=opt.accepted_steps,
        closure_evaluations=closures, stop_reason=opt.stop_reason,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        fixed_coefficients=sum(p.numel() for p in model.buffers()),
        elapsed_including_preparation_and_checkpoint_io_seconds=time.perf_counter()-start,
        fem_during_training=False, environment=resources(device)))

def evaluate(c, out, device, all_checkpoints=False, save_fields=False):
    record = read(out/'result.json')
    model = build(c, device)
    ref = load_npz(ROOT/f'data/fem/benchmark/references/{c["case"]}.npz')
    paths = sorted(out.glob('step_*.pt')) if all_checkpoints else []
    paths.append(out/'terminal.pt')
    rows = []
    for path in paths:
        steps = load_weights(model, path, device); model.eval()
        values = metrics.evaluate(model, ref, c['geometry'])
        row = dict(checkpoint=path.name, actual_steps=steps, metrics=values)
        if 'volume' in ref:
            s = metrics.predict(model, ref['xy_s'], 's', c['geometry']); w = ref['volume'].reshape(-1)
            if len(w) == len(s):
                row['stress_area_relative_l2_percent'] = float(100*np.sqrt((w[:,None]*(s-ref['s'])**2).sum()/(w[:,None]*ref['s']**2).sum()))
        rows.append(row)
        if save_fields:
            np.savez_compressed(out/(path.stem+'_fields.npz'),
                u=metrics.predict(model,ref['xy_u'],'u',c['geometry']),
                s=metrics.predict(model,ref['xy_s'],'s',c['geometry']))
    write(out/'evaluation.json',dict(configuration=c,results=rows))
    print(json.dumps(rows[-1]), flush=True)
