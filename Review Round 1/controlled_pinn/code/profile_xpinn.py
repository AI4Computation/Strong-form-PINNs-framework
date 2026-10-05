"""Instrument unchanged XPINN closure sections after formal training finishes.

The synchronized section timings describe this instrumented closure; they are
not subtractions of separately trained algorithms or end-to-end speedups.
"""
import runtime
from runtime import torch, sync, DEVICE
import models
from models import ROOT, build_model, samples_for, parts_loss, WEIGHTS
from summarize import write_csv
import inspect, json, time, hashlib
import numpy as np

def instrumented_function(section_end):
    source = inspect.getsource(parts_loss)
    transformed = source.replace('def parts_loss(', 'def profiled_parts_loss(', 1)
    replacements = [
        ("    ol,orr,ot,ob,oh=", "    section_end('domain_forward_and_spatial_derivatives')\n    ol,orr,ot,ob,oh="),
        ("    result={'equilibrium':eq", "    section_end('external_boundary_forward')\n    result={'equilibrium':eq"),
        ('    return result', "    section_end('interface_forward')\n    return result")]
    for old, new in replacements:
        assert transformed.count(old) == 1
        transformed = transformed.replace(old, new, 1)
    namespace = dict(vars(models)); namespace['section_end'] = section_end
    exec(compile(transformed, '<instrumented_parts_loss>', 'exec'), namespace)
    return namespace['profiled_parts_loss'], source

def main():
    records = []; checks = []
    for case in ['C1', 'C8']:
        folder = ROOT / 'runs' / f'{case}_xpinn_s42'
        result = json.loads((folder / 'result.json').read_text())
        c = result['configuration']; model = build_model('xpinn', 42)
        saved = torch.load(folder / f"step_{result['accepted_steps']:05d}.pt", map_location=DEVICE, weights_only=True)
        model.load_state_dict(saved['state_dict']); sample = samples_for(42, 'circle')
        sections = {}; last = 0.
        def section_end(name):
            nonlocal last
            sync(); current = time.perf_counter()
            sections[name] = current - last; last = current
        instrumented, source = instrumented_function(section_end)
        def calculate(function):
            nonlocal last, sections
            model.zero_grad(set_to_none=True); sections = {}; sync(); last = time.perf_counter()
            terms = function(model, sample, c['p_lateral'], c['p_top'])
            total = sum(WEIGHTS[k] * v for k, v in terms.items())
            total.backward(); section_end('combined_parameter_backward')
            return terms, total
        plain, plain_total = calculate(parts_loss)
        plain_grad = torch.cat([p.grad.flatten() for p in model.parameters()]).detach().clone()
        observed, observed_total = calculate(instrumented)
        observed_grad = torch.cat([p.grad.flatten() for p in model.parameters()]).detach()
        assert torch.equal(plain_grad, observed_grad)
        assert all(torch.equal(plain[k], observed[k]) for k in plain)
        checks.append({'case': case, 'seed': 42, 'loss_terms_bitwise_equal': True,
            'parameter_gradient_bitwise_equal': True, 'accepted_step': result['accepted_steps'],
            'training_closure_evaluations': result['closure_evaluations'],
            'interface_segments': 4, 'points_per_segment': 250,
            'paired_interface_point_evaluations_per_training_closure': 1000,
            'subnetwork_interface_forward_calls_per_training_closure': 8,
            'training_interface_segment_evaluations': 4 * result['closure_evaluations'],
            'training_interface_subnetwork_point_evaluations': 2000 * result['closure_evaluations']})
        for repeat in range(-5, 20):
            calculate(instrumented)
            if repeat < 0: continue
            total = sum(sections.values())
            for name, seconds in sections.items():
                records.append({'case': case, 'seed': 42, 'repeat': repeat,
                    'section': name, 'seconds': seconds, 'instrumented_closure_seconds': total,
                    'fraction_of_instrumented_closure': seconds / total})
        print('PROFILE ' + case, flush=True)
    write_csv('xpinn_closure_profile.csv', records)
    info = {'checks': checks, 'warmup': 5, 'measured_repeats': 20,
        'original_parts_loss_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'scope': 'Final seed-42 checkpoints; GPU synchronized at section boundaries. Backward is combined, so interface backward cannot be assigned from this table. Instrumentation adds synchronization overhead; ratios describe the profiled closure only.',
        'not_independent_training_replicates': True}
    (ROOT / 'checks/xpinn_profile.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
