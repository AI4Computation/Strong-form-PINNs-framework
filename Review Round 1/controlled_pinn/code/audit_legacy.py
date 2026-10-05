"""CPU-only consistency checks of submitted historical results.

No optimizer, no modification of old files, no use of excluded tunnel studies.
Fine-reference re-evaluation keeps the legacy evaluation locations, apart from
sub-1e-6 coordinate-rounding corrections on the analytical circular boundary.
"""
import runtime
from runtime import torch
torch.set_num_threads(1)
from torch import nn
from models import ROOT, PROJECT, legacy
from metrics import field_metrics, relative
from fem_interpolation import Field
from summarize import write_csv
import numpy as np, json, csv, hashlib, sys

class PINN_Network(nn.Module):
    def forward(self, x):
        if hasattr(self, 'fixed_layer'):
            return self.trainable_net(self.activation(self.fixed_layer(x)))
        return self.net(x)

def legacy_reference():
    basedir = PROJECT / '代码源文件/【abaqus-36】'
    configs = [('2out_hard', basedir/'sensitivity_results+仅输出位移+硬约束', True),
        ('5out_hard', basedir/'sensitivity_results+输出五个变量+硬约束位移', True),
        ('5out_soft', basedir/'sensitivity_results+位移软约束', False),
        ('anchored_legacy', PROJECT/'代码源文件/【abaqus-36 - validation】/sensitivity_results', False)]
    folder = PROJECT / 'abaqus-36/processed/displacement'
    template = np.loadtxt(folder/'abaqus-p_lateral1-p_top5.csv', delimiter=',', skiprows=1)
    original_xy = template[:, :2]; xy = original_xy.copy()
    radius = np.linalg.norm(xy, axis=1); boundary = abs(radius-.1) < 1e-6
    xy[boundary] *= (.1 + 2e-8) / radius[boundary, None]
    shift = np.linalg.norm(xy-original_xy, axis=1)
    assert shift.max() < 1e-6
    units = []; unit_sources = {}
    for step in ['UnitL', 'UnitT']:
        path = ROOT / 'fem' / f'tr3_circle_sq0p0025_{step}.npz'
        data = dict(np.load(path)); displacement, _, _ = Field(data, 1.333, .3333).evaluate(xy)
        units.append(displacement); unit_sources[step] = hashlib.sha256(path.read_bytes()).hexdigest()
    references = {}; reference_rows = []
    for p in range(0, -6, -1):
        for pt in range(0, -6, -1):
            token = lambda v: str(abs(v)) if v else '1e-300'
            path = folder / f'abaqus-p_lateral{token(p)}-p_top{token(pt)}.csv'
            old = np.loadtxt(path, delimiter=',', skiprows=1)
            assert np.array_equal(old[:, :2], original_xy), path
            fine = -p*units[0] - pt*units[1]; references[(p, pt)] = fine
            reference_rows.append({'p_lateral': p, 'p_top': pt, 'points': len(xy),
                'old_to_refined_u_vector_pct': relative(fine, old[:, 2:4]),
                'reference_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    # The original pickle classes were saved in __main__.
    setattr(sys.modules['__main__'], 'PINN_Network', PINN_Network)
    tensor = torch.tensor(xy, dtype=torch.float32); rows = []
    for name, folder, hard in configs:
        for (p, pt), fine in references.items():
            label = f'P_{p}_Ptop_{pt}'; path = folder/label/(label+'.pth')
            model = torch.load(path, map_location='cpu', weights_only=False).eval()
            with torch.no_grad():
                uv = model(tensor)[:, :2]
                if hard:
                    uv *= torch.cat([tensor[:, 0:1]**2+(tensor[:, 1:2]+.5)**2, tensor[:, 1:2]+.5], 1)
            rows.append({'architecture': name, 'p_lateral': p, 'p_top': pt,
                'relative_defined': bool(np.linalg.norm(fine) > 0),
                **field_metrics(fine, uv.numpy().astype(np.float64), ['u', 'v']),
                'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print('LEGACY REFERENCE ' + name, flush=True)
    write_csv('legacy_reference_sensitivity.csv', reference_rows)
    write_csv('legacy_36case_refined_metrics.csv', rows)
    np.savez_compressed(ROOT/'summary/legacy_evaluation_points.npz', original_xy=original_xy, evaluation_xy=xy,
        unit_lateral_u=units[0], unit_top_u=units[1])
    report = {'old_reference_max_vector_difference_pct': max(r['old_to_refined_u_vector_pct'] or 0 for r in reference_rows),
        'old_reference_min_nonzero_vector_difference_pct': min(r['old_to_refined_u_vector_pct'] for r in reference_rows if r['old_to_refined_u_vector_pct'] is not None),
        'evaluation_points': len(xy), 'boundary_rounding_adjusted_points': int(boundary.sum()),
        'max_coordinate_adjustment_normalized': float(shift.max()), 'fine_unit_sources_sha256': unit_sources,
        'scope': 'Same 144 saved checkpoints; CPU float32 inference; 36 legacy location sets verified identical. Fine-reference displacement interpolated by containing-element shape functions. Circular boundary coordinates rounded by the old CSV were projected radially to r=0.10000002 (offset 2e-8 in normalized coordinates) before both new prediction and new reference evaluation; max adjustment <1e-6. No stress or timing claim added to historical data.',
        'architecture_summary': {name: {'nonzero_cases': 35,
            'mean_nonzero_vector_pct': float(np.mean([r['vector_pct'] for r in rows if r['architecture']==name and r['relative_defined']]))}
            for name, _, _ in configs}}
    (ROOT/'checks/legacy_reference_sensitivity.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

def legacy_dictionary():
    source = PROJECT/'_work_revision/feature_ablation_enriched.csv'
    with source.open(encoding='utf-8-sig', newline='') as f: old = list(csv.DictReader(f))
    assert len(old) == 33
    axis = np.linspace(-.495, .495, 64); x, y = np.meshgrid(axis, axis)
    xy = np.column_stack([x.ravel(), y.ravel()]); xy = xy[np.linalg.norm(xy, axis=1) > .1]
    points = torch.tensor(xy, dtype=torch.float32); rows = []
    for r in old:
        seed = int(float(r['seed'])); n = int(float(r['n_features'])); wmax = float(r['wmax'])
        # Original constructor draws centers then W from CPU default RNG.
        torch.random.default_generator.manual_seed(seed)
        model = legacy.AnchoredMixed(n_features=n, w_max=wmax)
        with torch.no_grad():
            h = torch.tanh(points @ model.weights.T + model.bias).numpy().astype(np.float64)
        h -= h.mean(0); norm = np.linalg.norm(h, axis=0); alive = norm > 1e-12
        q = np.zeros_like(h); q[:, alive] = h[:, alive]/norm[alive]
        gram = q.T @ q; corr = np.abs(gram[np.ix_(alive, alive)])
        off = corr[~np.eye(len(corr), dtype=bool)]
        eigen = np.maximum(np.linalg.eigvalsh(gram), 0.); largest = eigen[-1]; cut = 1e-6*largest
        active = eigen[eigen > cut]; prob = eigen[eigen > 0]/eigen.sum()
        rows.append({'kind': r['kind'], 'value': float(r['value']), 'seed': seed, 'wmax': wmax, 'n_features': n,
            'points': len(xy), 'constant_columns': int((~alive).sum()),
            'coherence_max': float(off.max()), 'coherence_q95': float(np.quantile(off, .95)),
            'numerical_rank': len(active), 'effective_rank': float(np.exp(-np.sum(prob*np.log(prob)))),
            'regularized_condition': float((largest+cut)/(eigen[0]+cut)),
            'effective_spectral_condition': float(largest/active[0]),
            'legacy_u_pct': float(r['rel_l2_u_pct']), 'legacy_v_pct': float(r['rel_l2_v_pct']),
            'legacy_optimizer_budget': int(float(r['optimizer_max_iter'])),
            'legacy_closure_evaluations': int(float(r['closure_evaluations']))})
    write_csv('legacy_dictionary_diagnostics.csv', rows)
    note = {'source': str(source.relative_to(PROJECT)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'n_rows': 33, 'no_retraining': True, 'checkpoints_available': False,
        'activation': 'Fixed weights/centers reconstructed with the original CPU constructor and seed in the current runtime; float32 CPU tanh. Historical fixed arrays were not archived, so bytewise identity to those past arrays cannot be verified.',
        'evaluation_grid': 'New common 64x64 grid in [-0.495,0.495]^2, r>0.1, consistent with new E5 diagnostics. These are not claims of exact recomputation of the old random-grid Gram values.',
        'gram': 'float64 centered correlation Gram; zero columns retained as zero, no forced unit diagonal; nondefined correlations excluded; rank cutoff 1e-6 lambda_max.',
        'legacy_accuracy': 'Original logged component errors kept explicitly under legacy_ columns; model weights were not saved by the old ablation runner, so these errors cannot be re-evaluated against refined FEM. They do not enter new paired statistics or time-to-accuracy.',
        'stale_old_runner': 'Original feature_ablation.py calls removed bench.evaluate; the revision does not run or rewrite that historical script.'}
    (ROOT/'checks/legacy_dictionary_audit.json').write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding='utf-8')
    print('LEGACY DICTIONARY 33 mappings checked', flush=True)

if __name__ == '__main__':
    legacy_reference(); legacy_dictionary()
