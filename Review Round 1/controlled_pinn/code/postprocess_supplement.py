"""Complete planned path, elapsed-time, and bandwidth tables from fixed outputs."""
import runtime
from runtime import torch, DEVICE
from models import ROOT, POLYGON, build_model
from metrics import predict, relative
from summarize import write_csv
import numpy as np, json, hashlib, datetime, csv
from collections import defaultdict

def elapsed_time_tables(manifest):
    rows = []; grouped = defaultdict(list)
    for c in manifest:
        r = json.loads((ROOT / 'runs' / c['id'] / 'result.json').read_text())
        trace = r['metric_trace']
        for metric in ['u_vector_pct', 's_vector_pct']:
            for target in [5., 2., 1.]:
                match = next((i for i, t in enumerate(trace) if t[metric] is not None and t[metric] <= target), None)
                row = {'id': c['id'], 'case': c['case'], 'method': c['method'], 'seed': c['seed'],
                    'metric': metric, 'threshold_pct': target, 'reached': match is not None,
                    'budget': c['max_iter'], 'actual_accepted_steps': r['accepted_steps'],
                    'stop_reason': r['stop_reason'], 'final_error_pct': r['metrics'][metric],
                    'accepted_step': None, 'state_wall_lower_s': None, 'state_wall_upper_s': None,
                    'observation_available_wall_s': None, 'optimization_lower_s': None, 'optimization_upper_s': None}
                if match is not None:
                    before = trace[max(match - 1, 0)]; at = trace[match]
                    row.update(accepted_step=at['accepted_step'], state_wall_lower_s=before['state_wall_s'],
                        state_wall_upper_s=at['state_wall_s'], observation_available_wall_s=at['available_wall_s'],
                        optimization_lower_s=before['optimization_s'], optimization_upper_s=at['optimization_s'])
                rows.append(row); grouped[(c['case'], c['method'], metric, target)].append(row)
    groups = []
    for (case, method, metric, target), values in sorted(grouped.items()):
        reached = [v for v in values if v['reached']]
        all_reached = len(reached) == len(values)
        groups.append({'case': case, 'method': method, 'metric': metric, 'threshold_pct': target,
            'n': len(values), 'reached_n': len(reached), 'not_reached_n': len(values) - len(reached),
            'display': 'all reached' if all_reached else 'NR present',
            'median_available_wall_s_if_all_reached': float(np.median([v['observation_available_wall_s'] for v in reached])) if all_reached else None,
            'median_net_optimization_s_if_all_reached': float(np.median([v['optimization_upper_s'] for v in reached])) if all_reached else None})
    write_csv('time_to_accuracy_per_run.csv', rows)
    write_csv('time_to_accuracy_elapsed_summary.csv', groups)

def cost_table(manifest):
    grouped = defaultdict(list); records = []
    for c in manifest:
        r = json.loads((ROOT/'runs'/c['id']/'result.json').read_text())
        grouped[(c['geometry'],c['case'],c['method'])].append(r)
    for (geometry,case,method), values in sorted(grouped.items()):
        counts = {v['trainable_parameters'] for v in values}; fixed = {v['fixed_coefficients'] for v in values}
        assert len(counts) == len(fixed) == 1
        row = {'geometry':geometry,'case':case,'method':method,'n':len(values),
            'trainable_parameters':counts.pop(),'fixed_coefficients':fixed.pop(),
            'budget':values[0]['configuration']['max_iter'],
            'stop_reason_counts':json.dumps({reason:sum(v['stop_reason']==reason for v in values) for reason in sorted({v['stop_reason'] for v in values})})}
        for key in ['accepted_steps','closure_evaluations','optimization_s','total_wall_s','diagnostic_and_checkpoint_s',
                    'peak_cuda_allocated_MiB','peak_cuda_reserved_MiB']:
            a=np.array([v[key] for v in values]);row[key+'_mean']=float(a.mean());row[key+'_sd']=float(a.std(ddof=1))
        records.append(row)
    write_csv('model_cost_summary.csv',records)

def export_trajectories(manifest):
    prefix = ['id','case','method','seed']
    common = ['accepted_step','closure_evaluations','loss','gradient_max','optimization_s','state_wall_s']
    metric_names = common + ['available_wall_s','u_vector_pct','s_vector_pct','s_near_pct']
    with (ROOT/'summary/accepted_trajectories.csv').open('w',encoding='utf-8-sig',newline='') as a, \
         (ROOT/'summary/observed_error_trajectories.csv').open('w',encoding='utf-8-sig',newline='') as b:
        aw = csv.DictWriter(a,fieldnames=prefix+common);bw = csv.DictWriter(b,fieldnames=prefix+metric_names)
        aw.writeheader();bw.writeheader()
        for c in manifest:
            r=json.loads((ROOT/'runs'/c['id']/'result.json').read_text());base={k:c[k] for k in prefix}
            for v in r['accepted_history']:aw.writerow({**base,**{k:v[k] for k in common}})
            for v in r['metric_trace']:bw.writerow({**base,**{k:v.get(k) for k in metric_names}})

def tunnel_paths(manifest):
    reference = dict(np.load(ROOT / 'fem/tunnel_probe_reference.npz'))
    xy = reference['path']; us = reference['path_u']; ss = reference['path_s']
    n_poly = len(POLYGON) - 1; n_ellipse = 360
    groups = [('tunnel_offset_0p5m', slice(0, n_poly)),
              ('karst_offset_0p5m', slice(n_poly, n_poly + n_ellipse)),
              ('rock_bridge', slice(n_poly + n_ellipse, len(xy)))]
    assert len(xy) == n_poly + n_ellipse + 51
    rows = []; maps = {'xy_m': xy, 'u_reference_m': us, 's_reference_MPa': ss}
    for c in [v for v in manifest if v['geometry'] == 'tunnel']:
        folder = ROOT / 'runs' / c['id']; r = json.loads((folder / 'result.json').read_text())
        model = build_model(c['method'], c['seed'], 'tunnel')
        for step in sorted(set([2000, r['accepted_steps']])):
            path = folder / f'step_{step:05d}.pt'
            if not path.exists(): continue
            saved = torch.load(path, map_location=DEVICE, weights_only=True); model.load_state_dict(saved['state_dict'])
            up = predict(model, xy, 'u', 'tunnel'); sp = predict(model, xy, 's', 'tunnel')
            for name, idx in groups:
                rows.append({'id': c['id'], 'method': c['method'], 'seed': c['seed'], 'accepted_step': step,
                    'is_final': step == r['accepted_steps'], 'path': name, 'n_points': len(xy[idx]),
                    'u_vector_pct': relative(us[idx], up[idx]), 's_vector_pct': relative(ss[idx], sp[idx]),
                    'u_vector_max_abs_mm': float(np.linalg.norm(up[idx] - us[idx], axis=1).max() * 1000),
                    's_vector_max_abs_MPa': float(np.linalg.norm(sp[idx] - ss[idx], axis=1).max())})
            if c['seed'] == 42 and step == r['accepted_steps']:
                maps[c['method'] + '_u_m'] = up; maps[c['method'] + '_s_MPa'] = sp
    write_csv('tunnel_local_path_metrics.csv', rows)
    np.savez_compressed(ROOT / 'summary/tunnel_path_fields_seed42.npz', **maps)

def bandwidth_table(manifest):
    rows = []
    for c in manifest:
        if c['case'] not in ['C1', 'C8'] or c['method'] not in ['fourier_half', 'fourier', 'fourier_double']: continue
        r = json.loads((ROOT / 'runs' / c['id'] / 'result.json').read_text())
        rows.append({'id': c['id'], 'case': c['case'], 'seed': c['seed'],
            'sigma_over_default': {'fourier_half': .5, 'fourier': 1., 'fourier_double': 2.}[c['method']],
            'sigma_default': 20 / (2 * np.pi), **r['metrics'], 'optimization_s': r['optimization_s'], 'total_wall_s': r['total_wall_s']})
    assert len(rows) == 48
    write_csv('fourier_bandwidth_sensitivity.csv', rows)
    (ROOT / 'summary/supplementary_definitions.json').write_text(json.dumps({
        'time_origin': 'After reference loading, model creation and sampling; before initial closure. The recorded wall window includes diagnostics and checkpoint writes, but not preparation or final result JSON serialization.',
        'time_to_accuracy_primary': 'Elapsed time until first successful observation is available; raw state-time bracket and net optimization bracket also supplied. NR is not imputed.',
        'path_reference': 'Element-interpolated U and constitutive stress from FE displacement gradients at fixed physical paths; these supplement, not replace, native-IP primary stress errors.',
        'bandwidth': 'All 48 paired runs, no selected-best replacement of registered default. Edge extension needs scientific review after complete results.'}, indent=2))

def main(manifest):
    elapsed_time_tables(manifest); cost_table(manifest); export_trajectories(manifest)
    tunnel_paths(manifest); bandwidth_table(manifest)
    from profile_xpinn import main as profile
    profile()
    provenance = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'code').glob('*.py'))},
        'run_results_sha256':{c['id']:hashlib.sha256((ROOT/'runs'/c['id']/'result.json').read_bytes()).hexdigest() for c in manifest},
        'scope':'Final postprocessing provenance; formal training and reference hashes are recorded separately in every result.json.'}
    (ROOT/'summary/postprocessing_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main(json.loads((ROOT / 'config/run_manifest.json').read_text()))
