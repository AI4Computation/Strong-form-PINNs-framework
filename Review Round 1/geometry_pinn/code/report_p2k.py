"""Complete prospective two-seed and descriptive three-seed transfer report."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2K_transfer_replication';J=ROOT/'results/R2_P2J_reference_repair'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def flatten(v,prefix=''):
    if isinstance(v,dict):
        for k,x in v.items():yield from flatten(x,prefix+'.'+k if prefix else k)
    elif isinstance(v,list):
        for i,x in enumerate(v):yield from flatten(x,f'{prefix}[{i}]')
    elif isinstance(v,(int,float)):yield prefix,float(v)
def main():
    p=read(ROOT/'protocols/R2_P2K_transfer_replication.json');status=read(B/'evaluation_progress.json');assert status['status']=='complete' and status['completed_fields']==48
    paths=[J/'evaluation/analysis.json']+[B/f'seed{s}/evaluation/analysis.json' for s in p['seeds']];allrows={};spatial={};sources={};seeds=[p['prior_seed']]+p['seeds'];labels={'fourier_half':'F','geometry_rbf_fourier':'G','uniform_rbf_fourier':'U'}
    for path in paths:
        a=read(path);assert len(a['results'])==24;sources[str(path.resolve())]=sha(path)
        for tag,row in a['results'].items():
            assert tag not in allrows;allrows[tag]=dict(row,source_analysis=str(path.resolve()));spatial[f'{row["case"]}_{row["seed"]}_{row["step"]}']=a['spatial'][f'{row["case"]}_step{row["step"]:04d}']
    def row(c,seed,step,method,arm):return allrows[f'{c}_seed{seed}_{method}_{arm}_step{step:04d}']
    def values(r):
        v=dict(flatten(r['metrics']));v={k:x for k,x in v.items() if any(t in k for t in ['relative_l2_percent','vector_mae','vector_absolute_error_quantiles','max_vector_absolute_error','extrema_vector_error','convergence_absolute_error'])}
        for i in range(2):
            v[f'physics.global[{i}]']=r['validation'][i]['total_mean']
            for name,s in r['regions'][i].items():v[f'physics.{name}[{i}]']=s['total_mean']
        return v
    summary=dict(protocol_sha256=sha(ROOT/'protocols/R2_P2K_transfer_replication.json'),source_analysis_sha256=sources,report_source_sha256=sha(__file__),seeds=seeds,prospective_seeds=p['seeds'],results=allrows,groups={},contrasts={},targets={},spatial=spatial,resources={})
    for c in p['cases']:
     for step in p['evaluation_steps']:
      for arm in p['arms']:
       for method in p['methods']:
        vs=[values(row(c,s,step,method,arm)) for s in seeds];summary['groups'][f'{c}_{step}_{arm}_{method}']={k:dict(values=[v[k] for v in vs],mean=float(np.mean([v[k] for v in vs])),minimum=min(v[k] for v in vs),maximum=max(v[k] for v in vs)) for k in vs[0]}
       for control in ['fourier_half','uniform_rbf_fourier']:
        gv=[values(row(c,s,step,'geometry_rbf_fourier',arm)) for s in seeds];cv=[values(row(c,s,step,control,arm)) for s in seeds];out={}
        for metric in gv[0]:
            v=np.array([x[metric]/max(y[metric],1e-300) for x,y in zip(gv,cv)]);assert np.isfinite(v).all();out[metric]=dict(ratios=v.tolist(),geometric_mean=float(np.exp(np.log(np.maximum(v,1e-300)).mean())),minimum=float(v.min()),maximum=float(v.max()),favorable_count=int((v<1).sum()),prospective_favorable_count=int((v[1:]<1).sum()),both_new_seeds_favorable=bool((v[1:]<1).all()))
        summary['contrasts'][f'{c}_{step}_{arm}_{control}']=out
    s1keys=['u_area.relative_l2_percent','s_area.relative_l2_percent','wall_u.relative_l2_percent'];t1keys=['u_area.relative_l2_percent','s_area.relative_l2_percent']+[f'{f}.{k}' for f in ['u_area','s_area'] for k in ['vector_mae','vector_absolute_error_quantiles[2]']]
    summary['registered_decisions']={}
    for name,c,step,keys in [('S1_primary','S1',1600,s1keys),('T1_early_distribution','T1',800,t1keys)]:
        cells={control:{k:summary['contrasts'][f'{c}_{step}_uniform_refresh_{control}'][k]['both_new_seeds_favorable'] for k in keys} for control in ['fourier_half','uniform_rbf_fourier']};summary['registered_decisions'][name]=dict(all_named_measures_repeated=all(all(x.values()) for x in cells.values()),by_control_and_metric=cells)
    for c in p['cases']:
     for method in p['methods']:
      for arm in p['arms']:
       for target in [(5,20,10),(2,10,5),(1,5,2)]:
        out={}
        for seed in seeds:
            hits=[all(row(c,seed,step,method,arm)['metrics'][k]['relative_l2_percent']<=t for k,t in zip(['u_area','s_area','wall_u'],target)) for step in p['evaluation_steps']];out[str(seed)]=dict(attainment=hits,first_observed_step=next((step for step,hit in zip(p['evaluation_steps'],hits) if hit),None))
        summary['targets'][f'{c}_{method}_{arm}_{target}']=dict(target=target,by_seed=out,ever_count=sum(x['first_observed_step'] is not None for x in out.values()),at_1600_count=sum(x['attainment'][1] for x in out.values()))
    for seed in seeds:
        train=ROOT/'results/R2_P2I_shape_transfer' if seed==p['prior_seed'] else B/f'seed{seed}'
        for name in read(train/'manifest.json')['completed']:summary['resources'][name]=dict(read(train/name/'result.json'),source_path=str((train/name/'result.json').resolve()))
    lines=['# P2K: two-seed prospective replication of T1/S1 transfer','',
      'The two prespecified new seeds are 260931 and 260932. The discovery seed 260930 is shown separately in every seed table and included only in descriptive three-seed summaries. All 24 new trajectories completed; all 48 new endpoints are included with the 24 frozen discovery endpoints (72 fields total). There is no selected seed, FEM-best checkpoint, per-cavity tuning, formal timing, or new architecture.','',
      '## Registered replication decisions','',
      'A named benefit counts as repeated only when it favors geometry against BOTH controls in BOTH new seeds, as preregistered. A failed multimetric headline does not erase individually repeated benefits or other measured trade-offs. Three-seed means and favorable counts are descriptive, not significance tests.','',
      '| Registered comparison | All named measures repeat? |','|---|---|']
    for k,v in summary['registered_decisions'].items():lines.append(f"| {k} | {v['all_named_measures_repeated']} |")
    lines+=['','S1 primary: common refresh, 1600 steps, area U/S and wall U. T1 early distribution: common refresh, 800 steps, area U/S, U/S MAE and U/S 95th error-norm percentile. Full pass/fail by metric and control is in `summary.json`. A failed headline requires narrowing that claim; it does not authorize another tuning or seed search.','',
      '## Protocol and verification','',
      'Both cases reuse the frozen P2I trainer and shared PINN (110705 trainable parameters). Geometry centres/widths, actual materials/tractions, 6000 domain points, boundary counts, loss weights and four matched-reset 400-step L-BFGS blocks are unchanged. Fixed/reset and uniform-refresh arms have exactly identical first blocks. Fresh seeds alter initialization and the common point stream. All 24 new fits freeze before reference-array evaluation.','',
      'Byte-verified P2J references and both 24000-point independent physical sets are reused for all three seeds. The precision-aware solid-side wall rule and original reference sensitivity limits remain unchanged. All endpoints receive complete FEM, region, distribution, wall, engineering and derivative checks; original P2I/P2J files remain immutable.','',
      '## Complete primary errors by seed','',
      'F = complete half-bandwidth Fourier; G = geometry Gaussian–Fourier; U = equal-capacity uniform Gaussian–Fourier. Percent errors are area-weighted vector U/S and arc-length-weighted total wall U. Seed 260930 is discovery; the other two are prospective replication.','',
      '| Case | Seed | Steps | Arm | Model | U (%) | S (%) | Wall U (%) |','|---|---:|---:|---|---|---:|---:|---:|']
    for c in p['cases']:
     for seed in seeds:
      for step in p['evaluation_steps']:
       for arm in p['arms']:
        for method in p['methods']:
            r=row(c,seed,step,method,arm);v=r['metrics'];lines.append(f'| {c} | {seed} | {step} | {arm} | {labels[method]} | '+' | '.join(f"{v[k]['relative_l2_percent']:.5f}" for k in ['u_area','s_area','wall_u'])+' |')
    lines+=['','## Paired primary contrasts','','Ratios below one favor G. GM is the geometric mean of the three paired ratios, not a ratio of arithmetic means. The last column records favorable counts among the two NEW seeds.','',
      '| Case | Steps | Arm | Control | Metric | Discovery ratio | New seed 1 | New seed 2 | GM ratio | Favorable all/new |','|---|---:|---|---|---|---:|---:|---:|---:|---|']
    for c in p['cases']:
     for step in p['evaluation_steps']:
      for arm in p['arms']:
       for control in ['fourier_half','uniform_rbf_fourier']:
        for metric in s1keys:
            v=summary['contrasts'][f'{c}_{step}_{arm}_{control}'][metric];lines.append(f'| {c} | {step} | {arm} | {labels[control]} | {metric.split(".")[0]} | '+' | '.join(f'{x:.5f}' for x in v['ratios'])+f" | {v['geometric_mean']:.5f} | {v['favorable_count']}/3; {v['prospective_favorable_count']}/2 |")
    lines+=['','## Distribution, spatial coverage and engineering trade-offs','','Absolute U/S error statistics use each case’s normalized physical units. MAE and p95 are vector-norm errors. V/H are absolute vertical/horizontal convergence errors. Remaining p50/p90/p99/max, near/far/corner, all extrema, node/point and per-hole wall metrics are retained in every source metric JSON and aggregate summary.','',
      '| Case | Seed | Steps | Arm | Model | U MAE | U p95 | S MAE | S p95 | V error | H error |','|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in allrows.values():
        v=r['metrics'];nums=[v['u_area']['vector_mae'],v['u_area']['vector_absolute_error_quantiles'][2],v['s_area']['vector_mae'],v['s_area']['vector_absolute_error_quantiles'][2],*v['engineering']['convergence_absolute_error']];lines.append(f"| {r['case']} | {r['seed']} | {r['step']} | {r['arm']} | {labels[r['method']]} | "+' | '.join(f'{x:.6g}' for x in nums)+' |')
    lines+=['','| Case | Seed | Steps | Arm | Control | U improved area (%) | S improved area (%) | Wall improved length (%) |','|---|---:|---:|---|---|---:|---:|---:|']
    for c in p['cases']:
     for seed in seeds:
      for step in p['evaluation_steps']:
       for arm in p['arms']:
        for control in ['fourier_half','uniform_rbf_fourier']:
            v=spatial[f'{c}_{seed}_{step}']['geometry_rbf_fourier_'+arm][control+'_'+arm];lines.append(f'| {c} | {seed} | {step} | {arm} | {labels[control]} | '+' | '.join(f'{100*v[k]:.3f}' for k in ['u','s','wall_u'])+' |')
    lines+=['','## Target persistence and cost','','Targets (U%, S%, wall U%) are research comparisons, not engineering acceptance. Entries for each seed show attainment at 800/1600; first observed attainment is not an optimized stop. All costs are development records, so this study does not establish fair time-to-target.','',
      '| Case | Model | Arm | Target | Discovery 800/1600 | New 1 800/1600 | New 2 800/1600 | Ever / at 1600 |','|---|---|---|---|---|---|---|---|']
    for c in p['cases']:
     for method in p['methods']:
      for arm in p['arms']:
       for target in [(5,20,10),(2,10,5),(1,5,2)]:
            v=summary['targets'][f'{c}_{method}_{arm}_{target}'];hits=['/'.join('yes' if x else 'no' for x in v['by_seed'][str(seed)]['attainment']) for seed in seeds];lines.append(f'| {c} | {labels[method]} | {arm} | {target} | '+' | '.join(hits)+f" | {v['ever_count']}/3; {v['at_1600_count']}/3 |")
    lines+=['','| Run | Closures | Peak allocated MiB | Peak reserved MiB |','|---|---:|---:|---:|']
    for name,r in summary['resources'].items():lines.append(f"| {name} | {r['closure_evaluations']} | {r['peak_allocated_bytes']/2**20:.4f} | {r['peak_reserved_bytes']/2**20:.4f} |")
    lines+=['','## Raw data and boundaries','','All files are under `research/geometry_pinn/`; all 72 source-row paths and hashes are in the aggregate `results/R2_P2K_transfer_replication/summary.json`.','',
      '- Master protocol: `protocols/R2_P2K_transfer_replication.json`; immutable child protocols: `R2_P2K_seed260931.json` and `R2_P2K_seed260932.json`.',
      '- New training root: `results/R2_P2K_transfer_replication/seed<seed>/`; each `<case>_seed<seed>_<method>_<arm>/` has all four checkpoint files, complete accepted-step/closure traces, and per-block development preparation/optimization durations and memory in `result.json`. The per-seed point NPZ files retain all coordinates, normals, tags, pressures and weights.',
      '- Each new seed’s `evaluation/<run>_step<0800|1600>_predictions.npz` contains all IP U/S, original-node U, wall and engineering-point predictions. `_error_norms.npz` contains every IP/wall error magnitude; `_residuals.npz` contains training/independent/boundary residual arrays and numerical audit samples. `_metrics.json` contains all scalar/distribution/engineering results.',
      '- Frozen discovery predictions and metrics: `results/R2_P2J_reference_repair/evaluation/`. Geometry, FEM weights/masks, reference fields and common independent probe arrays are reused there with hashes; they are not regenerated for favorable outcomes.',
      '- `summary.json` retains all 72 endpoint metric rows with source analysis paths, all metric-family ratios, per-seed target persistence, spatial comparisons and resource records. `audit.json` and `completion.json` record raw arithmetic and immutable preservation checks.',
      '- Code: `train_p2k.py` serially wraps verified `train_p2i.py`; `evaluate_p2k.py` reuses the verified material-aware physics routines; `report_p2k.py` and `audit_close_p2k.py` produce this report and closure.',
      '',
      'These tests concern the new geometry-feature combination, not proof of the original anchor binding. T1/S1 are development geometries; C1/C8 limitations and original-bandwidth-only C1 claims remain unchanged. No fair timing, lower-memory, statistical-significance or universal-stability claim is licensed by this batch.','']
    interpretation=[
      '## What the replication establishes','',
      'Neither preregistered multimetric headline repeats in full. S1 does not consistently improve area U, S and wall U together; T1 does not improve every named early-budget field/distribution measure against both controls. The useful result is narrower: independent geometry-feature benefits are repeatable for several physically interpretable outcomes, with displacement, sampling and engineering trade-offs that must remain explicit.','',
      '**S1, common refresh, 1600 steps: stress is the strongest repeatable gain.** Area S and total wall U errors are lower than BOTH controls in both new seeds and the discovery seed. Against half-Fourier, paired geometric-mean reductions over all three seeds are 5.90% (S) and 10.90% (wall U); against uniform Gaussian features, 4.47% and 7.24%. The S1 primary contrast therefore supports these two named outcomes, but not the full three-outcome headline. New-seed wall reductions against Fourier are only 0.58% and 2.04%, far smaller than the discovery reduction of 27.37%.','',
      'S1 stress improvement is spatially broad: the solid-area fraction with smaller stress-vector error is 67.28%, 64.84% and 62.46% against Fourier, and 65.38%, 69.97% and 57.72% against uniform features (discovery, new 1, new 2). Stress MAE and p50/p90/p95/p99 all improve against both controls in all three seeds. Paired geometric-mean MAE/p95 reductions are 11.22%/12.32% against Fourier and 8.58%/9.64% against uniform. This supports a distributional contribution beyond one RMS scalar.','',
      'The S1 displacement trade-off is substantial. At common-refresh 1600, both new seeds have larger area U error than Fourier, by 12.32% and 63.16%; the pooled paired geometric-mean U ratio is 1.1550 despite the favorable discovery seed. Only 23.90% and 26.56% of the solid area improve in those new seeds. The arithmetic three-seed means are G 2.17795% versus F 2.13268%, which look close; the individual ratios and spatial maps reveal the inconsistent gain that an average would conceal. Vertical convergence errors also worsen against Fourier in both new seeds.','',
      'Lower S1 wall L2 does not mean most of the wall improves. Against Fourier, improved wall length falls from 91.55% in discovery to 36.47% and 49.56% in the new seeds, while wall MAE becomes 3.20% and 4.05% worse. Wall p99 and maximum errors improve; the RMS gain coexists with worsened typical wall locations. Against the uniform control, horizontal convergence errors worsen in all three seeds. The full distributions and engineering values below must accompany any wall-error claim.','',
      '**T1, common refresh, 800 steps: the independent geometry comparison repeats more strongly than the Fourier comparison.** Against equal-capacity uniform Gaussian features, U/S/wall L2, U/S MAE and p95, and both engineering convergence errors all improve in all three seeds. Paired geometric-mean U/S/wall reductions are 17.42%/2.95%/5.56%. This is evidence that the geometry rule contributes beyond the feature family and parameter count for this registered condition.','',
      'Against Fourier, T1 early U/S improve in only two of three seeds, with the first new seed slightly worse in both. Stress p90/p95 and horizontal-convergence error improve in all three; p95 has a paired geometric-mean reduction of 8.37%. U MAE nominally improves in every seed, but the first new-seed margin is only about 0.008%, so it should not be presented as a substantial repeatable improvement. The broad preregistered T1 headline fails. At T1 fixed/reset 800 and 1600, all three primary errors beat Fourier in all three seeds, but the uniform-feature control wins some discovery comparisons; this is a sampling-specific result, not a reason to choose different arms after seeing FEM.','',
      '**Sampling sensitivity affects the combined model as well.** For S1 new seed 260931 at fixed/reset 1600, G wall U error reaches 14.4604%, compared with Fourier 4.9022% and uniform 7.6471%; at common refresh it falls to 2.6625%. Thus the combined model cannot be portrayed as intrinsically stable under fixed collocation. S1 total independent residual is higher than both refreshed controls on both probe sets in all three seeds, although stress distributions and far-region residuals improve. These measures diagnose different aspects of the field; neither can replace the others.','',
      '**Accuracy targets and resources do not establish an efficiency advantage.** At the intermediate joint target (U<=2%, S<=10%, wall<=5%), all three refreshed S1 methods attain 2/3, and all three refreshed T1 methods attain 2/3 at an earlier endpoint but retain 0/3 at 1600. Fixed/reset S1 attains G 2/3, F 1/3, U 0/3, accompanied by the severe failed G wall trajectory above. All runs use 110705 parameters; observed peak allocated memory spans 305.463–305.604 MiB and closure counts span 1671–1702 across these 36 original/new trajectories. No fair runtime or resource-saving inference is made.','',
      '**Decision.** Close this replication batch and stop further seed/budget/nearby-architecture expansion. Retain the repeated S1 stress-distribution and scoped T1 geometry-control evidence, alongside all displacement/wall/sampling limitations. The next work is a concise contribution-and-claims review across C1/C8/L1/T1/S1, followed by a fixed common accuracy–cost comparison design if these scoped benefits justify it. Fair timing remains conditional on a separate notification and fresh idle-machine confirmation. There is no basis to restore the original anchoring-only accuracy attribution or claim universal superiority.',''
    ]
    index=lines.index('## Registered replication decisions');lines[index:index]=interpretation
    write(B/'summary.json',summary);(ROOT/'P2K_Transfer_Replication_Report.md').write_text('\n'.join(lines),encoding='utf-8');print('REPORT_72_ENDPOINTS_WRITTEN',summary['registered_decisions'],flush=True)
if __name__=='__main__':main()
