"""Tabulate all registered transfer endpoints without model/metric selection."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2I_shape_transfer';E=B/'evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    a=read(E/'analysis.json');p=read(ROOT/'protocols/R2_P2I_shape_transfer.json');assert len(a['results'])==24
    summary=dict(analysis_sha256=sha(E/'analysis.json'),report_source_sha256=sha(__file__),contrasts={},targets={},resources={})
    lines=['# P2I: fixed-rule T1/S1 development transfer','',
      'Twelve fresh training trajectories and all 24 preregistered 800/1600-step endpoints are included. This is a single-seed development study (260930), not a replication claim or unseen-geometry validation. No FEM values selected features, regions, checkpoints or training hyperparameters. This batch does not establish the independent contribution of the original anchored mapping.','',
      '## Question and controls','',
      'Does the unchanged geometry-scaled Gaussian–Fourier input rule retain useful accuracy or spatial benefits on the submitted tunnel plus pressurized ellipse (T1) and square cavity (S1), against complete half-bandwidth Fourier and equal-capacity uniform Gaussian–Fourier controls? Every method uses both fixed points with matched optimizer resets and common uniform refresh. No method receives an individually selected sampling arm.','',
      'All models have 110705 trainable parameters and the same 1000-input shared network. Frozen P2D centres and widths are reused: 360 geometric features for T1 and 376 for S1. Both arms run four 400-accepted-step L-BFGS blocks and share exactly equal first-block states. Six thousand domain samples, boundary counts, loss weights and optimizer settings are common. Refresh changes only the uniform interior subset; support probes and boundary points remain fixed. Runtime logs are development telemetry, not fair timing evidence.','',
      'T1 uses its actual nondimensional E=1, nu=0.26, outer pressure -4 and ellipse pressure -1; the tunnel is free. Length, displacement and stress scales are 75 m, 0.01875 m and 2.5 MPa. S1 uses E=1.333, nu=0.3333, side pressure -1, top pressure -5 and a free cavity. Physical problem data change; representation rules and training weights do not. Each physical hole tag contributes a separate normalized boundary mean.','',
      '## Reference and numerical verification','',
      'Both references were checked after all training had frozen and before any method ranking was evaluated. FEM stresses are retained at integration points; displacement there is interpolated using the corresponding finite element. S1 uses exact input mesh coordinates. Wall interpolation uses solid-side offsets.','',
      '| Case | IP count | Stress reconstruction error (%) | Area | Maximum wall-offset sensitivity |','|---|---:|---:|---:|---:|']
    for c,r in a['references'].items():lines.append(f"| {c} | {r['integration_points']} | {r['stress_reconstruction_percent']:.7g} | {r['area']:.9g} | {r['wall_offset_sensitivity_max']:.4g} |")
    e64=max(v['numerical_audit']['float64_scaled_max_error'] for v in a['results'].values());e32=max(v['numerical_audit']['float32_scaled_max_error'] for v in a['results'].values())
    lines+=['',f'All 24 hotspot/random-point derivative audits passed (maximum scaled float64/autograd discrepancy {e64:.4g}; float32/float64 discrepancy {e32:.4g}). Training objectives were independently recomputed against the stored traces. Two 24000-point independent physical sets are common to all methods, arms and endpoints within each case.','',
      '## Complete primary error matrix','',
      'Errors are relative vector L2 percentages. U and S are area-weighted integration-point metrics; wall U is arc-length-weighted over all cavities. Each case has its own reference scale; no mean across cases is used. F = half-bandwidth Fourier; G = geometry Gaussian–Fourier; U = uniform Gaussian–Fourier.','',
      '| Case | Steps | Arm | Model | Area U | Area S | Wall U | Cavity wall U | Water wall U |','|---|---:|---|---|---:|---:|---:|---:|---:|']
    labels={'fourier_half':'F','geometry_rbf_fourier':'G','uniform_rbf_fourier':'U'}
    def row(c,s,m,arm):return a['results'][f'{c}_seed{p["seed"]}_{m}_{arm}_step{s:04d}']
    for c in p['cases']:
      for step in p['evaluation_steps']:
       for arm in p['arms']:
        for method in p['methods']:
         r=row(c,step,method,arm);v=r['metrics'];water=f"{v['wall_u_water']['relative_l2_percent']:.4f}" if 'wall_u_water' in v else '—'
         lines.append(f"| {c} | {step} | {arm} | {labels[method]} | {v['u_area']['relative_l2_percent']:.4f} | {v['s_area']['relative_l2_percent']:.4f} | {v['wall_u']['relative_l2_percent']:.4f} | {v['wall_u_cavity']['relative_l2_percent']:.4f} | {water} |")
    lines+=['','## Geometry contribution and spatial coverage','','A ratio below one favors G. Improvement fractions compare per-location error magnitudes, not relative errors with local near-zero denominators. All contrasts are within the same case, budget and sampling arm.','',
      '| Case | Steps | Arm | Control | G/control U | S | Wall U | U improved area (%) | S improved area (%) | Wall improved length (%) |','|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for c in p['cases']:
     for step in p['evaluation_steps']:
      for arm in p['arms']:
       gv=row(c,step,'geometry_rbf_fourier',arm)['metrics']
       for control in ['fourier_half','uniform_rbf_fourier']:
        cv=row(c,step,control,arm)['metrics'];ratios={k:gv[k]['relative_l2_percent']/cv[k]['relative_l2_percent'] for k in ['u_area','s_area','wall_u']};space=a['spatial'][f'{c}_step{step:04d}']['geometry_rbf_fourier_'+arm][control+'_'+arm]
        summary['contrasts'][f'{c}_{step}_{arm}_{control}']=dict(primary_ratios=ratios,spatial_fraction=space)
        lines.append(f"| {c} | {step} | {arm} | {labels[control]} | "+' | '.join(f'{ratios[k]:.4f}' for k in ['u_area','s_area','wall_u'])+' | '+' | '.join(f'{100*space[k]:.2f}' for k in ['u','s','wall_u'])+' |')
    lines+=['','## Error distributions and engineering quantities','','All errors in this table use the case-specific nondimensional units. U/S MAE and 95th/99th percentiles concern vector error norms. Vertical and horizontal convergence errors use the geometry-defined crown/invert and right/left points. Peaks at sharp corners are not claimed to be mesh-converged physical stresses. Full 50/90/95/99/max, near/far/corner, original-node, unweighted-point and individual extrema values remain in each metrics JSON.','',
      '| Case | Steps | Arm | Model | U MAE | U p95 | U p99 | S MAE | S p95 | S p99 | Vertical convergence error | Horizontal convergence error |','|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in a['results'].values():
        v=r['metrics'];nums=[]
        for k in ['u_area','s_area']:nums.extend([v[k]['vector_mae'],*v[k]['vector_absolute_error_quantiles'][2:]])
        nums.extend(v['engineering']['convergence_absolute_error']);lines.append(f"| {r['case']} | {r['step']} | {r['arm']} | {labels[r['method']]} | "+' | '.join(f'{x:.6g}' for x in nums)+' |')
    lines+=['','## Fixed-endpoint targets and resources','','The triples (U%, S%, wall U%) are research comparison targets, not engineering acceptance criteria. “First” means the first registered evaluated endpoint, not an FEM-selected checkpoint. Both observations are shown to expose lost attainment. No time-to-target or resource-saving claim is made.','',
      '| Case | Model | Arm | Target | At 800 | At 1600 | First observed |','|---|---|---|---|---|---|---:|']
    for c in p['cases']:
     for method in p['methods']:
      for arm in p['arms']:
       for target in [(5,20,10),(2,10,5),(1,5,2)]:
        hit=[all(row(c,s,method,arm)['metrics'][k]['relative_l2_percent']<=t for k,t in zip(['u_area','s_area','wall_u'],target)) for s in p['evaluation_steps']];first=next((s for s,h in zip(p['evaluation_steps'],hit) if h),None);key=f'{c}_{method}_{arm}_{target}';summary['targets'][key]=dict(target=target,attainment=hit,first_observed_step=first)
        lines.append(f'| {c} | {labels[method]} | {arm} | {target} | {hit[0]} | {hit[1]} | {first if first else "—"} |')
    lines+=['','| Run | Parameters | Accepted steps | Closure evaluations | Peak allocated MiB | Peak reserved MiB |','|---|---:|---:|---:|---:|---:|']
    for run in read(B/'manifest.json')['completed']:
        r=read(B/run/'result.json');summary['resources'][run]=r;lines.append(f"| {run} | {r['parameters']} | {r['accepted_steps']} | {r['closure_evaluations']} | {r['peak_allocated_bytes']/2**20:.4f} | {r['peak_reserved_bytes']/2**20:.4f} |")
    lines+=['','## Independent physical diagnostics','','These are residual mean squares on independent physical points, not FEM error estimators. Near/far regions include both holes in T1; the corner mask requires an actual turn exceeding 30 degrees rather than every vertex of a discretized smooth contour.','',
      '| Case | Steps | Arm | Model | Global set 0 | Global set 1 | Far set 0 | Far set 1 | Independent traction |','|---|---:|---|---|---:|---:|---:|---:|---:|']
    for r in a['results'].values():
        nums=[x['total_mean'] for x in r['validation']]+[x['far_wall']['total_mean'] for x in r['regions']]+[r['independent_boundary']['traction']];lines.append(f"| {r['case']} | {r['step']} | {r['arm']} | {labels[r['method']]} | "+' | '.join(f'{x:.6g}' for x in nums)+' |')
    lines+=['','## Source-data map','','All paths below are relative to this report directory. No submitted manuscript, response letter or frozen first-round file is modified.','',
      '- Protocol: `protocols/R2_P2I_shape_transfer.json`.',
      '- Implementations: `code/train_p2i.py`, `transfer_mechanics.py`, `evaluate_p2i.py`, `transfer_evaluation_helpers.py`, `report_p2i.py`, `audit_close_p2i.py`; reused dependencies are hashed in the manifest/evaluation record.',
      '- Batch root: `results/R2_P2I_shape_transfer/`. `preflight.json`, `evaluator_preflight.json` and `manifest.json` identify material, provenance, numerical checks and all frozen inputs.',
      '- Each `<case>_seed260930_<method>_<arm>/` contains `step_0400.pt`, `step_0800.pt`, `step_1200.pt`, `step_1600.pt`, `trace.json` and `result.json`. Development preparation/optimization durations are in each block record.',
      '- `evaluation/<case>_reference.npz` contains all derived normalized FEM fields, quadrature weights, masks and both tagged walls; `<case>_reference_checks.json` records original finest-mesh source paths and SHA256, scaling and interpolation checks.',
      '- `evaluation/<case>_physics_points.npz` contains both independent sets, region masks, boundary points/normals/tags/pressures and integration weights.',
      '- For every registered `<run>_step0800` and `<run>_step1600`, `evaluation/<tag>_predictions.npz` stores IP U/S, original-node U, wall U and extrema U; `<tag>_error_norms.npz` stores all IP/wall vector error magnitudes.',
      '- `<tag>_residuals.npz` stores all five active-training and independent residual components, boundary residual arrays, and double/float32 derivative-audit arrays. `<tag>_metrics.json` contains every reported scalar, distribution and engineering value with source hashes.',
      '- `evaluation/analysis.json` aggregates all 24 endpoints and complete paired spatial comparisons; `<case>_step<budget>_spatial.json` contains the same paired fractions. `summary.json` contains derived contrasts, target observations and full resource records.',
      '- `evaluation/audit.json` and `completion.json` record independent raw arithmetic, immutable source/results hashes and preservation checks.','']
    write(E/'summary.json',summary);(ROOT/'P2I_Shape_Transfer_Report.md').write_text('\n'.join(lines),encoding='utf-8');print('P2I_REPORT_TABLES_WRITTEN',len(lines))
if __name__=='__main__':main()
