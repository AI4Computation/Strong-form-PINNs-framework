"""Join completed physics diagnostics with already frozen FEM evidence; no fitting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2G_frozen_physics';F=ROOT/'results/R2_P2F_repetition_budget'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link(label,p):return f'[{label}](<{Path(p).resolve().as_posix()}>)'
def rng(x,digits=4):return f'{min(x):.{digits}g}–{max(x):.{digits}g}'
def main():
    assert not (B/'completion.json').exists() and read(B/'progress.json')['completed_fields']==81
    a=read(B/'analysis.json');f=read(F/'evaluation/analysis.json');fc=read(F/'completion.json');p=read(ROOT/'protocols/R2_P2G_frozen_physics.json')
    assert sha(F/'evaluation/analysis.json')==fc['files_sha256']['evaluation/analysis.json']
    rows=a['results'];summary=dict(diagnostic_analysis_sha256=sha(B/'analysis.json'),frozen_FEM_analysis_sha256=sha(F/'evaluation/analysis.json'),report_code_sha256=sha(__file__),comparisons={},changes={},formal_timing=False,training_runs=0)
    for case in p['cases']:
        for step in p['steps']:
            key=f'{case}_step{step:04d}';summary['comparisons'][key]={}
            for control in ['fourier_half','uniform_rbf_fourier']:
                entries=[]
                for i in range(2):
                    g=[rows[f'{case}_seed{s}_geometry_rbf_fourier_step{step:04d}'] for s in p['seeds']];c=[rows[f'{case}_seed{s}_{control}_step{step:04d}'] for s in p['seeds']]
                    metrics={}
                    getters=dict(total=lambda r:r['validation'][i]['total_mean'],far_wall=lambda r:r['regions'][i]['far_wall']['total_mean'],near_wall=lambda r:r['regions'][i]['near_wall']['total_mean'],median=lambda r:r['validation'][i]['norm_quantiles'][0],q95=lambda r:r['validation'][i]['norm_quantiles'][2])
                    for label,get in getters.items():
                        ratios=np.array([get(x)/get(y) for x,y in zip(g,c)])
                        metrics[label]=dict(paired_ratios=ratios.tolist(),geometric_mean_ratio=float(np.exp(np.log(ratios).mean())),wins=int((ratios<1).sum()))
                    entries.append(metrics)
                summary['comparisons'][key][control]=entries
        for seed in p['seeds']:
            for method in p['methods']:
                name=f'{case}_seed{seed}_{method}';r=[rows[name+f'_step{k:04d}'] for k in p['steps']]
                fem=[f['cases'][case][str(seed)][str(k)][method]['metrics'] for k in p['steps']]
                summary['changes'][name]=dict(steps=p['steps'],training_objective=[v['training_objective_recomputed'] for v in r],
                    independent_total=[[v['validation'][i]['total_mean'] for v in r] for i in range(2)],
                    independent_far=[[v['regions'][i]['far_wall']['total_mean'] for v in r] for i in range(2)],
                    boundary_traction=[v['independent_boundary']['traction'] for v in r],
                    fem_u_s_wall=[[v[k]['relative_l2_percent'] for k in ['u_area','s_area','wall_u']] for v in fem])
    (B/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# P2G: frozen-field physical diagnosis of budget-dependent behavior','',
      'All 81 registered P2F fields were evaluated on two common, geometry-only sets of 24000 independent solid points, the actual training points and dense independent boundary quadrature. This diagnostic performs no training, repairs, checkpoint selection or new FEM computation. All cases, methods, seeds and fixed budgets are retained. FEM values below come from the previously closed P2F evaluation and are used only to interpret frozen behavior.', '',
      '## What the evidence supports','',
      '**The deteriorating L1 Fourier trajectories fit their sampled equations increasingly well while developing highly localized physical-residual peaks outside the training points.** In seeds 260927 and 260928, from 800 to 1600 steps the sampled objective falls from 0.43683 to 0.05984 and from 0.79845 to 0.11597, respectively. Their area-U errors nevertheless rise from 12.857% to 22.906% and from 25.172% to 48.967%. Independent domain residual mean squares increase on BOTH fresh point sets. The independently checked boundary traction residuals decrease, so simple deterioration of boundary constraint satisfaction does not account for these observations.', '',
      '**Location and residual component distinguish these failures from a single global residual score.** At 1600 steps, 90.01–96.14% of residual-square mass in the first Fourier trajectory, and 99.55–99.86% in the second, lies more than 0.05 normalized units from the cavity wall. Equilibrium residuals dominate these large domain totals. The strongest audited locations include the upper outer-boundary strip in seed 260927 and an interior region near (0.05, 0.26) in seed 260928, not the cavity vertices. The separately matched uniform-feature control also develops a large interior peak in seed 260927. The phenomenon is therefore not unique to Fourier mapping.', '',
      '**The geometry rule shows a reproducible spatial benefit in this diagnostic.** For L1, its far-from-cavity residual mean square is lower than BOTH controls in every fresh seed, every registered budget and both independent sets. At 1600 steps its far-region means range from 0.100 to 0.797 across all six seed/set combinations; the two deteriorating Fourier trajectories range from 246.51 to 4172.72. Geometry-model high-residual mass is instead concentrated near the cavity: 94.22–96.25% of the total lies in the predeclared 0.02-distance vertex neighborhoods. These observations support reduced off-cavity residual peaks in the tested fields. They do not by themselves prove that the remaining corner behavior is physically correct or that the rule guarantees stability.', '',
      '**The geometry model still has a substantial residual-generalization limitation.** All three L1 geometry trajectories have increasing independent total residual mean squares as the budget grows from 400 to 1600, while their sampled losses fall. More than 99% of residual-square mass can be concentrated in the worst 1% of independent points. A geometry model can therefore have better field accuracy and median residuals while a small corner region dominates its squared residual integral. The third L1 Fourier seed remains competitive, and at 1600 has lower U/S/wall errors than geometry. The horizontal-convergence regressions reported in P2F are unchanged. This study supports a scoped reduction of harmful spatial artifacts, not complete removal of overfitting or universal superiority.', '',
      '**Numerical differentiation does not explain the audited peaks.** At eight largest-residual locations plus eight fixed random locations per field, float64 analytic derivatives agree with independent full autograd and independently assembled elasticity equations. The maximum scaled differences over all 81 fields are reported below. Recomputed training objectives also reproduce the saved trajectory losses within the registered tolerance. These checks address the audited computations, not the completeness of any finite sampling rule.', '',
      '## Mechanistic interpretation and limits','',
      'The discrete strong-form objective constrains residuals at the collocation coordinates. Decreasing that finite sum alone provides no bound on residual magnitude between the points. In the two deteriorating L1 trajectories, the combination of smaller sampled losses, increasing independent equilibrium residuals and narrow off-cavity peaks is consistent with under-resolved residual fitting. The geometry-informed representation changes the spatial allocation of features; relative to both matched controls, its repeated reduction of far-region residuals is evidence for a useful geometry-specific inductive bias.', '',
      'This is a numerical diagnosis and controlled representation comparison, not a causal intervention proving why a particular hidden-layer trajectory develops a peak. A prospective matched-sampling or resolution intervention would be needed to establish that mechanism more directly. No theorem of stability, Fourier-specific failure, exact error bound or novelty of generic Gaussian/Fourier features follows. The two independent estimates can differ greatly because the squared residual is concentrated in very small regions; the reported means are finite-sample estimates, not converged domain integrals.', '',
      'For the paper, the defensible argument is that a common geometry-based feature allocation improves finite-budget L1 field accuracy and reduces off-cavity residual artifacts at comparable parameter and GPU-allocation scale. This explains more than a single favorable accuracy number. It must retain the corner-residual limitation, C1/C8 behavior, third-seed ranking reversal, selected engineering-quantity trade-offs and the absence of formal timing. It does not restore attribution to the original anchor-bias relation.', '',
      '## Complete L1 trajectory evidence','',
      'Residual values are squared norms in the original normalized equation convention; U/S/wall values are FEM relative-L2 percentages. Each range retains the two independent point-set estimates, not an interval or confidence bound.', '',
      '| Seed | Method | Steps | Sampled objective | Independent domain mean square | Independent/training-uniform ratio | Independent traction mean sum | FEM U / S / wall, % |',
      '|---|---|---:|---:|---:|---:|---:|---|']
    for seed in p['seeds']:
        for method in p['methods']:
            for step in p['steps']:
                r=rows[f'L1_seed{seed}_{method}_step{step:04d}'];v=f['cases']['L1'][str(seed)][str(step)][method]['metrics']
                lines.append(f"| {seed} | {method} | {step} | {r['training_objective_recomputed']:.5g} | {rng([v['total_mean'] for v in r['validation']])} | {rng(r['independent_training_uniform_ratio'])} | {r['independent_boundary']['traction']:.5g} | "+' / '.join(f"{v[k]['relative_l2_percent']:.5g}" for k in ['u_area','s_area','wall_u'])+' |')
    lines += ['', '## Spatial localization on L1','',
      'Near/far masks use cavity-wall distance 0.05. Corner masks use cavity-vertex distance 0.02. Near and corner masks overlap and must not be added. The highest-1% share is a residual-square concentration statistic; lower concentration alone is not necessarily better.', '',
      '| Seed | Method | Steps | Far mean square | Near-wall residual mass, % | Corner residual mass, % | Highest-1% point mass, % |',
      '|---|---|---:|---:|---:|---:|---:|']
    for seed in p['seeds']:
        for method in p['methods']:
            for step in p['steps']:
                r=rows[f'L1_seed{seed}_{method}_step{step:04d}']
                vals=[rng([v['far_wall']['total_mean'] for v in r['regions']])]+[rng([100*v[k]['residual_square_mass'] for v in r['regions']]) for k in ['near_wall','corner']]+[rng([100*v['top_one_percent_mass'] for v in r['validation']])]
                lines.append(f'| {seed} | {method} | {step} | '+' | '.join(vals)+' |')
    lines += ['', '## All cases: paired physical comparisons','',
      'Entries show geometry/control geometric-mean ratios over the three training seeds and favorable pair counts. Separate rows preserve each independent point set. Below one favors geometry. These residual comparisons do not replace P2F field/engineering comparisons.', '',
      '| Case | Budget | Control | Independent set | Total residual ratio (wins) | Far-region ratio (wins) | Median residual ratio (wins) | 95th residual ratio (wins) |',
      '|---|---:|---|---:|---:|---:|---:|---:|']
    for key,controls in summary['comparisons'].items():
        case,step=key.split('_step')
        for control,sets in controls.items():
            for i,s in enumerate(sets):
                vals=[f"{s[k]['geometric_mean_ratio']:.5g} ({s[k]['wins']}/3)" for k in ['total','far_wall','median','q95']]
                lines.append(f'| {case} | {int(step)} | {control} | {i} | '+' | '.join(vals)+' |')
    lines += ['', '## Complete 81-field numerical and physical audit','',
      f"Maximum scaled float64 analytic/autograd residual difference: {max(v['numerical_audit']['float64_scaled_max_error'] for v in rows.values()):.6g}. Maximum saved-float32/float64 difference: {max(v['numerical_audit']['float32_scaled_max_error'] for v in rows.values()):.6g}. All 81 checks pass the registered 1e-9 and 3e-4 thresholds. All sampled objectives pass the registered reproduction tolerance.", '',
      '| Field | Train-uniform residual mean square | Independent mean square | Eq fraction of independent total | Independent bottom uy mean square | Gauge ux square |',
      '|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        vs=r['validation'];b=r['independent_boundary']['constraints']
        lines.append(f"| {name} | {r['training_uniform']['total_mean']:.5g} | {rng([v['total_mean'] for v in vs])} | {rng([sum(v['component_mean'][:2])/v['total_mean'] for v in vs])} | {b['bottom_uy']['total_mean']:.5g} | {b['gauge_ux']['total_mean']:.5g} |")
    lines += ['', '## Data and next decision','',
      '- '+link('Registered diagnostic protocol',ROOT/'protocols/R2_P2G_frozen_physics.json')+'.',
      '- '+link('Full physical analysis',B/'analysis.json')+' contains every residual component, distribution, fixed region, boundary constraint and numerical check for all 81 fields. The two geometry point files preserve independent coordinates, region masks, quadrature points/normals and weights.',
      '- Every `<case>_seed<seed>_<method>_step<budget>_residuals.npz` stores training/independent residuals, all boundary residuals and hotspot/random-point derivative-audit arrays; companion metrics JSON stores its hash. Checkpoint/source hashes link back to the frozen P2F inputs.',
      '- '+link('Paired physical comparisons and full budget sequences',B/'summary.json')+' and '+link('already frozen FEM evidence',F/'evaluation/analysis.json')+'.',
      '- '+link('Diagnostic code',ROOT/'code/diagnose_p2g.py')+' and '+link('report generator',Path(__file__))+'.', '',
      'Close this diagnostic without repair training or parameter changes. The next bounded study should test the proposed under-resolution mechanism through a common physics-only sampling intervention with matched work and reset controls, including the strong Fourier baseline. It must retain the original geometry construction and all unfavorable outcomes, and must be registered before any fit. A strengthened baseline is necessary to determine whether the geometry rule has an independent remaining benefit. T1/S1 and formal timing are separate subsequent decisions; original first-round files and all closed batches remain unchanged.']
    (ROOT/'P2G_Frozen_Physics_Report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('P2G_FULL_REPORT_AND_COMPARISONS_WRITTEN')
    for step in p['steps']:
        for control,sets in summary['comparisons'][f'L1_step{step:04d}'].items():print(step,control,'far',[v['far_wall'] for v in sets])
if __name__=='__main__':main()
