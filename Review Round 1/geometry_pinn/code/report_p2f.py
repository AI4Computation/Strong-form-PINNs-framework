"""Full multidimensional fresh-seed/budget report, generated from frozen data."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2F_repetition_budget';E=B/'evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def link(label,p):return f'[{label}](<{Path(p).resolve().as_posix()}>)'
def mean_sd(v):return f"{v['mean']:.5g} ± {v['sample_sd']:.3g}"
def pair(v):return 'undefined' if v['geometric_mean_ratio'] is None else f"{v['geometric_mean_ratio']:.4g} ({v['wins']}/3)"
def main():
    assert not (B/'completion.json').exists()
    p=read(ROOT/'protocols/R2_P2F_repetition_budget.json');a=read(E/'analysis.json');s=read(E/'summary.json');m=read(B/'manifest.json')
    lines=['# P2F: fresh-seed repetition and fixed-budget study','',
      'This study repeats the closed P2D shared geometry-feature construction with three new initialization and collocation seeds, comparing all three representations on C1, C8 and L1. All 27 trajectories and registered checkpoints were frozen before FEM evaluation. All 81 prescribed seed/case/method/budget endpoints are reported. No checkpoint, seed, feature width or method was changed after seeing the engineering results.', '',
      'This is evidence about the scope and repeatability of benefits, not a requirement to win every metric. Three seeds do not establish universal superiority, statistical significance, cross-shape transfer to T1/S1 or adequate engineering precision. Original anchoring attribution and formal timing are separate questions.', '',
      '## Findings and research decision','',
      '**The strongest repeatable benefit is the L1 finite-budget response.** At both registered 400- and 800-step budgets, geometry-Fourier has lower area displacement, area stress and wall displacement errors than BOTH controls in all three new seeds. At 800 steps its mean U/S/wall errors are 4.637/15.754/7.515%, versus 15.397/19.707/19.468% for complete half-Fourier and 20.487/22.696/26.118% for uniform-scale Gaussian-Fourier. Geometric-mean paired error reductions against half-Fourier are 66.83%, 19.39% and 56.89%, respectively. These paired reductions differ from reductions computed from the ratio of arithmetic means.', '',
      '**The displacement improvement extends over substantial physical regions.** At 400 steps, 90.91–99.62% of solid area has lower U error than half-Fourier; all sampled wall locations improve, corresponding to 100% of the sampled arc-length-weighted wall measure in each seed. At 800 steps the improved U area is 71.48–98.70%, and U MAE and all registered 50/90/95/99% quantiles improve in every pair against both controls. This is not only a change at a few outliers. At 800 steps, stress MAE, median, 90th and 95th quantiles also improve in all pairs, while the 99th-quantile improvement is not unanimous.', '',
      '**Engineering quantities limit the claim.** L1 vertical-convergence error improves against both controls in all three seeds at 400 and 800 steps. Horizontal-convergence error is worse than half-Fourier in all three seeds at 800 steps (geometric-mean ratio 1.682); its absolute values are retained in the full data. Broad displacement accuracy therefore does not imply that every relative movement between selected wall points is improved. At 1600 steps mean geometry U/S/wall errors are 2.578/15.166/8.603%; the still substantial stress and wall errors prevent an adequate-engineering-accuracy claim.', '',
      '**Higher-budget rankings are less uniform.** At 1600 steps geometry beats half-Fourier on U/S/wall in two seeds but loses on all three in seed 260929. Against uniform-scale features it wins U in two seeds and S/wall in only one. Mean geometry errors are lower, partly because competing trajectories have large errors in some seeds. Geometry has smaller observed between-seed error dispersion, but three seeds do not establish a general robustness probability. Two L1 half-Fourier trajectories have larger FEM errors at 1600 than at 800 steps even though their training objective decreases. This observation calls for independent physical-residual diagnosis; it is not evidence about DEM and does not yet prove the cause.', '',
      '**Work and resources provide a qualified positive result.** For the registered joint research target U≤5%, S≤20%, wall U≤10%, geometry reaches it in 3/3 L1 seeds, first observed at 800/1600/800 steps; half-Fourier reaches it in 1/3 (1600), and uniform-scale features in 2/3 (both 1600). The attained target persists at subsequent observed endpoints. Neither stricter joint target is reached on L1 by any method. All models have 110705 trainable parameters; geometry peak allocated GPU memory is 305.886–306.011 MiB versus 305.5625 MiB for half-Fourier, an approximately 0.11–0.15% increase. This supports better observed target attainment at essentially the same parameter and allocation scale, not a proven speedup, lower-memory method or complete-cost advantage.', '',
      '**C1 and C8 do not support a general gain.** C1 global-displacement improvement from the earlier single seed does not repeat consistently; at 800 steps geometry has worse area U than both controls in all three new seeds. C8 area S is worse than half-Fourier at every tested seed and budget, and wall U is worse at 800/1600 in every seed. Narrow favorable outcomes remain in the full tables but do not establish cross-shape performance superiority. The geometry rule is common and automatic; its broadly useful performance still needs separate testing.', '',
      '**Decision:** retain the candidate for its repeated finite-budget L1 spatial and target-attainment benefits, while keeping the C1/C8 and convergence limitations explicit. Close this batch without extra training, tuning or endpoint selection. Before expanding to T1/S1, audit the frozen L1 physical residuals at common independent points across all budgets and methods to distinguish training-point fitting from domain behavior and explain the late-budget divergence. Such a diagnostic needs a separate prospective protocol and cannot choose a FEM-best model. Formal time-to-accuracy measurement requires its own idle-machine session. This study establishes neither novelty of generic RBF/Fourier features nor independent original-anchoring benefit.', '',
      '## Protocol and completeness','',
      'New seeds: 260927, 260928 and 260929. The exploratory seed 260926 is excluded from every paired summary. Each run is one continuous L-BFGS trajectory with endpoints at 400, 800 and 1600 accepted steps, max_eval 2400, history 50 and unchanged loss weights. Six newly generated geometry/seed point sets follow the frozen common sampling rule. All methods within a pair share exact points and downstream initialization, with the same 110705 trainable parameters.', '',
      'All eighteen geometry/seed/representation preflight combinations passed independent objective/gradient and full-batch GPU checks. Runs use float32, TF32 disabled, deterministic algorithms and one GPU job at a time. Development runtimes are not fair time benchmarks. Legitimate early optimizer stops, if present, are reported as unchanged terminal states at later nominal endpoints, with actual work retained.', '',
      '## Main field and wall errors by budget','',
      'Values are arithmetic mean ± sample SD over the three new seeds, in relative-L2 percentage units. Area U/S use FEM quadrature; wall U uses arc-length weights. These three measures are shown together; the complete point/node, region and distribution families remain available below and in the machine-readable summary.', '',
      '| Case | Accepted-step budget | Representation | Area U, % | Area S, % | Wall U, % |',
      '|---|---:|---|---:|---:|---:|']
    for case in p['case_settings']:
        for step in p['evaluation_steps']:
            tag=f'{case}_step{step:04d}'
            for method in p['methods']:
                v=s['groups'][tag][method]
                lines.append(f"| {case} | {step} | {method} | {mean_sd(v['u_area.relative_l2_percent'])} | {mean_sd(v['s_area.relative_l2_percent'])} | {mean_sd(v['wall_u.relative_l2_percent'])} |")
    lines += ['', '## Paired mechanism and strong-baseline comparisons','',
      'Each entry gives the geometric mean of the three geometry/control error ratios, followed by the number of favorable paired seeds. Below one favors geometry. A ratio ≤0.9 with 3/3 favorable pairs marks a material, consistent signal within this small study for that stated outcome; it does not license a claim about other metrics. The same controls remain visible when a result is unfavorable.', '',
      '| Case | Budget | Control | Area U ratio (wins) | Area S ratio (wins) | Wall U ratio (wins) |',
      '|---|---:|---|---:|---:|---:|']
    for case in p['case_settings']:
        for step in p['evaluation_steps']:
            for control,v in s['paired'][f'{case}_step{step:04d}'].items():
                lines.append(f"| {case} | {step} | {control} | {pair(v['u_area.relative_l2_percent'])} | {pair(v['s_area.relative_l2_percent'])} | {pair(v['wall_u.relative_l2_percent'])} |")
    lines += ['', '## Distributional comparisons','',
      'The complete 50/90/95/99% absolute-error quantile family is retained. Entries are paired geometric-mean error ratios (favorable seeds / 3). A lower error-concentration fraction alone is not classified as better: the absolute error can decrease while its remaining concentration increases.', '',
      '| Case | Budget | Control | Field | MAE | Median | 90th | 95th | 99th |',
      '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for case in p['case_settings']:
        for step in p['evaluation_steps']:
            for control,v in s['paired'][f'{case}_step{step:04d}'].items():
                for field in ['u_area','s_area']:
                    values=[pair(v[field+'.'+k]) for k in ['vector_mae','q50','q90','q95','q99']]
                    lines.append(f'| {case} | {step} | {control} | {field} | '+' | '.join(values)+' |')
    lines += ['', '## Spatial extent and engineering quantities','',
      'Improved-area and improved-wall fractions compare error norms at identical coordinates. The displayed percentages are mean [minimum, maximum] across the three new seeds. Convergence errors are absolute differences from FEM in normalized study units; their ratios measure comparative error, not relative physical convergence.', '',
      '| Case | Budget | Control | U improved area | S improved area | Improved wall length | Vertical convergence error ratio | Horizontal convergence error ratio |',
      '|---|---:|---|---:|---:|---:|---:|---:|']
    for case in p['case_settings']:
        for step in p['evaluation_steps']:
            tag=f'{case}_step{step:04d}'
            for control,v in s['spatial'][tag].items():
                fmt=lambda z:f"{100*z['mean']:.2f} [{100*z['min']:.2f}, {100*z['max']:.2f}]%"
                ratios=s['paired'][tag][control]
                lines.append(f"| {case} | {step} | {control} | {fmt(v['u'])} | {fmt(v['s'])} | {fmt(v['wall_u'])} | {pair(ratios['convergence.vertical'])} | {pair(ratios['convergence.horizontal'])} |")
    lines += ['', '## Region-specific relative errors','',
      'Full-method averages at each budget, in relative-L2 percentages. Near-wall distance is ≤0.05; polygon corner distance is <0.02. Geometry-only masks are unchanged from the verified reference. The corner column is absent for circular cavities. Region denominators differ, so these percentages are not additive.', '',
      '| Case | Budget | Method | Near U | Far U | Near S | Far S | Corner U | Corner S |',
      '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for case in p['case_settings']:
        for step in p['evaluation_steps']:
            for method,v in s['groups'][f'{case}_step{step:04d}'].items():
                vals=[f"{v[k+'.relative_l2_percent']['mean']:.5g}" if k+'.relative_l2_percent' in v else '—' for k in ['u_near_wall','u_far_wall','s_near_wall','s_far_wall','u_corner','s_corner']]
                lines.append(f'| {case} | {step} | {method} | '+' | '.join(vals)+' |')
    lines += ['', '## Fixed research targets and computational work','',
      'The three registered joint targets are (area U%, area S%, wall U%) = (5,20,10), (2,10,5), (1,5,2). They are research comparison targets, not engineering acceptance limits. Each entry is the first observed registered nominal budget and the number of seeds reaching it by the maximum budget. Only 400/800/1600 were evaluated: no crossing is interpolated, and no time-to-accuracy inference is made.', '',
      '| Case | Method | Target U5/S20/W10, first observed budgets | Target U2/S10/W5 | Target U1/S5/W2 |',
      '|---|---|---|---|---|']
    for case in p['case_settings']:
        for method in p['methods']:
            cells=[]
            for i in range(3):
                entries=[s['target_attainment'][f'{case}_seed{seed}_{method}'][i] for seed in p['seeds']]
                cells.append(', '.join('not reached' if v['first_observed_nominal_step'] is None else str(v['first_observed_nominal_step'])+(' (later lost)' if not v['sustained_through_later_endpoints'] else '') for v in entries))
            lines.append(f'| {case} | {method} | '+' | '.join(cells)+' |')
    lines += ['', 'All models retain 110705 trainable parameters. The following resource table summarizes the complete trajectories, not a fair idle-machine time benchmark. Peak allocation includes the actual cached features, derivatives and optimizer history. Exact counts and stage durations remain in result.json/trace.json.', '',
      '| Case | Method | Actual terminal steps, range | Closure evaluations, range | Peak allocated GPU MiB, range |',
      '|---|---|---:|---:|---:|']
    for case in p['case_settings']:
        for method in p['methods']:
            rows=[read(B/f'{case}_seed{seed}_{method}/result.json') for seed in p['seeds']]
            span=lambda key:f"{min(v[key] for v in rows)}–{max(v[key] for v in rows)}"
            mem=[v['peak_allocated_bytes']/2**20 for v in rows]
            lines.append(f"| {case} | {method} | {span('accepted_steps')} | {span('closure_evaluations')} | {min(mem):.3f}–{max(mem):.3f} |")
    lines += ['', '## Individual seed results','',
      'These are the complete 81 original records, not selected best cases. Reference data and metric conventions match the preceding tables.', '',
      '| Case | Seed | Nominal step | Method | Actual step | Area U, % | Area S, % | Wall U, % |',
      '|---|---:|---:|---|---:|---:|---:|---:|']
    for case in p['case_settings']:
        for seed in p['seeds']:
            for step in p['evaluation_steps']:
                for method in p['methods']:
                    row=a['cases'][case][str(seed)][str(step)][method];v=row['metrics']
                    lines.append(f"| {case} | {seed} | {step} | {method} | {row['actual_step']} | {v['u_area']['relative_l2_percent']:.6g} | {v['s_area']['relative_l2_percent']:.6g} | {v['wall_u']['relative_l2_percent']:.6g} |")
    lines += ['', '## Files, provenance and claim boundaries','',
      '- '+link('Frozen protocol',ROOT/'protocols/R2_P2F_repetition_budget.json')+' and '+link('preflight',B/'preflight.json')+'.',
      '- '+link('Training manifest',B/'manifest.json')+' contains source, input and checkpoint hashes. Each `<case>_seed<seed>_<method>` directory contains all reached checkpoints, terminal weights, full trace and actual work/resource records.',
      '- '+link('Complete FEM evaluation',E/'analysis.json')+' gives all field/reference paths and hashes, all 81 metric sets and paired spatial fractions. Each endpoint has a full prediction array and U/S/wall error norms. Nine per-seed/budget method groups per case are separately stored as metrics JSON.',
      '- '+link('Full metric-family summary',E/'summary.json')+' includes each raw paired-seed ratio, means, sample SDs, ranges, favorable-pair counts, error-concentration summaries and joint-target attainment with persistence. No pooled mean across different cases or weighting conventions is used.',
      '- '+link('Evaluator',ROOT/'code/evaluate_p2f.py')+', '+link('aggregation code',ROOT/'code/summarize_p2f.py')+' and '+link('report generator',Path(__file__))+'.', '',
      'No first-round file, earlier closed batch, manuscript or response letter was edited. The old original-bandwidth C1 advantage remains a separate result; these Fourier/geometry comparisons do not establish independent original-anchoring benefit. Any further T1/S1 or timing study needs its own protocol; no such expansion is included here.']
    out=ROOT/'P2F_Repetition_Budget_Report.md';out.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(out)
if __name__=='__main__':main()
