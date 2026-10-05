"""Full paired intervention summaries and English report; no model selection."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2H_sampling_intervention';E=B/'evaluation';F=ROOT/'results/R2_P2F_repetition_budget'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def link(label,p):return f'[{label}](<{Path(p).resolve().as_posix()}>)'
def stats(x):
    v=np.array(x,dtype=float);return dict(values=v.tolist(),mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),min=float(v.min()),max=float(v.max()))
def flat(r):
    out={}
    for field,values in r['metrics'].items():
        if field=='engineering':continue
        for key in ['relative_l2_percent','vector_mae','max_vector_absolute_error']:out[field+'.'+key]=values[key]
        for label,val in zip(['q50','q90','q95','q99'],values['vector_absolute_error_quantiles']):out[field+'.'+label]=val
    for label,v in zip(['vertical','horizontal'],r['metrics']['engineering']['convergence_absolute_error']):out['convergence.'+label]=v
    for i in range(2):
        out[f'physics.set{i}.total']=r['validation'][i]['total_mean']
        out[f'physics.set{i}.far']=r['regions'][i]['far_wall']['total_mean']
        out[f'physics.set{i}.near']=r['regions'][i]['near_wall']['total_mean']
        for label,v in zip(['q50','q90','q95','q99','max'],r['validation'][i]['norm_quantiles']):out[f'physics.set{i}.'+label]=v
    return out
def paired(x,y):
    x=np.asarray(x);y=np.asarray(y);r=x/y if np.all(y>0) else None
    return dict(numerator=x.tolist(),denominator=y.tolist(),paired_ratios=None if r is None else r.tolist(),geometric_mean_ratio=None if r is None or np.any(r<=0) else float(np.exp(np.log(r).mean())),wins=int((x<y).sum()))
def cell(x):return f"{x['mean']:.5g} ± {x['sample_sd']:.3g}"
def ratio(x):return 'undefined' if x['geometric_mean_ratio'] is None else f"{x['geometric_mean_ratio']:.4g} ({x['wins']}/3)"
def main():
    assert not (B/'completion.json').exists();a=read(E/'analysis.json');p=read(ROOT/'protocols/R2_P2H_sampling_intervention.json');assert len(a['results'])==54
    arms=['continuous']+p['arms'];rows={(v['seed'],v['step'],v['method'],v['arm']):v for v in a['results'].values()}
    s=dict(analysis_sha256=sha(E/'analysis.json'),report_source_sha256=sha(__file__),n_seeds=3,groups={},intervention={},representation={},targets={},formal_timing=False)
    for step in p['evaluation_steps']:
        for method in p['methods']:
            for arm in arms:
                fs=[flat(rows[seed,step,method,arm]) for seed in p['seeds']];keys=list(fs[0]);s['groups'][f'{step}_{method}_{arm}']={k:stats([r[k] for r in fs]) for k in keys}
            for numerator,denominator in [('uniform_refresh','fixed_reset'),('fixed_reset','continuous'),('uniform_refresh','continuous')]:
                x=[flat(rows[seed,step,method,numerator]) for seed in p['seeds']];y=[flat(rows[seed,step,method,denominator]) for seed in p['seeds']]
                s['intervention'][f'{step}_{method}_{numerator}_over_{denominator}']={k:paired([v[k] for v in x],[v[k] for v in y]) for k in x[0]}
        for arm in arms:
            for control in ['fourier_half','uniform_rbf_fourier']:
                x=[flat(rows[seed,step,'geometry_rbf_fourier',arm]) for seed in p['seeds']];y=[flat(rows[seed,step,control,arm]) for seed in p['seeds']]
                s['representation'][f'{step}_{arm}_geometry_over_{control}']={k:paired([v[k] for v in x],[v[k] for v in y]) for k in x[0]}
    for seed in p['seeds']:
        for method in p['methods']:
            for arm in arms:
                entries=[]
                for target in [(5,20,10),(2,10,5),(1,5,2)]:
                    met=[all(rows[seed,step,method,arm]['metrics'][k]['relative_l2_percent']<=limit for k,limit in zip(['u_area','s_area','wall_u'],target)) for step in p['evaluation_steps']]
                    first=met.index(True) if any(met) else None
                    entries.append(dict(target=list(target),met_at_800_1600=met,first_observed_step=None if first is None else p['evaluation_steps'][first],sustained=None if first is None else all(met[first:])))
                s['targets'][f'{seed}_{method}_{arm}']=entries
    (E/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    lines=['# P2H: paired sampling intervention in the shared strong-form PINN','',
      'The experiment branches from all nine frozen L1 P2F 400-step states. Each state receives fixed-point continuation or uniform-interior refresh, both with L-BFGS restarted every 400 steps. All 18 continuations reach cumulative step 1600. All trajectories freeze before FEM access. All 36 registered new endpoints at 800/1600 are evaluated alongside 18 continuous-training reference endpoints, using identical independent physical points and the same verified FEM reference.', '',
      '## Findings and decision','',
      '**The intervention supports a causal contribution of fixed collocation to the two severe Fourier deteriorations.** At 1600 steps, refreshing points reduces U error from 21.815% to 3.203% in seed 260927 and from 42.489% to 6.536% in seed 260928, relative to the fixed-point arm with the SAME optimizer resets. Their independent domain and far-region residuals also decrease markedly. Resetting history alone does not remove the severe error growth. Across all three seeds, refreshed Fourier has lower stress, wall-displacement and independent residual measures than fixed/reset; global U improves in two seeds and worsens in the already competitive third seed. The full seed record prevents this mechanism finding from becoming an unconditional statement about Fourier instability.', '',
      '**Geometry retains a narrower early-budget advantage after the baseline receives the same refresh.** At 800 steps, geometry U/S/wall mean errors are 5.569/16.503/7.759%, compared with 7.388/17.432/9.490% for half-Fourier and 11.878/18.260/15.653% for uniform-scale features. Paired geometric-mean reductions against half-Fourier are 26.68/5.35/16.41%; the favorable counts are 2/3, 3/3 and 2/3. Geometry stress MAE and the 95th stress-error percentile improve against half-Fourier in all three seeds (paired reductions 10.52% and 12.27%). Far-region residual mean squares are lower than BOTH controls in every seed and both independent sets at this budget. This is useful scoped evidence, not full metric dominance.', '',
      '**The geometry/control attribution is strongest against the matched uniform-scale representation at 800 steps.** With both models refreshed, geometry U and wall errors improve in 3/3 seeds, with paired geometric-mean ratios 0.5168 and 0.5636. Its stress MAE, median and 95th percentile also improve in 3/3. However, the strong complete Fourier baseline is much closer than this feature-family control. The independent value of the geometry rule must be judged against both.', '',
      '**At 1600 steps the strong-baseline gap largely narrows.** Refreshed geometry and refreshed half-Fourier mean U/S/wall errors are 4.096/15.089/7.421% and 4.411/15.548/7.614%; each measure favors geometry in only two seeds, with paired ratios approximately 0.96–0.97. Geometry total independent residual is higher than refreshed half-Fourier in two seeds on set 0 and all three on set 1, even though some field errors are lower. Its vertical-convergence error has a paired geometric-mean ratio of 2.193 against half-Fourier and is better in only one seed. A large universal final-accuracy or stability advantage is not supported.', '',
      '**Refreshing is not uniformly beneficial to the geometry model.** Against its own fixed/reset arm, refreshed geometry has lower independent total residual in every seed, but higher U error in two seeds at 1600 (paired ratio 1.593). Wall error improves in two seeds; stress improves slightly in all three. Reducing a residual statistic cannot substitute for the full field and engineering evaluation. The original continuous geometry results remain unchanged and should not be relabeled as an adaptive-sampling result.', '',
      '**The previous 3/3 versus 1/3 joint-target advantage does not survive unchanged after strengthening the sampling baseline.** Under refresh the U≤5%, S≤20%, wall U≤10% target is reached by 2/3 geometry, 2/3 half-Fourier and 3/3 uniform-scale trajectories. Geometry reaches it at 800 in seed 260929 versus Fourier at 1600; other seeds and both stricter targets remain in the table. No stronger target or favorable checkpoint is chosen after inspection.', '',
      '**Decision:** close the mechanism experiment without more L1 tuning. The evidence now supports a qualified story about spatial allocation, finite-budget field/stress-distribution benefits and collocation generalization. It does not support depicting the strong baseline as irreparably unstable. The next bounded question should be transfer of the unchanged common geometry rule to T1/S1 with both fixed and refreshed strong controls, after reference/preflight checks and a new protocol. Keep the original fixed-sampling candidate and the common-refresh arm distinct; do not select a method or checkpoint by FEM for each geometry. Performance transfer and novelty remain unestablished, and formal timing is still pending.', '',
      '## Protocol and interpretation','',
      'The three representations have 110705 trainable parameters, the same loss weights and 6000 domain collocation points. Within each seed and refresh block the point set is identical across representations. Only uniform interior points change; the fixed probes, boundary points and weights remain unchanged. Both continuation arms use the same restart schedule and inherited weights. The original continuous trajectories separate the effect of restarting optimizer history from changing collocation.', '',
      'The two independent physical sets each contain 24000 fresh solid points, shared across all 54 fields and never used in training. Physical metrics include all five residual components, fixed near/far/corner regions, full quantiles and dense boundary quadrature. The complete P2F FEM metric family is retained. The two point-set estimates remain separate. Three seeds are not a general reliability probability or a significance test.', '',
      'Equal accepted-step and collocation budgets are not equal closure counts or complete runtime. Per-block closures, reused-prefix work, preparation, point-generation telemetry and peak allocated/reserved GPU memory are retained. There is no formal timing experiment and no speed claim. The saved 1200-step states are not prespecified evaluation endpoints and were not searched for favorable FEM results.', '',
      '## Field accuracy: all representations and arms','',
      'Mean ± sample SD over three paired seeds; relative-L2 percentages. The continuous rows reproduce frozen P2F FEM metrics exactly.', '',
      '| Budget | Representation | Sampling/history arm | Area U, % | Area S, % | Wall U, % |',
      '|---:|---|---|---:|---:|---:|']
    for step in p['evaluation_steps']:
        for method in p['methods']:
            for arm in arms:
                v=s['groups'][f'{step}_{method}_{arm}'];lines.append(f'| {step} | {method} | {arm} | '+' | '.join(cell(v[k+'.relative_l2_percent']) for k in ['u_area','s_area','wall_u'])+' |')
    lines+=['','## Sampling and restart effects','','Entries give paired geometric-mean numerator/denominator ratios (favorable seeds / 3). Values below one favor the numerator. Independent residual sets are reported separately.','',
      '| Comparison | U | S | Wall U | Domain residual set 0 | Domain residual set 1 | Far residual set 0 | Far residual set 1 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    mainkeys=['u_area.relative_l2_percent','s_area.relative_l2_percent','wall_u.relative_l2_percent','physics.set0.total','physics.set1.total','physics.set0.far','physics.set1.far']
    for name,v in s['intervention'].items():lines.append('| '+name+' | '+' | '.join(ratio(v[k]) for k in mainkeys)+' |')
    lines+=['','## Geometry benefit under each common sampling rule','','Both feature-family/capacity control and complete half-bandwidth Fourier remain visible after receiving the same intervention.','',
      '| Geometry/control comparison | U | S | Wall U | Domain residual set 0 | Domain residual set 1 | Far residual set 0 | Far residual set 1 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in s['representation'].items():lines.append('| '+name+' | '+' | '.join(ratio(v[k]) for k in mainkeys)+' |')
    lines+=['','## Engineering and distributional trade-offs','','These are geometry/control ratios, not selected winning metrics. Full absolute values and point/node/region metrics are in summary.json.','',
      '| Comparison | U median | U 95th | U 99th | S median | S 95th | S 99th | Vertical convergence | Horizontal convergence |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in s['representation'].items():lines.append('| '+name+' | '+' | '.join(ratio(v[k]) for k in ['u_area.q50','u_area.q95','u_area.q99','s_area.q50','s_area.q95','s_area.q99','convergence.vertical','convergence.horizontal'])+' |')
    lines+=['','## Joint research targets and work','','Targets (U%,S%,wall U%) are research comparison thresholds, not engineering acceptance limits. The first observed 800/1600 endpoint is listed in seed order 260927/260928/260929; no crossing is interpolated.','',
      '| Method | Arm | U5/S20/W10 | U2/S10/W5 | U1/S5/W2 | Cumulative closures at 1600, range | Peak allocated MiB, range |',
      '|---|---|---|---|---|---:|---:|']
    for method in p['methods']:
        for arm in arms:
            cells=[]
            for i in range(3):
                target=[s['targets'][f'{seed}_{method}_{arm}'][i] for seed in p['seeds']]
                cells.append(', '.join('not reached' if v['first_observed_step'] is None else str(v['first_observed_step'])+(' (later lost)' if not v['sustained'] else '') for v in target))
            cr=[rows[seed,1600,method,arm]['cumulative_closures'] for seed in p['seeds']]
            rr=[read((F/f'L1_seed{seed}_{method}' if arm=='continuous' else B/f'L1_seed{seed}_{method}_{arm}')/'result.json') for seed in p['seeds']];mem=[v['peak_allocated_bytes']/2**20 for v in rr]
            lines.append(f'| {method} | {arm} | '+' | '.join(cells)+f' | {min(cr)}–{max(cr)} | {min(mem):.3f}–{max(mem):.3f} |')
    lines+=['','## All 54 individual endpoint records','','Raw independent residuals are squared norms of the normalized mixed equations; their relation to field error must be assessed, not assumed.','',
      '| Seed | Step | Method | Arm | U, % | S, % | Wall U, % | Independent residual mean square (set 0 / 1) | Far residual mean square (set 0 / 1) |',
      '|---:|---:|---|---|---:|---:|---:|---|---|']
    for r in a['results'].values():
        vals=[f"{r['metrics'][k]['relative_l2_percent']:.6g}" for k in ['u_area','s_area','wall_u']]
        vals+=[' / '.join(f"{v['total_mean']:.5g}" for v in r['validation']),' / '.join(f"{v['far_wall']['total_mean']:.5g}" for v in r['regions'])]
        lines.append(f"| {r['seed']} | {r['step']} | {r['method']} | {r['arm']} | "+' | '.join(vals)+' |')
    lines+=['','## Source data and boundaries','',
      '- '+link('Frozen prospective protocol',ROOT/'protocols/R2_P2H_sampling_intervention.json')+', '+link('preflight',B/'preflight.json')+' and '+link('training manifest',B/'manifest.json')+'. Nine refresh point files and all 18 continuation directories contain exact inputs, block checkpoints, traces, source hashes, actual work and resource records.',
      '- '+link('Complete evaluation',E/'analysis.json')+' contains all 54 physical/FEM metric sets, raw-array/checkpoint/reference paths and hashes, full pairwise improved-area/wall fractions, and derivative audits. Each new endpoint has a full prediction file; continuous prediction files are reused by hash. Every endpoint has raw residuals and error norms.',
      '- '+link('Complete paired summaries',E/'summary.json')+' preserves full error/distribution/region families, all paired ratios and research-target attainment. The 54 numerical audits and original/active training-objective reproduction checks are required, not optional.',
      '- '+link('Evaluation code',ROOT/'code/evaluate_p2h.py')+' and '+link('report generator',Path(__file__))+'.', '',
      'This batch tests only geometry-independent uniform interior refresh with the registered block/reset schedule. It does not test all adaptive sampling, establish universal stability, validate T1/S1, supply fair time-to-accuracy, or restore original anchoring attribution. All prior batches and first-round materials are preserved.']
    (ROOT/'P2H_Sampling_Intervention_Report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('P2H_REPORT_AND_FULL_SUMMARY_WRITTEN')
    for name,v in s['intervention'].items():
        if name.startswith('1600'):print(name,{k:(v[k]['geometric_mean_ratio'],v[k]['wins']) for k in mainkeys})
    for name,v in s['representation'].items():
        if 'uniform_refresh' in name:print(name,{k:(v[k]['geometric_mean_ratio'],v[k]['wins']) for k in mainkeys})
if __name__=='__main__':main()
