"""Join completed exact timing replays to frozen accuracy, never select training."""
import json,hashlib,math,statistics,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2L_formal_cost';K=ROOT/'results/R2_P2K_transfer_replication'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def stats(v):return dict(values=v,mean=statistics.mean(v),median=statistics.median(v),minimum=min(v),maximum=max(v))
def gm(v):return math.exp(statistics.mean(map(math.log,v)))
def main():
    m=read(B/'manifest.json');p=read(ROOT/'protocols/R2_P2L_formal_cost.json');assert m['status']=='timing_complete' and len(m['completed'])==18
    assert sha(ROOT/'protocols/R2_P2L_formal_cost.json')==m['protocol_sha256']
    frozen=read(K/'completion.json');assert sha(K/'completion.json')==p['P2K_completion_sha256'];assert sha(K/'summary.json')==frozen['files_sha256']['summary.json'];accuracy=read(K/'summary.json')
    for path,h in accuracy['source_analysis_sha256'].items():assert sha(path)==h
    for f,h in m['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    rows={};runs={};targets={};groups={};contrasts={};MIB=2**20
    for rr in m['run_records']:
        folder=B/rr['folder'];r=read(folder/'result.json');assert r['formal_timing'] and r['all_ac_samples_on'] and all(r['exact_replay'].values());name=r['name'];assert name not in runs
        assert r['accepted_steps']==1600 and r['parameters']==110705 and len(r['blocks'])==4
        elapsed=[v['end_to_end_seconds'] for v in r['blocks']];assert all(a<b for a,b in zip([0]+elapsed,elapsed))
        runs[name]=dict(r,process_envelope_seconds=rr['process_envelope_seconds'],result_path=str((folder/'result.json').resolve()),result_sha256=sha(folder/'result.json'),formal_index=rr['index'])
        for step in [800,1600]:
            tag=name+f'_step{step:04d}';old=accuracy['results'][tag];assert sha(old['checkpoint_path'])==old['checkpoint_sha256'];block=r['blocks'][step//400-1]
            rows[tag]=dict(case=r['case'],seed=r['seed'],method=r['method'],step=step,elapsed_seconds=block['end_to_end_seconds'],optimization_seconds=sum(v['optimization_seconds'] for v in r['blocks'][:step//400]),metrics=old['metrics'],source_analysis=old['source_analysis'],source_checkpoint=old['checkpoint_path'],source_checkpoint_sha256=old['checkpoint_sha256'],timed_checkpoint=str((folder/f'step_{step:04d}.pt').resolve()),timed_checkpoint_sha256=sha(folder/f'step_{step:04d}.pt'),run=name)
    def name(c,s,method):return f'{c}_seed{s}_{method}_uniform_refresh'
    def row(c,s,method,step):return rows[name(c,s,method)+f'_step{step:04d}']
    for c in p['cases']:
     for method in p['methods']:
        rs=[runs[name(c,s,method)] for s in p['seeds']]
        groups[c+'_'+method]={key:stats([float(func(r)) for r in rs]) for key,func in dict(seconds800=lambda r:r['blocks'][1]['end_to_end_seconds'],seconds1600=lambda r:r['blocks'][3]['end_to_end_seconds'],common_geometry_seconds=lambda r:r['components']['common_geometry_seconds'],initial_points_seconds=lambda r:r['components']['initial_points_seconds'],representation_seconds=lambda r:r['components']['representation_initialization_seconds'],optimization_seconds=lambda r:sum(x['optimization_seconds'] for x in r['blocks']),refresh_seconds=lambda r:sum(x['refresh_generation_seconds'] for x in r['blocks']),cache_seconds=lambda r:sum(x['cache_preparation_seconds'] for x in r['blocks']),optimizer_setup_seconds=lambda r:sum(x['optimizer_setup_seconds'] for x in r['blocks']),io_seconds=lambda r:sum(x['checkpoint_trace_io_seconds'] for x in r['blocks']),gpu_allocated_mib=lambda r:r['peak_allocated_bytes']/MIB,gpu_reserved_mib=lambda r:r['peak_reserved_bytes']/MIB,rss_mib=lambda r:r['peak_sampled_rss_bytes']/MIB,rss_baseline_mib=lambda r:r['baseline_rss_bytes']/MIB,inference_ms=lambda r:1000*r['query_inference_seconds'],query_generation_ms=lambda r:1000*r['query_generation_seconds'],warmup_seconds=lambda r:r['warmup_seconds'],envelope_seconds=lambda r:r['process_envelope_seconds']).items()}
        for target in p['targets']:
            byseed={}
            for s in p['seeds']:
                hits=[all(row(c,s,method,step)['metrics'][k]['relative_l2_percent']<=t for k,t in zip(['u_area','s_area','wall_u'],target)) for step in [800,1600]]
                first=next((step for step,hit in zip([800,1600],hits) if hit),None)
                byseed[str(s)]=dict(hits=hits,first_observed_step=first,first_observed_seconds=row(c,s,method,first)['elapsed_seconds'] if first else None,censor_seconds=row(c,s,method,1600)['elapsed_seconds'] if first is None else None,lost_at1600=hits[0] and not hits[1])
            targets[f'{c}_{method}_{target}']=dict(case=c,method=method,target=target,by_seed=byseed,ever_count=sum(x['first_observed_step'] is not None for x in byseed.values()),at1600_count=sum(x['hits'][1] for x in byseed.values()))
     for control in ['fourier_half','uniform_rbf_fourier']:
      for step in [800,1600]:
        ratios=[row(c,s,'geometry_rbf_fourier',step)['elapsed_seconds']/row(c,s,control,step)['elapsed_seconds'] for s in p['seeds']]
        entry=dict(elapsed_ratios=ratios,geometric_mean=gm(ratios),geometry_faster_count=sum(x<1 for x in ratios),errors={})
        for field in ['u_area','s_area','wall_u']:
         for metric in ['relative_l2_percent','vector_mae','p95']:
            def value(r):return r['metrics'][field]['vector_absolute_error_quantiles'][2] if metric=='p95' else r['metrics'][field][metric]
            er=[value(row(c,s,'geometry_rbf_fourier',step))/value(row(c,s,control,step)) for s in p['seeds']];entry['errors'][field+'.'+metric]=dict(ratios=er,geometric_mean=gm(er),lower_error_count=sum(x<1 for x in er),lower_error_and_time_count=sum(x<1 and y<1 for x,y in zip(er,ratios)))
        contrasts[f'{c}_{step}_{control}']=entry
    target_pairs={}
    for c in p['cases']:
     for threshold in p['targets']:
      for control in ['fourier_half','uniform_rbf_fourier']:
        g=targets[f'{c}_geometry_rbf_fourier_{threshold}'];v=targets[f'{c}_{control}_{threshold}']
        gs=[seed for seed,q in g['by_seed'].items() if q['first_observed_seconds'] is not None];cs=[seed for seed,q in v['by_seed'].items() if q['first_observed_seconds'] is not None]
        ratios=[g['by_seed'][seed]['first_observed_seconds']/v['by_seed'][seed]['first_observed_seconds'] for seed in gs] if gs==cs and gs else None
        target_pairs[f'{c}_{threshold}_{control}']=dict(case=c,target=threshold,control=control,geometry_attaining_seeds=gs,control_attaining_seeds=cs,same_success_set=gs==cs,paired_ratios_on_identical_success_set=ratios,conditional_geometric_mean=gm(ratios) if ratios else None,geometry_retained_at1600=g['at1600_count'],control_retained_at1600=v['at1600_count'])
    telemetry=list(csv.DictReader((B/'gpu_telemetry.csv').open(encoding='utf-8')))
    numeric={}
    for key in telemetry[0]:
        vals=[]
        for line in telemetry:
            try:vals.append(float(line[key].strip().split()[0]))
            except ValueError:pass
        if vals:numeric[key.strip()]=dict(minimum=min(vals),maximum=max(vals),mean=statistics.mean(vals))
    summary=dict(protocol_sha256=m['protocol_sha256'],timing_manifest_sha256=sha(B/'manifest.json'),accuracy_summary_sha256=sha(K/'summary.json'),formal_replays=18,new_scientific_conditions=0,accuracy_endpoints=36,runs=runs,rows=rows,groups=groups,contrasts=contrasts,targets=targets,target_pairs=target_pairs,telemetry_summary=numeric,all_idle_gates_pass=all(x['passed'] for x in m['gate_records']),source_analysis_sha256=accuracy['source_analysis_sha256'])
    write(B/'summary.json',summary)
    F={'fourier_half':'F','geometry_rbf_fourier':'G','uniform_rbf_fourier':'U'}
    lines=['# P2L: formal cost replay of frozen cross-shape PINN comparisons','',
      'The strongest supported cost–accuracy result is a modest measured training-time premium for the replicated S1 stress-distribution benefit, not a general acceleration or memory saving. At1600, G averages36.100s versus35.543s for F (paired time GM ratio1.0157), while stress MAE/p95 improve by11.22%/12.32%. G’s active CUDA allocation is almost unchanged, but peak allocator-reserved memory rises from348MiB for F to396–398MiB for G. These costs and the adverse displacement/wall-distribution findings belong in the same engineering interpretation.','',
      'This batch measures 18 serial executions of the existing T1/S1 common-refresh trajectories: three methods and three seeds per geometry. It adds no scientific conditions. All four checkpoints per run exactly reproduce the frozen tensors (72 exact state comparisons); existing FEM accuracy is joined only after all timed runs finish. Seed 260930 is the discovery seed; 260931/260932 are the two prior prospective replications. Three-seed statistics describe these runs, not population-level significance.','',
      'F = half-bandwidth Fourier, G = geometry Gaussian–Fourier, U = equal-capacity uniform Gaussian–Fourier. All are fully trainable strong-form mixed PINNs with 110705 trainable parameters and the same common-refresh point stream. The uniform feature control isolates the geometry rule within the combined representation; it does not establish an independent benefit of the original anchoring method.','',
      '## Timing and resource boundary','',
      'The author explicitly confirmed an idle computer and received a separate formal-timing notice. Each job uses a fresh Python process, two CPU threads, deterministic CUDA and TF32 disabled. Six method permutations balance order position, with an eight-second inter-run rest. Every pre-run gate requires AC power, system CPU <=15% over two seconds and GPU utilization <=10%. Desktop processes are not killed. GPU telemetry is recorded every two seconds and worker RSS/CPU/AC every 0.5 seconds.','',
      'The primary warmed-process elapsed includes reconstructing the geometry and common support cover, initial and refreshed point generation, method representation and network construction, cache preparation, optimizer setup, all four 400-step training blocks, checkpoint and trace I/O, and lightweight monitoring. Common geometry/support preparation is required by this controlled protocol and charged to every method; this is not the cheapest standalone Fourier pipeline. Component times are reported separately.','',
      'Python/CUDA startup and 50 identical disposable full-objective backward warmups are excluded from primary solve time and reported separately. Parent process-envelope time also includes startup, warmup, output query inference, exact replay verification and process shutdown, so it must not be interpreted as cold-start solve time alone. Post-training inference uses 10000 identical reference-free solid queries per geometry. No FEM evaluation, FEM-selected early stop or tuning occurs during timing.','',
      'Hardware: NVIDIA GeForce RTX5080 Laptop GPU, driver610.78, Windows11, Python3.12.7, 24 logical CPUs and approximately63.67GiB RAM; only two CPU threads are assigned to the numerical libraries. Idle-gate system CPU is1.6–4.6%; recorded GPU temperatures span41–55°C including rests. Full runtime metadata and telemetry remain in the source files.','',
      'Torch peak allocated/reserved memory measures this process’s CUDA allocation, not total device usage. Host RSS is a sampled peak, not an exact peak. WDDM does not provide reliable per-process attribution for every desktop application. Balanced ordering and idle gates reduce interference but do not create laboratory clock/thermal isolation or establish statistical equivalence of timings. Millisecond-level inference measurements are single batched queries per seed and are descriptive.','',
      '## Complete elapsed times, accuracy and memory','',
      '| Case | Seed | Model | t800 (s) | t1600 (s) | U/S/wall at800 (%) | U/S/wall at1600 (%) | GPU allocated/reserved (MiB) | Peak sampled RSS (MiB) |','|---|---:|---|---:|---:|---|---|---|---:|']
    for c in p['cases']:
     for s in p['seeds']:
      for method in p['methods']:
        r=runs[name(c,s,method)];a,b=[row(c,s,method,step) for step in [800,1600]]
        errors=[' / '.join(f"{v['metrics'][k]['relative_l2_percent']:.4f}" for k in ['u_area','s_area','wall_u']) for v in [a,b]]
        lines.append(f"| {c} | {s} | {F[method]} | {a['elapsed_seconds']:.3f} | {b['elapsed_seconds']:.3f} | {errors[0]} | {errors[1]} | {r['peak_allocated_bytes']/MIB:.3f} / {r['peak_reserved_bytes']/MIB:.3f} | {r['peak_sampled_rss_bytes']/MIB:.2f} |")
    lines+=['','## Cost decomposition (three-seed means)','','All entries are seconds except the last inference column. Checkpoint elapsed is primary; component totals do not include every small assertion, trace or Python operation.','','| Case | Model | Geometry | Initial points | Representation | Refresh | Caches | Optimizer setup | Optimization | I/O | Total1600 | Warmup | Process envelope | Inference10k (ms) |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in p['cases']:
     for method in p['methods']:
        g=groups[c+'_'+method];keys=['common_geometry_seconds','initial_points_seconds','representation_seconds','refresh_seconds','cache_seconds','optimizer_setup_seconds','optimization_seconds','io_seconds','seconds1600','warmup_seconds','envelope_seconds','inference_ms'];lines.append(f'| {c} | {F[method]} | '+' | '.join(f"{g[k]['mean']:.4f}" for k in keys)+' |')
    lines+=['','## Paired cost and error contrasts','','Ratios below one favor G. GM is the geometric mean of the three paired ratios. Equal step counts are not equal elapsed times. A lower error with a longer elapsed time is a trade-off, not a speedup. Full MAE/p95/wall and paired cost contrasts are retained in `summary.json`.','','| Case | Step | Control | Time ratios (three seeds) | Time GM | U GM | Stress GM | Wall GM | Faster count |','|---|---:|---|---|---:|---:|---:|---:|---:|']
    for c in p['cases']:
     for step in [800,1600]:
      for control in ['fourier_half','uniform_rbf_fourier']:
        q=contrasts[f'{c}_{step}_{control}'];lines.append(f"| {c} | {step} | {F[control]} | "+' / '.join(f'{v:.4f}' for v in q['elapsed_ratios'])+f" | {q['geometric_mean']:.4f} | "+' | '.join(f"{q['errors'][k+'.relative_l2_percent']['geometric_mean']:.4f}" for k in ['u_area','s_area','wall_u'])+f" | {q['geometry_faster_count']}/3 |")
    lines+=['','## Prespecified joint targets and nonattainment','','Targets are joint area displacement / area stress / wall displacement relative L2 percentages. Accuracy is observed only at800 and1600 steps. These are first **observed** attainment times, not true first-passage times. A run can attain at800 and fail at1600; this is shown explicitly. Nonattainment means no recorded endpoint meets the target, not proof that no unobserved iteration ever did. No successful-only pooled speedup is calculated.','','| Case | Target U/S/wall (%) | Model | Seed | At800 / at1600 | First observed time (s) | Nonattainment horizon (s) | Lost at1600 |','|---|---|---|---:|---|---:|---:|---|']
    for v in targets.values():
     for s,q in v['by_seed'].items():
        first='—' if q['first_observed_seconds'] is None else f"{q['first_observed_seconds']:.3f}";censor='—' if q['censor_seconds'] is None else f"{q['censor_seconds']:.3f}"
        lines.append(f"| {v['case']} | {'/'.join(map(str,v['target']))} | {F[v['method']]} | {s} | {q['hits'][0]} / {q['hits'][1]} | {first} | {censor} | {q['lost_at1600']} |")
    lines+=['','| Case | Target | Control | G/control attaining counts | Same attaining seeds? | Conditional G/control time GM | G/control retained at1600 |','|---|---|---|---|---|---:|---|']
    for v in target_pairs.values():
        ratio='—' if v['conditional_geometric_mean'] is None else f"{v['conditional_geometric_mean']:.4f}"
        lines.append(f"| {v['case']} | {'/'.join(map(str,v['target']))} | {F[v['control']]} | {len(v['geometry_attaining_seeds'])}/3 / {len(v['control_attaining_seeds'])}/3 | {v['same_success_set']} | {ratio} | {v['geometry_retained_at1600']}/3 / {v['control_retained_at1600']}/3 |")
    lines+=['','Conditional time ratios are calculated only for IDENTICAL nonempty attaining-seed sets. They describe those shared successes; missing seeds remain nonattainments in the table above. When sets differ, no pooled successful-only ratio is reported. They are not unconditional expected solution times.','',
      'The apparently favorable S1 intermediate-target ratio0.7264 against U is fragile: for seed260932, U’s stress error at800 is10.002815%, only0.002815 percentage points above the10% threshold. G and F pass at800; U first passes at1600. Thus the apparent twofold checkpoint delay reflects sparse observation and a near-threshold miss, not a demonstrated robust speedup. Seed260931 reaches at1600 for all methods; seed260930 fails for all. Against F, G is slower on the shared attaining seeds (conditional time ratio1.0254). No threshold or budget is changed to promote this result.','',
      '## Accuracy context and claim boundary','',
      'P2K established a repeated S1 stress-distribution benefit at1600 under common refresh: geometry reduces stress MAE and p95 against both controls across all three seeds. Against half-Fourier, paired geometric-mean reductions are about11.2% and12.3%. Area stress and wall L2 also improve in all three seeds, but displacement worsens in both new seeds, and typical wall MAE worsens despite lower wall L2. T1 at800 repeats the independent geometry-control benefit; complete superiority against half-Fourier does not repeat. These spatial and engineering qualifications remain valid because the timed states are exact copies.','',
      'Use the measured cost ratios above to describe the price of those scoped benefits. No claim of universal accuracy, faster attainment, lower memory, original anchoring attribution, or independently established novelty follows merely from a lower training loss or an improved stress percentile. The joint targets remain the preregistered ones; stress-only targets are not introduced after inspecting favorable stress results.','',
      '## Source map','',
      '- `manifest.json`: registered protocol/source/input hashes, chronological order, all idle gates, worker process envelopes and completion status.','- `environment.json` and `gpu_telemetry.csv`: machine/runtime metadata and continuous GPU telemetry.','- Each numbered run folder: all four checkpoint states, accepted-step trace, component timings, sampled RSS/CPU/AC, reference-free10000-query outputs, and exact original-state verification.','- `summary.json`: all18 timing records,36 joined accuracy endpoints, paired costs/errors, full metrics and nonattainment entries. Each row records original and timed checkpoint paths/hashes and frozen evaluation source.','- Accuracy raw fields remain at their immutable P2J/P2K source paths; P2K `summary.json` and its completion hash are verified before this join.','- Protocol: `01_研究/protocols/R2_P2L_formal_cost.json`; runner/worker/report scripts: `01_研究/code/bench_p2l.py`, `bench_p2l_worker.py`, `report_p2l.py`.','']
    (ROOT/'P2L_Formal_Cost_Report.md').write_text('\n'.join(lines),encoding='utf-8');print('REPORT_WRITTEN',len(rows),'accuracy endpoints')
if __name__=='__main__':main()
