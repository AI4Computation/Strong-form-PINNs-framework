"""Complete-run tables, paired-seed intervals, and transparent NR accounting."""
import runtime
from models import ROOT
import numpy as np,json,csv,math
from collections import defaultdict

def write_csv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT/'summary'/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize(require_complete=False):
    manifest=json.loads((ROOT/'config/run_manifest.json').read_text())
    results=[];missing=[];expected=defaultdict(int)
    for config in manifest:
        expected[(config['case'],config['method'])]+=1
        path=ROOT/'runs'/config['id']/'result.json'
        if not path.exists():missing.append(config['id']);continue
        value=json.loads(path.read_text(encoding='utf-8'))
        assert value['configuration']==config and not value['pilot'] and value['status']=='complete'
        results.append(value)
    if require_complete:assert not missing,missing
    rows=[];groups=defaultdict(list)
    for value in results:
        c=value['configuration'];groups[(c['case'],c['method'])].append(value)
        rows.append({**c,**value['metrics'],**{k:value[k] for k in ('accepted_steps','closure_evaluations','stop_reason',
            'optimization_s','total_wall_s','diagnostic_and_checkpoint_s','peak_cuda_allocated_MiB','trainable_parameters')}})
    write_csv('all_runs.csv',rows)
    metrics=['s_vector_pct','u_vector_pct','s_near_pct','optimization_s','total_wall_s','peak_cuda_allocated_MiB']
    summary=[];thresholds=[]
    for key,values in sorted(groups.items()):
        case,method=key;n=len(values);base={'case':case,'method':method,'n':n,'planned_n':expected[key],'complete':n==expected[key]}
        for metric in metrics:
            a=[v['metrics'].get(metric,v.get(metric)) for v in values]
            a=np.array([x for x in a if x is not None])
            if len(a):summary.append({**base,'metric':metric,'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else None})
        for name in values[0]['thresholds']:
            reached=[v['thresholds'][name] for v in values if v['thresholds'][name]['reached']]
            thresholds.append({**base,'target':name,'reached':len(reached),'not_reached':n-len(reached),
                'all_runs_reached':len(reached)==n,
                'median_upper_s_if_all_reached':float(np.median([t['optimization_s_upper'] for t in reached])) if len(reached)==n else None,
                'display':'NR present' if len(reached)<n else 'all reached'})
    write_csv('group_summary.csv',summary);write_csv('time_to_accuracy.csv',thresholds)
    pairs=[];rng=np.random.default_rng(20260920)
    for (case,method),values in sorted(groups.items()):
        if method=='anchored' or (case,'anchored') not in groups:continue
        anch={v['configuration']['seed']:v for v in groups[(case,'anchored')]}
        control={v['configuration']['seed']:v for v in values}
        seeds=sorted(set(anch)&set(control))
        if len(seeds)!=expected[(case,method)] or len(seeds)!=expected[(case,'anchored')]:continue
        indices=rng.integers(0,len(seeds),(10000,len(seeds)))
        for metric in ('s_vector_pct','u_vector_pct','s_near_pct'):
            if metric not in anch[seeds[0]]['metrics']:continue
            differences=np.array([anch[s]['metrics'][metric]-control[s]['metrics'][metric] for s in seeds])
            boots=differences[indices].mean(1);lo,hi=np.quantile(boots,[.025,.975])
            pairs.append({'case':case,'control':method,'metric':metric,'n_pairs':len(seeds),
                'mean_anchored_minus_control_pp':float(differences.mean()),'ci95_low':float(lo),'ci95_high':float(hi),
                'degenerate_interval':bool(lo==hi),'method':'paired seed percentile bootstrap, 10000 resamples',
                'seeds':';'.join(map(str,seeds)),'raw_differences_pp':';'.join(f'{v:.12g}' for v in differences)})
    write_csv('paired_differences.csv',pairs)
    state={'completed':len(results),'planned':len(manifest),'missing':missing,'all_training_complete':not missing,
        'statistics':'Descriptive per-case intervals; no multiplicity-adjusted significance claim. FEM sampling points are not replicates.'}
    (ROOT/'summary/completeness.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    return state

if __name__=='__main__':
    import sys
    result=summarize('--require-complete' in sys.argv)
    print(json.dumps({k:v for k,v in result.items() if k!='missing'},indent=2))
