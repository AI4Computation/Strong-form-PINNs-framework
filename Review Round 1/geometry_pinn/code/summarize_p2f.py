"""Complete registered error-family and paired-seed summaries; no selection."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2F_repetition_budget';E=B/'evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(values):
    x=np.asarray(values,dtype=float)
    return dict(values=x.tolist(),mean=float(x.mean()),sample_sd=float(x.std(ddof=1)),median=float(np.median(x)),min=float(x.min()),max=float(x.max()))
def errors(metrics):
    result={}
    for name,value in metrics.items():
        if name=='engineering':continue
        for key in ['relative_l2_percent','vector_mae','max_vector_absolute_error']:
            result[name+'.'+key]=value[key]
        for label,v in zip(['q50','q90','q95','q99'],value['vector_absolute_error_quantiles']):result[name+'.'+label]=v
    e=metrics['engineering']
    for label,v in zip(['vertical','horizontal'],e['convergence_absolute_error']):result['convergence.'+label]=v
    for label,v,xy in zip(['crown','invert','right','left'],e['extrema_vector_error'],e['extrema_absolute_error']):
        result['extrema.'+label+'.vector']=v
        result['extrema.'+label+'.ux']=xy[0];result['extrema.'+label+'.uy']=xy[1]
    # Error concentration is descriptive; lower concentration is not necessarily better.
    return result
def main():
    dest=E/'summary.json';assert not dest.exists()
    a=read(E/'analysis.json');p=read(ROOT/'protocols/R2_P2F_repetition_budget.json')
    assert read(E/'progress.json')['completed_fields']==81
    s=dict(source_analysis_sha256=sha(E/'analysis.json'),summary_code_sha256=sha(__file__),
        seeds=p['seeds'],n_seeds=3,exploratory_seed_excluded=260926,formal_timing=False,
        groups={},paired={},spatial={},concentration={},resources={},target_attainment={})
    for case in p['case_settings']:
        for step in p['evaluation_steps']:
            tag=f'{case}_step{step:04d}'
            rows=[a['cases'][case][str(seed)][str(step)] for seed in p['seeds']]
            flats=[{m:errors(row[m]['metrics']) for m in p['methods']} for row in rows]
            keys=list(flats[0][p['methods'][0]])
            assert all(set(f[m])==set(keys) for f in flats for m in p['methods'])
            s['groups'][tag]={m:{k:stats([f[m][k] for f in flats]) for k in keys} for m in p['methods']}
            s['paired'][tag]={};s['spatial'][tag]={}
            for control in ['fourier_half','uniform_rbf_fourier']:
                pair={}
                for k in keys:
                    x=np.array([f['geometry_rbf_fourier'][k] for f in flats]);y=np.array([f[control][k] for f in flats])
                    ratio=x/y if np.all(y>0) else None
                    gm=float(np.exp(np.mean(np.log(ratio)))) if ratio is not None and np.all(ratio>0) else None
                    pair[k]=dict(geometry_values=x.tolist(),control_values=y.tolist(),paired_ratios=None if ratio is None else ratio.tolist(),
                        geometric_mean_ratio=gm,wins=int(np.sum(x<y)),ties=int(np.sum(x==y)),
                        consistent_direction=bool(np.all(x<y)),material_consistent_signal=bool(np.all(x<y) and gm is not None and gm<=.9))
                s['paired'][tag][control]=pair
                s['spatial'][tag][control]={field:stats([row['geometry_rbf_fourier']['area_fraction_lower_vector_error'][control][field] for row in rows]) for field in ['u','s']}
                s['spatial'][tag][control]['wall_u']=stats([row['geometry_rbf_fourier']['wall_fraction_lower_displacement_error'][control] for row in rows])
            s['concentration'][tag]={m:{field:stats([row[m]['metrics'][field+'_area']['worst_one_percent_area_squared_error_share'] for row in rows]) for field in ['u','s']} for m in p['methods']}
            s['resources'][tag]={m:{key:stats([row[m][key] for row in rows]) for key in ['actual_step','closure_evaluations']} for m in p['methods']}
        for method in p['methods']:
            for seed in p['seeds']:
                name=f'{case}_seed{seed}_{method}';items=[]
                for target in [(5,20,10),(2,10,5),(1,5,2)]:
                    endpoints=[a['cases'][case][str(seed)][str(step)][method] for step in p['evaluation_steps']]
                    met=[all(row['metrics'][k]['relative_l2_percent']<=limit for k,limit in zip(['u_area','s_area','wall_u'],target)) for row in endpoints]
                    first=met.index(True) if any(met) else None
                    items.append(dict(target_u_s_wall_percent=list(target),met_at_registered_endpoints=met,
                        first_observed_nominal_step=None if first is None else p['evaluation_steps'][first],
                        first_observed_actual_step=None if first is None else endpoints[first]['actual_step'],
                        first_observed_closures=None if first is None else endpoints[first]['closure_evaluations'],
                        sustained_through_later_endpoints=None if first is None else all(met[first:])))
                s['target_attainment'][name]=items
    (E/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ALL_ERROR_FAMILIES_PAIRED_BY_NEW_SEED_AND_BUDGET',flush=True)
    for tag,controls in s['paired'].items():
        print(tag,{c:{k:(v['geometric_mean_ratio'],v['wins']) for k,v in groups.items() if k in ['u_area.relative_l2_percent','s_area.relative_l2_percent','wall_u.relative_l2_percent']} for c,groups in controls.items()},flush=True)
if __name__=='__main__':main()
