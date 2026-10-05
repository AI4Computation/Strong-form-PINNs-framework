"""Untimed spatial and wall-quantity checks of the 32 measured C1 endpoints."""
from pathlib import Path
import sys
import json
import benchmark_c1_cost as bench
import numpy as np

OUT = bench.OUT
METHODS = ['anchored', 'fourier', 'fourier_half', 'independent_marginal']


def moments(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), minimum=float(a.min()), maximum=float(a.max()))


def field_stats(error2, truth2, w):
    order = np.argsort(error2); error = np.sqrt(error2[order]); weight = w[order]
    total = w.sum(); area = np.cumsum(weight) / total
    scale = np.sqrt((w @ truth2) / total)
    descending_area = np.cumsum(weight[::-1]) / total
    descending_error = np.cumsum((weight * error2[order])[::-1]) / (w @ error2)
    return dict(area_stress_pct=float(100*np.sqrt((w@error2)/(w@truth2))),
        point_stress_pct=float(100*np.sqrt(error2.sum()/truth2.sum())),
        mean_absolute_error_pct=float(100*(weight@error)/total/scale),
        quantiles_pct={f'p{q:g}':float(100*np.interp(q/100,area,error)/scale) for q in [50,90,95,99,99.9]},
        worst_area_error_share_pct={str(q):float(100*np.interp(q/100,np.r_[0,descending_area],np.r_[0,descending_error])) for q in [1,5]})


def wall_stats(model, ref):
    xy,u=ref['xy_u'],ref['u']
    keep=np.abs(np.linalg.norm(xy,axis=1)-.1)<1e-7
    edge,truth=xy[keep],u[keep]
    theta=np.arctan2(edge[:,1],edge[:,0])%(2*np.pi)
    order=np.argsort(theta);edge,truth,theta=edge[order],truth[order],theta[order]
    dt=np.diff(np.r_[theta,theta[0]+2*np.pi]);weight=(dt+np.roll(dt,1))/2
    assert len(theta)==1840 and np.max(dt)<.004
    predicted=bench.predict(model,edge,'u','circle')
    result=dict(wall_vector_u_pct=float(100*np.sqrt(np.sum(weight[:,None]*(predicted-truth)**2)/np.sum(weight[:,None]*truth**2))))
    targets=np.array([[.1,0],[-.1,0],[0,.1],[0,-.1]])
    ids=[int(np.argmin(np.linalg.norm(xy-t,axis=1))) for t in targets]
    assert np.max(np.linalg.norm(xy[ids]-targets,axis=1))<1e-7
    predicted=bench.predict(model,xy[ids],'u','circle');truth=u[ids]
    for label,i,j,k in [('horizontal',0,1,0),('vertical',2,3,1)]:
        actual=float(predicted[i,k]-predicted[j,k]);reference=float(truth[i,k]-truth[j,k])
        error=actual-reference
        result[label]=dict(reference=reference,prediction=actual,absolute_error=abs(error),
                           relative_error_pct=100*abs(error/reference) if abs(reference)>1e-10 else None)
    return result


def main():
    m=bench.read(OUT/'manifest.json')
    assert m['status']=='complete' and len(m['completed'])==32 and not m['flagged']
    ref,w,near=bench.load_reference();truth2=np.sum(ref['s']**2,1)
    fields={};pairs={};hashes={}
    for seed in range(41,49):
        errors={}
        for method in METHODS:
            identity=f'C1_{method}_s{seed}';folder=OUT/'runs'/identity
            result=bench.read(folder/'result.json')
            assert bench.sha(folder/'result.json')==m['result_sha256'][identity]
            model=bench.build_model(method,seed,'circle')
            model.load_state_dict(bench.torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
            pred=bench.predict(model,ref['xy_s'],'s','circle')
            err2=np.sum((pred-ref['s'])**2,1);errors[method]=err2
            stats=field_stats(err2,truth2,w)
            diffs=[abs(stats['area_stress_pct']-result['metrics']['s_area_pct']),
                   abs(stats['point_stress_pct']-result['metrics']['s_vector_pct'])]
            assert max(diffs)<1e-8,(identity,diffs)
            stats['original_measured_metrics_difference_pp']=max(diffs)
            stats['near_cavity_area_stress_pct']=float(100*np.sqrt((w[near]@err2[near])/(w[near]@truth2[near])))
            stats['wall']=wall_stats(model,ref)
            fields[identity]=stats
            hashes[identity]=dict(result=bench.sha(folder/'result.json'),model=bench.sha(folder/'model.pt'))
            del model,pred;bench.torch.cuda.empty_cache()
        for other in METHODS[1:]:
            key=f'anchored_vs_{other}'
            difference=np.sqrt(errors['anchored'])-np.sqrt(errors[other])
            tolerance=1e-12*max(1.,float(np.sqrt(w@truth2/w.sum())))
            pairs.setdefault(key,[]).append(dict(seed=seed,
                anchored_better_area_pct=float(100*w[difference < -tolerance].sum()/w.sum()),
                other_better_area_pct=float(100*w[difference > tolerance].sum()/w.sum())))
        print(f'COST_ENDPOINT_SPATIAL_SEED_{seed}_COMPLETE',flush=True)
    groups={}
    for method in METHODS:
        rows=[fields[f'C1_{method}_s{s}'] for s in range(41,49)]
        groups[method]={key:moments([row[key] for row in rows]) for key in
                       ['area_stress_pct','point_stress_pct','mean_absolute_error_pct','near_cavity_area_stress_pct']}
        groups[method]['quantiles_pct']={q:moments([r['quantiles_pct'][q] for r in rows]) for q in ['p50','p90','p95','p99','p99.9']}
        groups[method]['wall']={axis:moments([r['wall'][axis]['relative_error_pct'] for r in rows]) for axis in ['horizontal','vertical']}
    comparisons={}
    for other in METHODS[1:]:
        key=f'anchored_vs_{other}'
        comparisons[key]=dict(better_area=moments([r['anchored_better_area_pct'] for r in pairs[key]]),
            majority_area_wins=sum(r['anchored_better_area_pct']>50 for r in pairs[key]),
            wall_wins={axis:sum(fields[f'C1_anchored_s{s}']['wall'][axis]['relative_error_pct']<
                                   fields[f'C1_{other}_s{s}']['wall'][axis]['relative_error_pct'] for s in range(41,49))
                       for axis in ['horizontal','vertical']})
    bench.write(OUT/'spatial_checks.json',dict(created_utc=bench.utc(),passed=True,fields=fields,groups=groups,
        pairs=pairs,comparisons=comparisons,source_sha256=hashes,
        scope='Post-timing evaluation of the same saved endpoints. No re-training or checkpoint selection; all32 current stress metrics reproduced. These extra spatial diagnostics are not included in the timed measurement loop.',
        normalization='Stress quantiles use one global FEM area-RMS scale; no division by local near-zero stress. Wall relative displacement uses the same exact FEM boundary nodes as the prior spatial audit.'))
    print('COST_ENDPOINT_SPATIAL_CHECKS_COMPLETE')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
