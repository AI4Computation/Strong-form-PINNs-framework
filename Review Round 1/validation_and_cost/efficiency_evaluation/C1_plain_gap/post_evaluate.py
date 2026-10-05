"""Evaluate only after all 24 measured models and observations are frozen."""
import sys
from common import HERE, ROOT, OLD, METHODS, read, write, sha, utc, verify_hashes

def preconditions():
    assert not (HERE/'evaluation_freeze.json').exists(), 'Completed frozen evaluation cannot be overwritten.'
    m=read(HERE/'manifest.json')
    assert m['status'] in ['timed_models_frozen_awaiting_postevaluation','post_evaluation_complete','delivered']
    assert len(m['completed'])==24 and len(set(m['completed']))==24 and not m['flagged']
    freeze=read(HERE/'cohort_freeze.json')
    assert freeze['runs']==24 and freeze['fem_evaluations']==0
    assert freeze['protocol_sha256']==sha(HERE/'protocol.json')==m['protocol_sha256']
    verify_hashes(freeze['files_sha256'])
    p=read(HERE/'protocol.json')
    verify_hashes(p['source_sha256']); verify_hashes(p['protected_inputs_sha256'])
    post=read(HERE/'post_protocol.json')
    verify_hashes(post['source_sha256']); verify_hashes(post['reference_sha256'])
    assert post['timing_protocol_sha256']==m['protocol_sha256']
    return m,p,post

def main():
    m,p,post=preconditions()
    # No torch/CUDA or reference data are loaded until the cohort gate above passes.
    sys.path.insert(0,str(OLD/'code'))
    sys.path.insert(0,str(ROOT/'code'))
    from runtime import torch
    from models import build_model
    from metrics import predict,relative
    from fem_interpolation import Field
    from evaluate_cost_fields import field_stats,wall_stats
    import numpy as np
    from post_metrics import crossing
    def load(path):
        with np.load(path) as z:
            return {key:z[key] for key in z.files}
    ref=load(OLD/'fem/references/C1.npz')
    mesh=load(OLD/'fem/tr3_circle_sq0p0025_UnitL.npz')
    assert np.array_equal(ref['xy_u'],mesh['xy_u']) and np.array_equal(ref['xy_s'],mesh['xy_s'])
    assert mesh['u'].shape==ref['u'].shape
    w=mesh['volume']; assert np.all(w>0) and np.isfinite(w).all()
    field=Field({**mesh,'u':ref['u']},1.333,.3333)
    u_ip=np.empty((len(w),2)); s_reconstruction=np.empty_like(ref['s'])
    for start in range(0,len(w),16384):
        sl=slice(start,start+16384)
        u_ip[sl],s_reconstruction[sl],_=field.evaluate(ref['xy_s'][sl])
    reconstruction_pct=float(100*np.sqrt(np.sum(w[:,None]*(s_reconstruction-ref['s'])**2)/np.sum(w[:,None]*ref['s']**2)))
    reference_audit=dict(reference_sha256=post['reference_sha256'],nodal_points=len(ref['u']),
        integration_points=len(w),area=float(w.sum()),reconstructed_stress_area_pct=reconstruction_pct,
        displacement_reference='Original C1 nodal displacement interpolated by original element shape functions at original stress integration points.',
        stress_reference='Original exported C1 integration-point S; reconstructed stress only checks the displacement interpolation.')
    write(HERE/'evaluation/reference_audit.json',reference_audit)
    assert reconstruction_pct<.1, 'New area-displacement reference consistency check requires review.'
    del field,mesh,s_reconstruction
    near=np.linalg.norm(ref['xy_s'],axis=1)<=.2
    truth2=np.sum(ref['s']**2,axis=1)
    refden_u=float(np.sum(w[:,None]*u_ip**2))
    refden_s=float(w@truth2)
    fields={}; errors_by_seed={}; query_hash=None
    for rec in p['run_order']:
        identity=rec['identity']; folder=HERE/'runs'/identity
        result=read(folder/'result.json')
        assert result['formal_timing'] and result['fem_reference_access_forbidden']
        model=build_model(rec['method'],rec['seed'],'circle')
        output=HERE/'evaluation'/f'{identity}.json'
        trace=[]
        for obs in result['observations']:
            checkpoint=folder/obs['checkpoint']
            assert sha(checkpoint)==obs['checkpoint_sha256']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert saved['accepted_step']==obs['accepted_step']
            model.load_state_dict(saved['state_dict'])
            un=predict(model,ref['xy_u'],'u','circle')
            ip=predict(model,ref['xy_s'],'s','circle')
            ui=predict(model,ref['xy_s'],'u','circle')
            assert np.isfinite(un).all() and np.isfinite(ip).all() and np.isfinite(ui).all()
            e2=np.sum((ip-ref['s'])**2,axis=1)
            row={**obs,'u_point_pct':relative(ref['u'],un),'s_point_pct':relative(ref['s'],ip),
                 'u_area_pct':float(100*np.sqrt(np.sum(w[:,None]*(ui-u_ip)**2)/refden_u)),
                 's_area_pct':float(100*np.sqrt((w@e2)/refden_s)),
                 'cold_available_s':result['before_solver_process_elapsed_s']+obs['available_elapsed_s']}
            trace.append(row)
        assert trace[-1]['accepted_step']==result['accepted_steps']
        endpoint=field_stats(e2,truth2,w)
        endpoint['near_cavity_area_stress_pct']=float(100*np.sqrt(w[near]@e2[near]/(w[near]@truth2[near])))
        endpoint['wall']=wall_stats(model,ref)
        endpoint['u_area_pct']=trace[-1]['u_area_pct']
        endpoint['u_point_pct']=trace[-1]['u_point_pct']
        assert abs(endpoint['area_stress_pct']-trace[-1]['s_area_pct'])<1e-9
        assert abs(endpoint['point_stress_pct']-trace[-1]['s_point_pct'])<1e-9
        targets={mode:{str(t):crossing(trace,keys,t) for t in p['accuracy_targets_pct']}
                 for mode,keys in {'point':['u_point_pct','s_point_pct'],
                                   'area':['u_area_pct','s_area_pct'],
                                   'point_u_area_s':['u_point_pct','s_area_pct']}.items()}
        # Check saved final artifact against the last observed state, without retraining.
        final=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state_dict']
        assert all(torch.equal(final[k],saved['state_dict'][k]) for k in final)
        if query_hash is None:
            query_hash=result['query']['point_sha256']
        assert result['query']['point_sha256']==query_hash
        write(output,dict(identity=identity,method=rec['method'],seed=rec['seed'],evaluated_utc=utc(),
            protocol_sha256=sha(HERE/'protocol.json'),post_protocol_sha256=sha(HERE/'post_protocol.json'),
            result_sha256=sha(folder/'result.json'),trace=trace,endpoint=endpoint,targets=targets,
            unique_observed_states=len(trace),timing_valid_for_comparison=False,
            note='Post-evaluation time is not solver time; threshold times come only from the frozen worker timestamps.'))
        fields[identity]=endpoint
        errors_by_seed.setdefault(rec['seed'],{})[rec['method']]=e2
        print(f'POST_EVAL {len(fields)}/24 {identity}; states={len(trace)}',flush=True)
        write(HERE/'evaluation/progress.json',dict(completed=list(fields),active=None,updated_utc=utc()))
        del model,un,ui,ip
        torch.cuda.empty_cache()
    pairs={}
    tolerance=1e-12*max(1.,float(np.sqrt(refden_s/w.sum())))
    for other in ['vanilla_matched','fourier_half']:
        values=[]
        for seed in range(41,49):
            a,b=errors_by_seed[seed]['anchored'],errors_by_seed[seed][other]
            d=np.sqrt(a)-np.sqrt(b)
            values.append(dict(seed=seed,anchored_better_area_pct=float(100*w[d < -tolerance].sum()/w.sum()),
                other_better_area_pct=float(100*w[d > tolerance].sum()/w.sum())))
        pairs['anchored_vs_'+other]=values
    write(HERE/'evaluation/spatial_pairs.json',dict(pairs=pairs,normalization='Global area-RMS stress scale; no division by local near-zero stress.'))
    files={str(file):sha(file) for file in (HERE/'evaluation').glob('*.json')}
    verify_hashes(read(HERE/'cohort_freeze.json')['files_sha256'])
    verify_hashes(post['reference_sha256'])
    write(HERE/'evaluation_freeze.json',dict(frozen_utc=utc(),runs=24,checkpoint_evaluations=sum(
        read(HERE/'evaluation'/f'{r["identity"]}.json')['unique_observed_states'] for r in p['run_order']),
        files_sha256=files,post_protocol_sha256=sha(HERE/'post_protocol.json')))
    m.update(status='post_evaluation_complete',active=None,updated_utc=utc())
    write(HERE/'manifest.json',m)
    print('POST_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
