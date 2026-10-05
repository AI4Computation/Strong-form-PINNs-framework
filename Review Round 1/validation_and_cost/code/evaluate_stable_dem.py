"""Original full reference metrics for all frozen stable-DEM pilot endpoints."""
from run_stable_dem import *


def evaluate(model,reference,volume):
    sums={key:np.zeros(2) for key in ['u','s','s_area','s_near']}
    for name,columns in [('u',[0,1]),('s',[2,3,4])]:
        points=reference['xy_'+name];truth=reference[name]
        for start in range(0,len(points),4096):
            sl=slice(start,start+4096);value=predict(model,points[sl])[:,columns]
            error=np.sum((value-truth[sl])**2,axis=1);norm=np.sum(truth[sl]**2,axis=1)
            sums[name]+=[error.sum(),norm.sum()]
            if name=='s':
                w=volume[sl];sums['s_area']+=[w@error,w@norm]
                near=np.linalg.norm(points[sl],axis=1)<=.2;sums['s_near']+=[error[near].sum(),norm[near].sum()]
    return {key:float(100*np.sqrt(a/b)) for key,(a,b) in sums.items()}


def main():
    m=initialize();p=read(BATCH/'protocol.json');assert m['status'] in ['fit_complete','complete'] and len(m['completed'])==len(p['jobs'])
    for name,hashes in m['endpoint_sha256'].items():
        for file,h in hashes.items():assert sha(BATCH/name/file)==h
    weights_path=ROOT.parent/'controlled_pinn/fem/tr3_circle_sq0p0025_UnitL.npz';weights=dict(np.load(weights_path));summary={}
    for config in p['jobs']:
        name=config['id'];folder=BATCH/name
        if (folder/'full_evaluation.json').exists():
            e=read(folder/'full_evaluation.json');assert e['endpoint_sha256']==m['endpoint_sha256'][name];summary[name]=e;continue
        progress(m,run=name,action='full_frozen_endpoint_FEM_evaluation')
        model=build(config['seed']);model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
        path=ROOT.parent/'controlled_pinn/fem/references'/f"{config['case']}.npz";reference=dict(np.load(path))
        assert np.array_equal(reference['xy_s'],weights['xy_s'])
        values=evaluate(model,reference,weights['volume']);r=read(folder/'result.json')
        old=read(ROOT.parent/'controlled_pinn/runs'/name/'result.json')
        e=dict(configuration=config,metrics_percent=values,original_terminal_metrics=old['metrics'],
            attempted_accepted_steps=r['attempted_accepted_steps'],retained_accepted_steps=r['retained_accepted_steps'],
            closure_evaluations=r['closure_evaluations'],stop_reason=r['stop_reason'],sampling_verified=r['sampling_verified'],
            promotions=r['promotions'],final_audit=r['final_audit'],evaluated_utc=utc(),
            endpoint_sha256=m['endpoint_sha256'][name],reference_sha256=sha(path),weights_sha256=sha(weights_path),
            evaluator_sha256=sha(__file__),selection_uses_fem=False,timing_valid_for_comparison=False)
        write(folder/'full_evaluation.json',e);summary[name]=e;print(json.dumps(dict(run=name,metrics=values,stop=r['stop_reason']),ensure_ascii=False),flush=True)
    path=ROOT/'summary/stable_dem_pilot.json';write(path,summary)
    m.update(status='complete',active=None,summary_sha256=sha(path),evaluation_sha256={name:sha(BATCH/name/'full_evaluation.json') for name in m['completed']})
    write(BATCH/'manifest.json',m);print('ALL_STABLE_DEM_PILOT_EVALUATIONS_COMPLETE',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
