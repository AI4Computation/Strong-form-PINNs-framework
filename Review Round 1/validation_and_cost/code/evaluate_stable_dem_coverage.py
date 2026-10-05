"""Full reference evaluation of the41 new endpoints, with5 explicitly reused."""
from evaluate_stable_dem import evaluate,build,torch,np
from run_stable_dem_coverage import *


def main():
    m=initialize();p=read(BATCH/'protocol.json');assert m['status'] in ['fit_complete','complete'] and len(m['completed'])==46
    for name,hashes in m['endpoint_sha256'].items():
        for file,h in hashes.items():assert sha(BATCH/name/file)==h
    weights_path=ROOT.parent/'controlled_pinn/fem/tr3_circle_sq0p0025_UnitL.npz';weights=dict(np.load(weights_path));summary={}
    for config in p['jobs']:
        name=config['id'];folder=BATCH/name
        if (folder/'full_evaluation.json').exists():
            e=read(folder/'full_evaluation.json');assert e['endpoint_sha256']==m['endpoint_sha256'][name]
        else:
            progress(m,run=name,action='full_frozen_endpoint_FEM_evaluation')
            model=build(config['seed']);model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
            path=ROOT.parent/'controlled_pinn/fem/references'/f"{config['case']}.npz";reference=dict(np.load(path))
            assert np.array_equal(reference['xy_s'],weights['xy_s'])
            values=evaluate(model,reference,weights['volume']);r=read(folder/'result.json');old=read(ROOT.parent/'controlled_pinn/runs'/name/'result.json')
            e=dict(configuration=config,metrics_percent=values,original_terminal_metrics=old['metrics'],
                attempted_accepted_steps=r['attempted_accepted_steps'],retained_accepted_steps=r['retained_accepted_steps'],
                closure_evaluations=r['closure_evaluations'],stop_reason=r['stop_reason'],sampling_verified=r['sampling_verified'],
                promotions=r['promotions'],final_audit=r['final_audit'],evaluated_utc=utc(),
                endpoint_sha256=m['endpoint_sha256'][name],reference_sha256=sha(path),weights_sha256=sha(weights_path),
                evaluator_sha256=sha(__file__),selection_uses_fem=False,timing_valid_for_comparison=False)
            write(folder/'full_evaluation.json',e)
            print(json.dumps(dict(run=name,metrics=values,stop=r['stop_reason']),ensure_ascii=False),flush=True)
        if name not in m['reused']:assert e['evaluated_utc']>m['all_new_endpoints_frozen_utc']
        summary[name]=dict(e,reused_from_pilot=name in m['reused'])
    path=ROOT/'summary/stable_dem_coverage.json';write(path,summary)
    m.update(status='complete',active=None,summary_sha256=sha(path),evaluation_sha256={name:sha(BATCH/name/'full_evaluation.json') for name in m['completed']})
    write(BATCH/'manifest.json',m);print('ALL46_STABLE_DEM_EVALUATIONS_COMPLETE',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
