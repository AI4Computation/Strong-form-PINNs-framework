"""Post-freeze, fixed saved-checkpoint diagnostics; no checkpoint selection."""
from evaluate_stable_dem import *


def main():
    m=initialize();protocol=read(BATCH/'protocol.json');assert m['status']=='complete'
    weights_path=ROOT.parent/'controlled_pinn/fem/tr3_circle_sq0p0025_UnitL.npz';weights=dict(np.load(weights_path))
    all_results={}
    for config in protocol['jobs']:
        name=config['id'];folder=BATCH/name;output=folder/'checkpoint_diagnostic.json'
        if output.exists():
            result=read(output)
            assert result['endpoint_sha256']==m['endpoint_sha256'][name]
            for path,h in result['checkpoint_sha256'].items():assert sha(path)==h
            all_results[name]=result;continue
        paths=[folder/'initial.pt']+sorted(folder.glob('checkpoint_*.pt'))+sorted(folder.glob('failed_*.pt'))
        frozen={str(path):sha(path) for path in paths}
        checkpoint_plan=dict(created_utc=utc(),checkpoint_sha256=frozen,selection_uses_fem=False,
            rule='Evaluate all saved initial, accepted-block and rejected-block checkpoints after all five endpoints were frozen. Never change the endpoint.')
        write(folder/'checkpoint_evaluation_plan.json',checkpoint_plan)
        reference=dict(np.load(ROOT.parent/'controlled_pinn/fem/references'/f"{config['case']}.npz"))
        assert np.array_equal(reference['xy_s'],weights['xy_s'])
        model=build(config['seed']);records=[]
        for path in paths:
            assert sha(path)==frozen[str(path)]
            checkpoint=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict'])
            step=int(checkpoint.get('attempted_steps',0))
            values=evaluate(model,reference,weights['volume'])
            record=dict(checkpoint=path.name,attempted_steps=step,rejected=path.name.startswith('failed_'),metrics_percent=values)
            records.append(record)
            print(json.dumps(dict(run=name,checkpoint=path.name,s=values['s']),ensure_ascii=False),flush=True)
        endpoint=read(folder/'full_evaluation.json')['metrics_percent']
        accepted=[r for r in records if not r['rejected']]
        descriptive={key:dict(lowest_observed_checkpoint=min(accepted,key=lambda r:r['metrics_percent'][key])['checkpoint'],
            lowest_observed_error=min(r['metrics_percent'][key] for r in accepted),
            endpoint_error=endpoint[key],endpoint_to_lowest_ratio=endpoint[key]/min(r['metrics_percent'][key] for r in accepted)) for key in endpoint}
        result=dict(configuration=config,records=records,endpoint_metrics_percent=endpoint,descriptive_only=descriptive,
            endpoint_sha256=m['endpoint_sha256'][name],checkpoint_sha256=frozen,
            evaluation_plan_sha256=sha(folder/'checkpoint_evaluation_plan.json'),evaluated_utc=utc(),
            code_sha256={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('evaluate_stable_dem.py')):sha(Path(__file__).with_name('evaluate_stable_dem.py'))},
            selection_uses_fem=False,timing_valid_for_comparison=False)
        write(output,result);all_results[name]=result
    write(ROOT/'summary/stable_dem_checkpoint_diagnostics.json',all_results)
    print('ALL_FIXED_CHECKPOINT_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
