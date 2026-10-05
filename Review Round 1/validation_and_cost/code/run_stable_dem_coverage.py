"""Cover all46 original DEM pairs using the unchanged, audited pilot algorithm."""
import run_stable_dem as engine
from run_stable_dem import ROOT,read,write,sha,utc,Path,sys,json,zipfile,threadpool_limits
import shutil

BATCH=ROOT/'stable_dem_coverage'
PILOT=ROOT/'stable_dem_pilot'


def initialize():
    if (BATCH/'manifest.json').exists():
        m=read(BATCH/'manifest.json')
        for group in ['source_sha256','input_sha256']:
            for path,h in m[group].items():assert sha(path)==h,path
        assert sha(BATCH/'protocol.json')==m['protocol_sha256']
        assert sha(BATCH/'source.zip')==m['source_archive_sha256']
        return m
    pilot=read(PILOT/'manifest.json');completed=read(PILOT/'completion.json')
    assert pilot['status']=='complete' and completed['passed'] and completed['pilot_numerically_reliable']
    for group in ['source_sha256','input_sha256']:
        for path,h in pilot[group].items():assert sha(path)==h,path
    assert sha(PILOT/'protocol.json')==pilot['protocol_sha256']
    assert sha(PILOT/'source.zip')==pilot['source_archive_sha256']
    prior=read(PILOT/'protocol.json');p=dict(prior)
    jobs=[c for c in read(ROOT.parent/'controlled_pinn/config/run_manifest.json') if c['method']=='dem']
    assert len(jobs)==46 and len({c['id'] for c in jobs})==46
    p.update(created_utc=utc(),jobs=jobs,question='Cover all46 original DEM case/seed pairs with the unchanged pilot stabilization algorithm.',
        scope='Eight original circle loading cases; C1/C8 have8 seeds and C2-C7 have5. Reuse all5 pilot endpoints unchanged; train the other41. No replacement using favorable old DEM endpoints.',
        evaluation='Freeze all41 new endpoints before evaluating any of them against FEM. The5 reused pilot FEM results are already known and are identified separately. No pooling as independent new confirmation.',
        pilot_acceptance='The completed5-run pilot passed its frozen numerical reliability gate; this is entry to coverage, not an accuracy threshold.',
        confirmation='This is baseline coverage under the same algorithm, not independent method-development confirmation. Report all46 results, numerical-budget stops, actual updates, discarded blocks and validation work.',
        data_seen='All original46 DEM results and the5 stable pilot outcomes are known. Algorithm, tolerances, quadrature budgets and optimizer controls remain exactly those of the pilot.',
        new_runs=41,reused_pilot_ids=pilot['completed'],pilot_protocol_sha256=pilot['protocol_sha256'],
        algorithm='Call the exact frozen run_stable_dem.solve function; override only output location and progress reporting.',
        stop_on_unexpected_failure='Preserve partial files and stop for diagnosis. Never overwrite an incomplete run automatically.')
    numerical_keys=['architecture','derivatives','changes','fit_order','initial_base_depth','maximum_base_depth','validation','final_audit',
        'energy_component_relative_tolerance','gradient_relative_tolerance','gradient_scale_floor','check_rule','total_optimizer_accepted_step_budget',
        'block_steps','max_evaluations_per_block','history_size','tolerance_grad','tolerance_change','optimizer','failure','termination','checkpoints']
    assert all(p[k]==prior[k] for k in numerical_keys)
    sources=dict(pilot['source_sha256'])
    for filename in ['run_stable_dem_coverage.py','evaluate_stable_dem_coverage.py','finalize_stable_dem_coverage.py','complete_stable_dem_coverage.py']:
        path=Path(__file__).with_name(filename);sources[str(path)]=sha(path)
    inputs=dict(pilot['input_sha256'])
    for path in [PILOT/'manifest.json',PILOT/'completion.json',ROOT/'summary/stable_dem_pilot_analysis.json']:
        inputs[str(path)]=sha(path)
    for c in jobs:
        for path in [ROOT.parent/'controlled_pinn/runs'/c['id']/'result.json',ROOT.parent/'controlled_pinn/fem/references'/f"{c['case']}.npz"]:
            inputs[str(path)]=sha(path)
        initial_path=ROOT.parent/'controlled_pinn/runs'/c['id']/'step_00000.pt'
        if not initial_path.exists():initial_path=ROOT.parent/'controlled_pinn/runs'/f"C1_dem_s{c['seed']}"/'step_00000.pt'
        inputs[str(initial_path)]=sha(initial_path)
    model_code=ROOT.parent/'controlled_pinn/code/models.py';inputs[str(model_code)]=sha(model_code)
    initialization_checks=[]
    for seed in sorted({c['seed'] for c in jobs}):
        checkpoint=ROOT.parent/'controlled_pinn/runs'/f'C1_dem_s{seed}'/'step_00000.pt'
        old=engine.torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']
        model=engine.build(seed,'cpu')
        assert all(engine.torch.equal(v.float(),old[k]) for k,v in model.state_dict().items())
        initialization_checks.append(dict(seed=seed,reference=str(checkpoint),passed=True))
    BATCH.mkdir(exist_ok=False);write(BATCH/'protocol.json',p);write(BATCH/'progress_before.json',read(ROOT/'进度.json'))
    for name in pilot['completed']:
        destination=BATCH/name;destination.mkdir()
        for file,h in pilot['endpoint_sha256'][name].items():assert sha(PILOT/name/file)==h
        for filename in ['model.pt','result.json','history.json','full_evaluation.json']:
            shutil.copyfile(PILOT/name/filename,destination/filename)
        write(destination/'reused_pilot.json',dict(source=str(PILOT/name),new_training_performed=False,
            source_endpoint_sha256=pilot['endpoint_sha256'][name],pilot_evaluation_sha256=sha(PILOT/name/'full_evaluation.json')))
    with zipfile.ZipFile(BATCH/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,path in enumerate(sources):z.write(path,f'{i:02}/{Path(path).name}')
        z.write(BATCH/'protocol.json','protocol.json');z.write(PILOT/'protocol.json','pilot_protocol.json')
    m=dict(status='prepared',completed=list(pilot['completed']),reused=list(pilot['completed']),active=None,
        source_sha256=sources,input_sha256=inputs,endpoint_sha256=dict(pilot['endpoint_sha256']),
        protocol_sha256=sha(BATCH/'protocol.json'),source_archive_sha256=sha(BATCH/'source.zip'),
        created_utc=utc(),environment=pilot['environment'],initialization_checks=initialization_checks,timing_valid_for_comparison=False)
    write(BATCH/'manifest.json',m);return m


def progress(m,**state):
    state.update(batch='stable_dem_coverage',completed=len(m['completed']),total=46,new_completed=len(m['completed'])-len(m['reused']))
    m.update(active=state,updated_utc=utc());write(BATCH/'manifest.json',m)
    p=read(ROOT/'进度.json');p.update(updated_utc=utc(),current_jobs=[state],milestone='stable_dem_coverage_running',
        stable_dem_coverage_completed=len(m['completed']),stable_dem_46_coverage_complete=False,
        full_fem_evaluation_complete=False,ready_to_adopt_final_method=False,new_timing_comparison_performed=False)
    write(ROOT/'进度.json',p);print(json.dumps(state,ensure_ascii=False),flush=True)


def main():
    m=initialize();p=read(BATCH/'protocol.json')
    if '--prepare' in sys.argv or m['status'] in ['fit_complete','complete']:return
    engine.BATCH=BATCH;engine.progress=progress
    m['status']='running';write(BATCH/'manifest.json',m)
    try:
        for config in p['jobs']:
            if config['id'] in m['completed']:continue
            folder=BATCH/config['id']
            if folder.exists():raise RuntimeError(f'Incomplete existing run preserved for diagnosis: {folder}')
            engine.solve(config,m,p)
        m.update(status='fit_complete',active=None,all_new_endpoints_frozen_utc=utc());write(BATCH/'manifest.json',m)
        print('ALL46_STABLE_DEM_FIELDS_AVAILABLE_41_NEW_FROZEN',flush=True)
    except Exception as error:
        m.update(status='failed',active=None,error=repr(error));write(BATCH/'manifest.json',m)
        state=read(ROOT/'进度.json');state.update(current_jobs=[],milestone='stable_dem_coverage_needs_attention');write(ROOT/'进度.json',state)
        raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
