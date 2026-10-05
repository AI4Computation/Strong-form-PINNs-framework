"""Prospective DEM reliability pilot; no FEM-driven steps or quadrature choices."""
from stable_dem import *
import zipfile

BATCH=ROOT/'stable_dem_pilot'
JOBS=[('C1',42),('C1',45),('C8',42),('C8',45),('C7',42)]


def initialize():
    if (BATCH/'manifest.json').exists():
        m=read(BATCH/'manifest.json')
        for group in ['source_sha256','input_sha256']:
            for path,h in m[group].items():assert sha(path)==h,path
        assert sha(BATCH/'protocol.json')==m['protocol_sha256'] and sha(BATCH/'source.zip')==m['source_archive_sha256']
        return m
    check=read(ROOT/'checks/stable_dem_verification.json');assert check['passed']
    for path,h in check['source_sha256'].items():assert sha(path)==h
    old_jobs=read(ROOT.parent/'controlled_pinn/config/run_manifest.json')
    configs=[next(c for c in old_jobs if c['method']=='dem' and c['case']==case and c['seed']==seed) for case,seed in JOBS]
    p=dict(created_utc=utc(),jobs=configs,
        question='Can geometry-fitted quadrature plus independent energy/gradient checks prevent the diagnosed late DEM deterioration under the original network and1000-step opportunity?',
        scope='Five known development runs: two declared seeds on asymmetricC1 and symmetricC8, plus reversed-loadC7 seed42. Not the final46-run confirmation.',
        architecture='Unmodified legacy.DEMNet:2-200-200-200-2 with tanh,81402 trainable parameters; same hard support/gauge construction and original CPU float32 initialization stream200000+seed, then cast tofloat64.',
        derivatives='Analytic propagation of spatial Jacobians through the same MLP, checked against autograd for fields and parameter gradients; no representation change.',
        changes='Float64 numerical stabilization package; exact-geometry sliced composite Gauss integration, separately integrated boundary work, blockwise independent energy and parameter-gradient validation. Not a single-factor attribution to quadrature alone.',
        fit_order=5,initial_base_depth=4,maximum_base_depth=6,validation='base+1/order5',final_audit='base+1/order7',
        energy_component_relative_tolerance=1e-4,gradient_relative_tolerance=.05,gradient_scale_floor=1e-6,
        check_rule='Compare internal energy and external work separately against the final independent rule, and parameter-gradientL2 between fit/validation. Check validation potential nonincrease within the component tolerance.',
        total_optimizer_accepted_step_budget=1000,block_steps=50,max_evaluations_per_block=125,history_size=50,
        tolerance_grad=1e-7,tolerance_change=1e-12,optimizer='Installed observed L-BFGS strong_wolfe; fixed objective within every block; reset history between blocks.',
        failure='Save failed field and audit; rollback that block; increase geometry quadrature depth and reset history. Discarded accepted updates still count toward the1000-step total. At maximum depth, stop with the last verified field and explicitly report a numerical-budget stop.',
        termination='Last numerically verified state after total budget or an optimizer convergence stop; a per-block max_eval stop starts a new block if updates remain. Never FEM-best or minimum-observed-error checkpoint. All accepted/rollback counts and repeated work retained.',
        checkpoints='Initial, each first retained block crossing a100-update boundary, every failure, and endpoint. No FEM is read by the runner.',
        evaluation='Freeze all five endpoints before FEM evaluation; full original U/S and near-cavity metrics plus area-weighted stress. Fixed checkpoint learning diagnostics are descriptive and may not select endpoints.',
        pilot_acceptance='All five endpoints pass independent numerical checks and complete the declared opportunity without an unresolved numerical-budget stop. Report accuracy separately; stability does not require monotonically decreasing FEM errors.',
        confirmation='If this pilot is numerically reliable, freeze the same algorithm for all46 original DEM case/seed pairs, not only the previously failed runs. No automatic final-method or efficiency claim.',
        data_seen='Original46 DEM trajectories and quadrature failure diagnostics are known. No stabilized pilot FEM errors observed before freeze.',
        timing_valid_for_comparison=False,selection_uses_fem=False,cpu_threads=2,device='cuda_float64')
    sources={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in ['stable_dem.py','verify_stable_dem.py','run_stable_dem.py','evaluate_stable_dem.py',
        'slice_quadrature.py','geometry.py','run_elastic_sources.py','problems.py','mechanics.py','features.py','linear.py','roller_sources.py','elastic_sources.py']}
    sources[str(LEGACY)]=sha(LEGACY)
    for path in [ROOT.parent/'controlled_pinn/code/observed_lbfgs.py',ROOT.parent/'controlled_pinn/config/geometry.json']:sources[str(path)]=sha(path)
    inputs={str(ROOT/'checks/stable_dem_verification.json'):sha(ROOT/'checks/stable_dem_verification.json')}
    for c in configs:
        for path in [ROOT.parent/'controlled_pinn/runs'/c['id']/'result.json',ROOT.parent/'controlled_pinn/fem/references'/f"{c['case']}.npz",
            ROOT.parent/'controlled_pinn/fem/tr3_circle_sq0p0025_UnitL.npz']:
            inputs[str(path)]=sha(path)
    path=ROOT.parent/'controlled_pinn/config/run_manifest.json';inputs[str(path)]=sha(path)
    BATCH.mkdir(exist_ok=False);write(BATCH/'protocol.json',p);write(BATCH/'progress_before.json',read(ROOT/'进度.json'))
    with zipfile.ZipFile(BATCH/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,path in enumerate(sources):z.write(path,f'{i:02}/{Path(path).name}')
        z.write(BATCH/'protocol.json','protocol.json');z.write(ROOT/'checks/stable_dem_verification.json','verification.json')
    m=dict(status='prepared',completed=[],active=None,source_sha256=sources,input_sha256=inputs,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
        protocol_sha256=sha(BATCH/'protocol.json'),source_archive_sha256=sha(BATCH/'source.zip'),created_utc=utc(),timing_valid_for_comparison=False)
    write(BATCH/'manifest.json',m);return m


def progress(m,**state):
    state['batch']='stable_dem_pilot';m.update(active=state,updated_utc=utc());write(BATCH/'manifest.json',m)
    p=read(ROOT/'进度.json');p.update(updated_utc=utc(),current_jobs=[state],milestone='stable_dem_pilot_running',
        full_fem_evaluation_complete=False,ready_to_adopt_final_method=False,new_timing_comparison_performed=False)
    write(ROOT/'进度.json',p);print(json.dumps(state,ensure_ascii=False),flush=True)


def solve(config,m,p):
    folder=BATCH/config['id'];folder.mkdir(exist_ok=True)
    model=build(config['seed']);initial=snapshot(model);torch.save(dict(state_dict=initial),folder/'initial.pt')
    attempted=retained=closures=0;checkpoint_bucket=0;depth=p['initial_base_depth'];promotions=[];history=[];last_energy=None;stop=None
    samples={}
    def rules(base):
        if base not in samples:samples[base]=[rule(base,5),rule(base+1,5),rule(base+1,7)]
        return samples[base]
    checked=None
    while depth<=p['maximum_base_depth']:
        progress(m,run=config['id'],action='initial_numerical_check',base_depth=depth)
        checked=numerical_check(model,*rules(depth),config,p)
        if checked['passed']:break
        promotions.append(dict(stage='initial',from_depth=depth,audit=checked));depth+=1
    assert checked is not None and checked['passed'],'Initial DEM field unresolved at declared quadrature budget.'
    last=snapshot(model);last_energy=checked['final']['potential_energy'];initial_audit=checked
    while attempted<p['total_optimizer_accepted_step_budget']:
        count=min(p['block_steps'],p['total_optimizer_accepted_step_budget']-attempted)
        progress(m,run=config['id'],action='training_block',attempted_steps=attempted,retained_steps=retained,base_depth=depth)
        optimizer=ObservedLBFGS(list(model.parameters()),max_iter=count,max_eval=p['max_evaluations_per_block'],
            history_size=p['history_size'],line_search_fn='strong_wolfe',tolerance_grad=p['tolerance_grad'],tolerance_change=p['tolerance_change'])
        loss_values=[]
        def closure():
            r=energy(model,rules(depth)[0],config['p_lateral'],config['p_top'],True)
            value=r['potential_energy']
            if not np.isfinite(value):raise FloatingPointError('Nonfinite DEM trial energy.')
            loss_values.append(value)
            return torch.tensor(value,dtype=torch.float64,device='cuda')
        optimizer.step(closure);steps=optimizer.accepted_steps;attempted+=steps;closures+=len(loss_values)
        checked=numerical_check(model,*rules(depth),config,p,last_energy)
        record=dict(attempted_steps=attempted,retained_steps_before=retained,block_steps=steps,closure_evaluations=len(loss_values),
            base_depth=depth,optimizer_stop=optimizer.stop_reason,audit=checked,closure_losses=loss_values)
        history.append(record);write(folder/'history.json',history)
        progress(m,run=config['id'],action='block_numerical_audit',attempted_steps=attempted,base_depth=depth,
            passed=checked['passed'],potential=checked['final']['potential_energy'],gradient_discrepancy=checked['gradient_relative_difference'])
        if not checked['passed']:
            torch.save(dict(state_dict=snapshot(model),attempted_steps=attempted),folder/f'failed_{attempted:04d}.pt')
            model.load_state_dict(last)
            if depth>=p['maximum_base_depth']:
                stop='numerical_quadrature_budget';break
            promotions.append(dict(stage='block_rollback',attempted_steps=attempted,discarded_steps=steps,from_depth=depth))
            depth+=1
            checked=numerical_check(model,*rules(depth),config,p)
            while not checked['passed'] and depth<p['maximum_base_depth']:
                promotions.append(dict(stage='rollback_state_check',from_depth=depth,audit=checked));depth+=1
                checked=numerical_check(model,*rules(depth),config,p)
            if not checked['passed']:stop='rollback_state_unresolved';break
            last_energy=checked['final']['potential_energy']
            continue
        retained+=steps;last=snapshot(model);last_energy=checked['final']['potential_energy']
        if attempted//100>checkpoint_bucket:
            torch.save(dict(state_dict=last,attempted_steps=attempted,retained_steps=retained),folder/f'checkpoint_{attempted:04d}.pt')
            checkpoint_bucket=attempted//100
        if steps<count and optimizer.stop_reason!='max_eval':stop=optimizer.stop_reason or 'optimizer_stopped';break
    stop=stop or 'total_accepted_update_budget'
    model.load_state_dict(last);final_check=numerical_check(model,*rules(depth),config,p)
    torch.save(dict(state_dict=snapshot(model)),folder/'model.pt')
    r=dict(configuration=config,attempted_accepted_steps=attempted,retained_accepted_steps=retained,closure_evaluations=closures,
        rejected_accepted_steps=attempted-retained,stop_reason=stop,sampling_verified=final_check['passed'],
        initial_audit=initial_audit,final_audit=final_check,promotions=promotions,final_base_depth=depth,
        trainable_parameters=sum(x.numel() for x in model.parameters()),new_dtype='float64',initial_weights_dtype='float32',
        quadrature={str(base):[dict(points=len(q['xy']),boundary_points=len(q['boundary']),area=q['area'],order=q['order']) for q in qlist] for base,qlist in samples.items()},
        selection_uses_fem=False,timing_valid_for_comparison=False,model_sha256=sha(folder/'model.pt'))
    write(folder/'result.json',r);m['completed'].append(config['id'])
    m.setdefault('endpoint_sha256',{})[config['id']]={file:sha(folder/file) for file in ['result.json','model.pt']}
    progress(m,run=config['id'],action='endpoint_frozen',stop=stop,retained_steps=retained,sampling_verified=final_check['passed'])
    del model,samples
    torch.cuda.empty_cache()


def main():
    m=initialize();p=read(BATCH/'protocol.json')
    if '--prepare' in sys.argv or m['status'] in ['fit_complete','complete']:return
    m['status']='running';write(BATCH/'manifest.json',m)
    try:
        for config in p['jobs']:
            if config['id'] not in m['completed']:solve(config,m,p)
        m.update(status='fit_complete',active=None,all_endpoints_frozen_utc=utc());write(BATCH/'manifest.json',m)
        print('ALL_FIVE_STABLE_DEM_ENDPOINTS_FROZEN',flush=True)
    except Exception as error:
        m.update(status='failed',active=None,error=repr(error));write(BATCH/'manifest.json',m)
        s=read(ROOT/'进度.json');s.update(current_jobs=[],milestone='stable_dem_pilot_needs_attention');write(ROOT/'进度.json',s)
        raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
