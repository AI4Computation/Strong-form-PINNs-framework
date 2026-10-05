"""Blockwise energy/gradient validation without FEM-guided selection."""
from stable_dem import *

def progress(m,**state):
    print(json.dumps(state),flush=True)

def solve(config,m,p,folder,device):
    folder.mkdir(parents=True,exist_ok=False)
    model=build(config['seed'],device=device);initial=snapshot(model);torch.save(dict(state_dict=initial),folder/'initial.pt')
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
            return torch.tensor(value,dtype=torch.float64,device=device)
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
    if device=='cuda':torch.cuda.empty_cache()

