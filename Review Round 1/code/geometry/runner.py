"""Common geometry rule, matched representations and prescribed sampling schedules."""
from common import *
from cavity_cover import normalized_domain, construct
from p2_components import uniform_cover, shared_points, prepare_loss, parts_loss, objective
from shared_geometry_features import make_shared_model, uniform_centres
from observed_lbfgs import ObservedLBFGS
import transfer_mechanics as transfer

def inputs(c):
    setting = read(ROOT/'configs/cases.json')[c['case']]
    data = read(ROOT/'configs/geometries.json')[setting['geometry']]
    domain, _, _ = normalized_domain(data)
    cover = construct(domain, read(ROOT/'configs/geometry_rule.json'))
    centres, widths = cover['centres'], cover['halfwidths']
    uc, uw = uniform_centres(len(centres), domain)
    pc, pw, _ = uniform_cover(domain, len(centres))
    covers = [(centres,widths), (pc,pw)]
    points = transfer.sample(domain,covers,c['seed'],setting) if c['case'] in ['T1','S1'] else shared_points(domain,covers,c['seed'])
    arrays = dict(centres=centres,halfwidths=widths,uniform_centres=uc,uniform_halfwidths=uw)
    return setting, domain, arrays, points

def build(c, cover, device):
    return make_shared_model(c['method'],c['seed'],cover['centres'],cover['halfwidths'],
        (cover['uniform_centres'],cover['uniform_halfwidths'])).float().to(device)

def point_block(base, domain, seed, block, arm):
    p = {k:v.copy() for k,v in base.items()}
    if arm == 'uniform_refresh' and block > 0:
        n = int(p['uniform_count'])
        p['domain'][:n] = domain.random_interior(n,np.random.default_rng(100000+seed+1000000*block))
    return p

def train(c, out, device):
    out.mkdir(parents=True,exist_ok=False)
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    synchronize(device); start=time.perf_counter()
    setting,domain,cover,base=inputs(c)
    np.savez_compressed(out/'cover.npz',**cover)
    np.savez_compressed(out/'initial_points.npz',**base)
    model=build(c,cover,device)
    trace=[]; records=[]; total=0; closures_total=0
    first_block=0
    if c['preset']=='sampling':
        prefix_id=f'budgets_{c["case"]}_{c["method"]}_s{c["seed"]}_continuous'
        prefix=ROOT/'results/budgets'/prefix_id
        prefix_record=read(prefix/'result.json')
        actual=load_weights(model,prefix/'step_0400.pt',device)
        if actual!=400:raise ValueError('The sampling intervention requires an exact 400-step prefix.')
        prior=next(t for t in read(prefix/'trace.json') if t['accepted_step']==400)
        closures_total=prior['closure_evaluations'];total=400;first_block=1
    transfer_case=c['case'] in ['T1','S1']
    blocks=[0] if c['arm']=='continuous' else list(range(first_block,4))
    stop=None
    for block in blocks:
        points=point_block(base,domain,c['seed'],block,c['arm'])
        prepared=transfer.prepare(model,points) if transfer_case else prepare_loss(model,points)
        def loss_parts():
            return transfer.parts(model,prepared,setting) if transfer_case else parts_loss(model,prepared,setting['lateral'],setting['top'])
        closures=0
        def observer(step,evaluations,loss,gradient):
            if not np.isfinite([loss,gradient]).all():raise FloatingPointError('Nonfinite accepted state.')
            current=total+step
            trace.append(dict(block=block,block_step=step,accepted_step=current,
                              closure_evaluations=closures_total+evaluations,loss=loss,max_gradient=gradient))
            if c['arm']=='continuous' and current in c['checkpoints']:
                save(model,out/f'step_{current:04d}.pt',current)
        opt=ObservedLBFGS(model.parameters(),observer=observer,**c['optimizer'])
        def closure():
            nonlocal closures
            opt.zero_grad(set_to_none=True)
            value=objective(loss_parts());finite_loss_and_gradients(value,model);closures+=1
            return value
        opt.step(closure)
        if c['arm']!='continuous' and opt.accepted_steps!=400:
            raise RuntimeError(f'Block {block} stopped at {opt.accepted_steps}: {opt.stop_reason}; no later endpoint substituted.')
        total+=opt.accepted_steps;closures_total+=closures;stop=opt.stop_reason
        if c['arm']!='continuous':save(model,out/f'step_{total:04d}.pt',total)
        with torch.no_grad():final_parts={k:float(v) for k,v in loss_parts().items()}
        records.append(dict(block=block,actual_steps=opt.accepted_steps,closures=closures,stop_reason=stop,loss_parts=final_parts))
        print(c['id'], 'accepted',total,flush=True)
        del opt,prepared
    save(model,out/'terminal.pt',total);synchronize(device)
    write(out/'trace.json',trace)
    write(out/'result.json',dict(configuration=c,accepted_steps=total,closure_evaluations=closures_total,
        blocks=records,stop_reason=stop,trainable_parameters=sum(p.numel() for p in model.parameters()),
        elapsed_including_preparation_and_checkpoint_io_seconds=time.perf_counter()-start,
        fem_during_training=False,environment=resources(device)))

def evaluate(c,out,device,all_checkpoints=False,save_fields=False):
    from evaluation import predict,score,metrics,stats,region_stats,domain_residual,independent_boundary,boundary
    rec=read(out/'result.json');setting,domain,_,_=inputs(c)
    model=build(c,load_npz(out/'cover.npz'),device)
    base=load_npz(out/'initial_points.npz')
    ref=load_npz(ROOT/f'data/fem/evaluation/{c["case"]}.npz')
    steps=[400] if c['preset']=='representations' else [400,800,1600] if c['preset']=='budgets' else [800,1600]
    rows=[]
    validation_seeds=[9274101,9274102] if c['preset']=='budgets' else [9285101,9285102] if c['preset']=='sampling' else [9296101,9296102] if c['preset']=='transfer' else [29260001,29260002]
    validations=[domain.random_interior(24000,np.random.default_rng(s)) for s in validation_seeds]
    bp,bw=independent_boundary(domain,setting)
    for step in steps:
        path=out/f'step_{step:04d}.pt'
        if not path.exists():
            if c['arm']!='continuous' or rec['accepted_steps']>=step:
                raise FileNotFoundError(path)
            path=out/'terminal.pt'
        actual=load_weights(model,path,device);model.eval()
        wall=predict(model,np.vstack([ref['wall'],ref['extrema']]))[:,:2]
        q=dict(q_ip=predict(model,ref['xy_ip']),u_node=predict(model,ref['xy_node'])[:,:2],
               u_wall=wall[:-4],u_extrema=wall[-4:])
        values,errors=score(q,ref)
        if 'wall_tags' in ref:
            for tag in np.unique(ref['wall_tags']):
                mask=ref['wall_tags']==tag
                values['wall_u_'+str(tag)]=metrics(q['u_wall'][mask],ref['wall_u'][mask],ref['wall_weight'][mask])
        active=point_block(base,domain,c['seed'],max(0,step//400-1),c['arm'])
        train_r=domain_residual(model,active['domain'],setting);n=int(active['uniform_count'])
        val_r=[domain_residual(model,x,setting) for x in validations]
        boundary_values,raw_boundary=boundary(model,bp,setting,bw)
        physics=dict(validation_seeds=validation_seeds,training_uniform=stats(train_r[:n]),
            training_probes=stats(train_r[n:]),training_weighted=stats(train_r,active['domain_weight']),
            validation=[stats(r) for r in val_r],regions=[region_stats(r,transfer.distance_masks(x,domain)) for x,r in zip(validations,val_r)],
            independent_boundary=boundary_values)
        physics['independent_objective']=[v['total_mean']+10*boundary_values['traction']+100*boundary_values['displacement'] for v in physics['validation']]
        physics['independent_training_uniform_ratio']=[v['total_mean']/max(physics['training_uniform']['total_mean'],1e-30) for v in physics['validation']]
        rows.append(dict(nominal_step=step,actual_steps=actual,early_stop=actual<step,metrics=values,physics=physics))
        if save_fields:
            np.savez_compressed(out/f'fields_{step:04d}.npz',**q,**{'error_'+k:v for k,v in errors.items()})
            arrays=dict(training=train_r)
            arrays.update({f'validation{i}':r for i,r in enumerate(val_r)})
            arrays.update({f'validation{i}_xy':x for i,x in enumerate(validations)})
            arrays.update({'boundary_'+k:v for k,v in raw_boundary.items()})
            np.savez_compressed(out/f'physics_{step:04d}.npz',**arrays)
    write(out/'evaluation.json',dict(configuration=c,results=rows))
    print(c['id'],'evaluated',len(rows),'prescribed endpoints',flush=True)
