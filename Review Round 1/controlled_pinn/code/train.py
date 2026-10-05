"""Sequential, resumable-by-completed-run execution of the frozen manifest."""
import runtime
from runtime import torch,DEVICE,sync
from models import *
from metrics import evaluate
from observed_lbfgs import ObservedLBFGS
import argparse,time,csv,gc,platform

def write_json(path,payload):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def legacy_reference(config):
    if config['geometry']=='tunnel':
        folder=PROJECT/'代码源文件/【隧道+溶洞】'
        u=np.loadtxt(folder/'displacement-karst.csv',delimiter=',',skiprows=1)
        s=np.loadtxt(folder/'stress-karst.csv',delimiter=',',skiprows=1)
    else:
        folder=PROJECT/'abaqus-36/processed'
        p,pt=config['p_lateral'],config['p_top']
        token=lambda x:str(int(abs(x)))
        u=np.loadtxt(folder/'displacement'/f'abaqus-p_lateral{token(p)}-p_top{token(pt)}.csv',delimiter=',',skiprows=1)
        s=np.loadtxt(folder/'stress'/f'abaqus-S-p_lateral{token(p)}-p_top{token(pt)}.csv',delimiter=',',skiprows=1)
    return {'xy_u':u[:,:2],'u':u[:,2:4],'xy_s':s[:,:2],'s':s[:,2:5]}

def reference(config,pilot=False):
    if pilot:return legacy_reference(config)
    approval=json.loads((ROOT/'config/reference_selection.json').read_text())
    if not approval[config['geometry']]['validated']:raise RuntimeError('FEM validation is required before formal training.')
    return dict(np.load(ROOT/'fem/references'/f"{config['case']}.npz"))

def code_hashes():
    names=['runtime.py','models.py','metrics.py','observed_lbfgs.py','train.py']
    out={p:hashlib.sha256((ROOT/'code'/p).read_bytes()).hexdigest() for p in names}
    out['legacy_circle']=hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    return out

def run_one(config,pilot=False):
    root=ROOT/('checks/pilot' if pilot else 'runs')/config['id']
    root.mkdir(parents=True,exist_ok=True)
    result_path=root/'result.json'
    hashes=code_hashes()
    if result_path.exists():
        saved=json.loads(result_path.read_text(encoding='utf-8'))
        if saved['code_hashes']!=hashes or saved['configuration']!=config:
            raise RuntimeError('Existing result uses a different protocol: '+str(root))
        return saved
    ref=reference(config,pilot)
    ref_hash=None if pilot else hashlib.sha256((ROOT/'fem/references'/f"{config['case']}.npz").read_bytes()).hexdigest()
    gc.collect();torch.cuda.empty_cache()
    model=build_model(config['method'],config['seed'],config['geometry'])
    samples=samples_for(config['seed'],config['geometry'])
    parameters=[p for p in model.parameters() if p.requires_grad]
    history=[];trace=[];closures=[];diagnostic_s=0.;last_loss=None
    checkpoint_steps={0,10,100,500}
    if config['geometry']=='tunnel':checkpoint_steps.add(2000)
    save_diagnostics=config['case'] in ('C1','C8','T1')
    torch.cuda.reset_peak_memory_stats();sync();started=time.perf_counter()

    def checkpoint(step):
        torch.save({'configuration':config,'accepted_step':step,
                    'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},root/f'step_{step:05d}.pt')

    def observe(step,evals,loss,gradient_max,final=False):
        nonlocal diagnostic_s,last_loss
        sync();observed=time.perf_counter();wall=observed-started;opt=wall-diagnostic_s
        last_loss=loss
        row=dict(accepted_step=step,closure_evaluations=evals,loss=loss,gradient_max=gradient_max,
                 optimization_s=opt,state_wall_s=wall)
        # Predefined schedule: every 50 accepted updates, the early 10-step point,
        # and termination. Crossings are interval-censored between these states.
        measure=step%50==0 or step==10 or final
        if measure:
            row.update(evaluate(model,ref,config['geometry']))
            sync();row['available_wall_s']=time.perf_counter()-started
            trace.append(row.copy())
        if save_diagnostics and (step in checkpoint_steps):checkpoint(step)
        if step%100==0 or final:
            write_json(root/'progress.json',{'configuration':config,'latest':row,'status':'running'})
        sync();diagnostic_s+=time.perf_counter()-observed
        if history and history[-1]['accepted_step']==step:history[-1].update(row)
        else:history.append(row)

    optimizer=ObservedLBFGS(parameters,observer=observe,max_iter=config['max_iter'],
        history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-12)
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss=objective(model,samples,config)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite loss: '+config['id'])
        loss.backward();closures.append(float(loss.detach()));return loss
    optimizer.step(closure)
    sync();optimizer_end=time.perf_counter();optimizer_wall=optimizer_end-started
    optimization_s=optimizer_wall-diagnostic_s
    state=optimizer.state[parameters[0]]
    step=optimizer.accepted_steps
    # Always evaluate the final accepted parameters, even when the last closure
    # evaluated a rejected trial or termination preceded a parameter update.
    if trace[-1]['accepted_step']!=step:
        observe(step,state['func_evals'],last_loss,history[-1]['gradient_max'],final=True)
    values={k:v for k,v in trace[-1].items() if k.startswith(('u_','s_'))}
    sync();peak=float(torch.cuda.max_memory_allocated()/1024**2)
    t0=time.perf_counter();checkpoint(step);sync();diagnostic_s+=time.perf_counter()-t0
    final_wall=time.perf_counter()-started
    thresholds={}
    for metric in ['u_vector_pct','s_vector_pct']:
        for target in [5.,2.,1.]:
            valid=[(i,r) for i,r in enumerate(trace) if r[metric] is not None and r[metric]<=target]
            name=f'{metric}_le_{target:g}'
            if valid:
                i,row=valid[0];before=trace[i-1] if i else row
                thresholds[name]={'reached':True,'accepted_step':row['accepted_step'],
                    'optimization_s_upper':row['optimization_s'],'optimization_s_lower':before['optimization_s'],
                    'observed_wall_s':row['available_wall_s']}
            else:thresholds[name]={'reached':False}
    payload={'configuration':config,'code_hashes':hashes,'status':'complete','pilot':pilot,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name(0),'mkl_threading':'SEQUENTIAL','cpu_torch_threads':torch.get_num_threads(),
            'tf32':False,'deterministic_algorithms':True},
        'metrics':values,'thresholds':thresholds,'reference_sha256':ref_hash,'accepted_steps':step,'closure_evaluations':len(closures),
        'optimizer_attempted_iterations':state['n_iter'],'stop_reason':optimizer.stop_reason,
        'max_eval':optimizer.param_groups[0]['max_eval'],'trainable_parameters':sum(p.numel() for p in parameters),
        'fixed_coefficients':sum(v.numel() for v in model.buffers()),'final_loss':last_loss,
        'optimization_s':optimization_s,'optimizer_wall_s':optimizer_wall,'total_wall_s':final_wall,
        'diagnostic_and_checkpoint_s':diagnostic_s,'peak_cuda_allocated_MiB':peak,
        'peak_cuda_reserved_MiB':float(torch.cuda.max_memory_reserved()/1024**2),
        'collocation_hashes':{k:array_hash(v) for k,v in samples.items()},
        'metric_trace':trace,'accepted_history':history,'closure_losses':closures}
    write_json(result_path,payload)
    write_json(root/'progress.json',{'status':'complete','accepted_steps':step,'stop_reason':optimizer.stop_reason})
    print(json.dumps({'id':config['id'],'accepted_steps':step,'closures':len(closures),'optimization_s':optimization_s,
                     'u_pct':values['u_vector_pct'],'s_pct':values['s_vector_pct'],'stop':optimizer.stop_reason}),flush=True)
    return payload

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--max-iter',type=int)
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--geometry',choices=['circle','tunnel'])
    args=parser.parse_args()
    if args.max_iter and not args.pilot:raise ValueError('Formal iteration budget is frozen in the manifest.')
    manifest=json.loads((ROOT/'config/run_manifest.json').read_text())
    if args.geometry:manifest=[c for c in manifest if c['geometry']==args.geometry]
    if args.ids:manifest=[c for c in manifest if c['id'] in args.ids]
    if args.limit:manifest=manifest[:args.limit]
    for config in manifest:
        if args.pilot:config={**config,'max_iter':args.max_iter or 100}
        print('START '+config['id'],flush=True)
        run_one(config,pilot=args.pilot)

if __name__=='__main__':main()
