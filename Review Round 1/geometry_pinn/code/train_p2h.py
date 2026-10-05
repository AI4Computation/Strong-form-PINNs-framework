"""Registered paired collocation-refresh intervention from frozen 400-step states."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,traceback
import numpy as np
import torch
from shared_geometry_features import make_shared_model
from p2_components import prepare_loss,parts_loss,objective
from prepare_p2 import independent_loss
from p2_observed_lbfgs import ObservedLBFGS
from cavity_cover import normalized_domain
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'results/R2_P2F_repetition_budget';B=ROOT/'results/R2_P2H_sampling_intervention';P=ROOT/'protocols/R2_P2H_sampling_intervention.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def utc():return datetime.now(timezone.utc).isoformat()
def main():
    assert not B.exists();p=read(P);fc=read(OLD/'completion.json');fm=read(OLD/'manifest.json')
    for f,h in fc['supporting_files_sha256'].items():assert sha(f)==h
    for f,h in fm['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in fm['prepared_sha256'].items():assert sha(OLD/f)==h
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    B.mkdir();domain,_,_=normalized_domain(read(ROOT/'inputs/cavity_geometries.json')['L1']);z=load(OLD/'L1_covers.npz');covers=(z['centres'],z['halfwidths'],(z['uniform_centres'],z['uniform_halfwidths']))
    def make(method,seed):return make_shared_model(method,seed,*covers)
    source=['train_p2h.py','prepare_p2.py','p2_components.py','p2_observed_lbfgs.py','shared_geometry_features.py','geometry_primitives.py','cavity_cover.py','sparse_mixed_pinn.py']
    pre=dict(passed=False,protocol_sha256=sha(P),source_sha256={f:sha(ROOT/'code'/f) for f in source},predecessor_sha256=sha(OLD/'completion.json'),inputs_sha256={},points_sha256={},numerical_checks={},point_generation_seconds={})
    allpoints={}
    for seed in p['seeds']:
        path=OLD/f'L1_seed{seed}_points.npz';pre['inputs_sha256'][str(path.resolve())]=sha(path);base=load(path);n=int(base['uniform_count']);blocks=[]
        for block in range(3):
            start=time.perf_counter();points={k:v.copy() for k,v in base.items()}
            points['domain'][:n]=domain.random_interior(n,np.random.default_rng(100000+seed+1000000*(block+1)))
            assert points['domain'].shape==(6000,2) and domain.contains(points['domain']).all()
            assert np.array_equal(points['domain'][n:],base['domain'][n:])
            assert all(np.array_equal(v,base[k]) for k,v in points.items() if k!='domain')
            assert abs(points['domain_weight'].sum()-1)<1e-12 and (points['domain_weight']>0).all()
            for c,h in [(covers[0],covers[1]),covers[2]]:
                assert np.exp(-.5*(((points['domain'][:,None]-c[None])/h[None])**2).sum(2)).max(0).min()>.1
            dest=B/f'L1_seed{seed}_refresh_block{block}_points.npz';np.savez_compressed(dest,**points);pre['points_sha256'][dest.name]=sha(dest);pre['point_generation_seconds'][dest.name]=time.perf_counter()-start;blocks.append(points)
            small={k:v.copy() for k,v in points.items()}
            for k in ['domain','left','right','top','bottom','hole','normal']:small[k]=small[k][:9]
            small['domain_weight']=np.full(9,1/9);small['hole_weight']=np.full(9,1/9)
            for method in p['methods']:
                initial=OLD/f'L1_seed{seed}_{method}/step_0400.pt';assert sha(initial)==fm['terminal_sha256'][f'L1_seed{seed}_{method}']['step_0400.pt'];pre['inputs_sha256'][str(initial.resolve())]=sha(initial)
                state=torch.load(initial,map_location='cpu',weights_only=True);model=make(method,seed).float();model.load_state_dict(state)
                assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items());model.double();assert sum(v.numel() for v in model.parameters())==110705
                fast=objective(parts_loss(model,prepare_loss(model,small),-1.,-5.));auto=independent_loss(model,small,-1.,-5.)
                fg=torch.autograd.grad(fast,tuple(model.parameters()));ag=torch.autograd.grad(auto,tuple(model.parameters()))
                le=abs(fast.item()-auto.item())/max(1,abs(auto.item()));ge=max(float((x-y).abs().max()) for x,y in zip(fg,ag))/max(1,max(float(x.abs().max()) for x in ag));assert max(le,ge)<1e-9
                model.float().cuda();prepared=prepare_loss(model,points);value=objective(parts_loss(model,prepared,-1.,-5.));value.backward();assert torch.isfinite(value) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
                pre['numerical_checks'][f'{seed}_{block}_{method}']=dict(objective_error=le,gradient_error=ge,full_gpu_finite=True)
                del model,state,prepared,fast,auto,fg,ag,value;torch.cuda.empty_cache()
        allpoints[seed]=(base,blocks);print('PREFLIGHT_SEED_PASS',seed,flush=True)
    pre['inputs_sha256'][str((OLD/'L1_covers.npz').resolve())]=sha(OLD/'L1_covers.npz');pre['passed']=True;write(B/'preflight.json',pre)
    manifest=dict(id=p['id'],status='running',created_utc=utc(),protocol_sha256=sha(P),preflight_sha256=sha(B/'preflight.json'),source_sha256=pre['source_sha256'],inputs_sha256=pre['inputs_sha256'],points_sha256=pre['points_sha256'],completed=[],active=None,terminal_sha256={},fem_read=False,timing_valid_for_comparison=False)
    write(B/'manifest.json',manifest)
    try:
        for si,seed in enumerate(p['seeds']):
            base,blocks=allpoints[seed];methods=p['methods'][si:]+p['methods'][:si]
            for method in methods:
                for arm in (p['arms'] if si%2==0 else p['arms'][::-1]):
                    name=f'L1_seed{seed}_{method}_{arm}';out=B/name;out.mkdir();manifest['active']=name;write(B/'manifest.json',manifest)
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model=make(method,seed).float().cuda()
                    initial=OLD/f'L1_seed{seed}_{method}/step_0400.pt';model.load_state_dict(torch.load(initial,map_location='cuda',weights_only=True))
                    trace=[];records=[];total_steps=0;total_closures=0
                    for block in range(3):
                        points=base if arm=='fixed_reset' else blocks[block];start=time.perf_counter();prepared=prepare_loss(model,points);torch.cuda.synchronize();prep_seconds=time.perf_counter()-start;closures=0;start=time.perf_counter()
                        def observer(step,evaluations,loss,gradient):
                            assert np.isfinite([loss,gradient]).all();trace.append(dict(block=block,block_step=step,cumulative_step=400+total_steps+step,continuation_closures=total_closures+evaluations,loss=loss,max_gradient=gradient))
                        opt=ObservedLBFGS(model.parameters(),observer=observer,**p['optimizer'])
                        def closure():
                            nonlocal closures
                            opt.zero_grad(set_to_none=True);v=objective(parts_loss(model,prepared,-1.,-5.));assert torch.isfinite(v);v.backward();closures+=1
                            assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters());return v
                        opt.step(closure);torch.cuda.synchronize();seconds=time.perf_counter()-start
                        assert opt.accepted_steps==400,(name,block,opt.accepted_steps,opt.stop_reason)
                        total_steps+=opt.accepted_steps;total_closures+=closures
                        with torch.no_grad():parts={k:float(v) for k,v in parts_loss(model,prepared,-1.,-5.).items()}
                        checkpoint=f'step_{400+total_steps:04d}.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/checkpoint)
                        records.append(dict(block=block,cumulative_step=400+total_steps,accepted_steps=opt.accepted_steps,closure_evaluations=closures,loss_parts=parts,stop_reason=opt.stop_reason,preparation_seconds=prep_seconds,optimization_seconds=seconds,point_file=str((OLD/f'L1_seed{seed}_points.npz' if arm=='fixed_reset' else B/f'L1_seed{seed}_refresh_block{block}_points.npz').resolve())))
                        print(name,'FROZEN',400+total_steps,'loss',trace[-1]['loss'],flush=True);del opt,prepared
                    prefix=read(OLD/f'L1_seed{seed}_{method}/trace.json');prefix_closures=next(v['closure_evaluations'] for v in prefix if v['accepted_step']==400)
                    result=dict(case='L1',seed=seed,method=method,arm=arm,initial_sha256=sha(initial),parameters=sum(v.numel() for v in model.parameters()),reused_prefix_steps=400,reused_prefix_closures=prefix_closures,continuation_steps=total_steps,continuation_closures=total_closures,blocks=records,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),fem_read=False,timing_valid_for_comparison=False)
                    write(out/'result.json',result);write(out/'trace.json',trace);manifest['completed'].append(name);manifest['terminal_sha256'][name]={f.name:sha(f) for f in out.iterdir() if f.is_file()};write(B/'manifest.json',manifest);del model;torch.cuda.empty_cache()
        manifest.update(status='fit_complete',active=None,finished_utc=utc());write(B/'manifest.json',manifest);print('ALL_18_CONTINUATIONS_FROZEN_NO_FEM',flush=True)
    except BaseException as exc:
        manifest.update(status='failed',failure=repr(exc),traceback=traceback.format_exc());write(B/'manifest.json',manifest);raise
if __name__=='__main__':main()
