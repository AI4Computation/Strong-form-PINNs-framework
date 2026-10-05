"""Single-seed T1/S1 development transfer, common unchanged representations."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,traceback
import numpy as np
import torch
from shared_geometry_features import make_shared_model
from p2_components import uniform_cover,objective,prepare_loss,parts_loss
from p2_observed_lbfgs import ObservedLBFGS
from cavity_cover import normalized_domain
from sparse_mixed_pinn import residual
from transfer_mechanics import sample,prepare,parts,independent_objective
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2I_shape_transfer';D=ROOT/'results/R2_P2D_shared_features';OLD=ROOT.parents[1]/'research';P=ROOT/'protocols/R2_P2I_shape_transfer.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def utc():return datetime.now(timezone.utc).isoformat()
def main():
    assert not B.exists();p=read(P);dc=read(D/'completion.json');geometries=read(ROOT/'inputs/cavity_geometries.json')
    for f,h in dc['supporting_files_sha256'].items():assert sha(f)==h
    dm=read(D/'manifest.json')
    for f,h in dm['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    # Reference provenance only; no reference solution arrays used before fitting.
    geomconfig=OLD/'controlled_pinn/config/geometry.json';g=read(geomconfig);assert g['nu']==p['cases']['T1']['nu'] and g['length_scale_m']==75
    assert abs(g['E_MPa']*g['displacement_scale_m']/(g['stress_scale_MPa']*g['length_scale_m'])-1)<1e-14
    assert g['side_pressure_MPa']/g['stress_scale_MPa']==4 and g['water_pressure_MPa']/g['stress_scale_MPa']==1
    refs=[OLD/'controlled_pinn/fem/tr3_tunnel_tq0p125_Load.npz',OLD/'validation_and_cost/shape_references/full_integration/tsf_square_g3_Load.npz',OLD/'validation_and_cost/shape_references/tsr_square_g3_mesh.npz']
    assert all(f.exists() for f in refs)
    inventory=read(ROOT.parent/'00_基线与规则/第一轮冻结清单.json');expected={v['relative_path']:v['sha256'] for v in inventory['files']}
    for f in refs+[geomconfig]:assert sha(f)==expected[f.relative_to(OLD).as_posix()]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;B.mkdir()
    sources=['train_p2i.py','transfer_mechanics.py','p2_components.py','shared_geometry_features.py','sparse_mixed_pinn.py','geometry_primitives.py','cavity_cover.py','p2_observed_lbfgs.py']
    pre=dict(passed=False,protocol_sha256=sha(P),source_sha256={f:sha(ROOT/'code'/f) for f in sources},reference_file_sha256={str(f.resolve()):sha(f) for f in refs},config_sha256=sha(geomconfig),fem_solution_arrays_read=False,numerical={},prepared_sha256={},inputs_sha256={})
    cache={}
    for case,setting in p['cases'].items():
        domain,_,_=normalized_domain(geometries[case]);path=D/f'{case}_covers.npz';assert sha(path)==dc['files_sha256'][path.name];z=load(path);pre['inputs_sha256'][str(path.resolve())]=sha(path)
        c,h=z['centres'],z['halfwidths'];pc,ph,_=uniform_cover(domain,len(c));base=sample(domain,[(c,h),(pc,ph)],p['seed'],setting);n=int(base['uniform_count']);blocks=[]
        E,nu=setting['E'],setting['nu'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
        grad=np.array([[.13,-.07],[.11,-.19]]);sigma=lam*np.trace(grad)*np.eye(2)+mu*(grad+grad.T)
        q=torch.tensor([[0.,0.,sigma[0,0],sigma[1,1],sigma[0,1]]],dtype=torch.float64);j=torch.zeros((1,5,2),dtype=torch.float64);j[0,:2]=torch.tensor(grad)
        assert float(abs(residual(q,j,E,nu)).max())<1e-12
        assert set(base['hole_tags'])==set(setting['hole_normal_stress'])
        for tag in setting['hole_normal_stress']:
            mask=base['hole_tags']==tag;assert abs(base['hole_weight'][mask].sum()-1)<1e-12
            stress=setting['hole_normal_stress'][tag]*np.eye(2);np.testing.assert_allclose(base['normal'][mask]@stress.T,base['normal'][mask]*base['hole_normal_stress'][mask,None],atol=1e-15)
        for block in range(4):
            points={k:v.copy() for k,v in base.items()}
            if block:points['domain'][:n]=domain.random_interior(n,np.random.default_rng(100000+p['seed']+1000000*block))
            assert len(points['domain'])==6000 and domain.contains(points['domain']).all();assert np.array_equal(points['domain'][n:],base['domain'][n:]);assert all(np.array_equal(v,base[k]) for k,v in points.items() if k!='domain')
            assert abs(points['domain_weight'].sum()-1)<1e-12
            for cc,hh in [(c,h),(z['uniform_centres'],z['uniform_halfwidths'])]:assert np.exp(-.5*(((points['domain'][:,None]-cc[None])/hh[None])**2).sum(2)).max(0).min()>.1
            dest=B/f'{case}_block{block}_points.npz';np.savez_compressed(dest,**points);pre['prepared_sha256'][dest.name]=sha(dest);blocks.append(points)
        init=None
        for method in p['methods']:
            model=make_shared_model(method,p['seed'],c,h,(z['uniform_centres'],z['uniform_halfwidths']));assert sum(v.numel() for v in model.parameters())==110705
            weights=[v.detach().clone() for v in model.net.parameters()]
            if init is None:init=weights
            else:assert all(torch.equal(x,y) for x,y in zip(init,weights))
            small={k:v.copy() for k,v in base.items()}
            for key in ['domain','left','right','top','bottom']:small[key]=small[key][:10]
            ix=np.r_[np.arange(5),np.arange(len(base['hole'])-5,len(base['hole']))]
            for key in ['hole','normal','hole_tags','hole_normal_stress','hole_weight']:small[key]=small[key][ix]
            small['domain_weight']=np.full(10,.1)
            for tag in set(small['hole_tags']):mask=small['hole_tags']==tag;small['hole_weight'][mask]/=small['hole_weight'][mask].sum()
            fast=objective(parts(model,prepare(model,small),setting));auto=independent_objective(model,small,setting);fg=torch.autograd.grad(fast,tuple(model.parameters()));ag=torch.autograd.grad(auto,tuple(model.parameters()))
            le=abs(fast.item()-auto.item())/max(1,abs(auto.item()));ge=max(float(abs(x-y).max()) for x,y in zip(fg,ag))/max(1,max(float(abs(v).max()) for v in ag));assert max(le,ge)<1e-9
            if case=='S1':
                old=objective(parts_loss(model,prepare_loss(model,small),-1.,-5.));og=torch.autograd.grad(old,tuple(model.parameters()));assert abs(old.item()-fast.item())<1e-10 and max(float(abs(x-y).max()) for x,y in zip(og,fg))<1e-9;del old,og
            model.float().cuda();pr=prepare(model,base);loss=objective(parts(model,pr,setting));loss.backward();assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
            pre['numerical'][case+'_'+method]=dict(objective_error=le,gradient_error=ge,full_gpu_finite=True,S1_matches_original=case=='S1');del model,fast,auto,fg,ag,pr,loss;torch.cuda.empty_cache()
        cache[case]=(z,blocks);print('PREFLIGHT_PASS',case,'features',len(c),'tags',list(setting['hole_normal_stress']),flush=True)
    pre['passed']=True;write(B/'preflight.json',pre)
    manifest=dict(id=p['id'],status='running',created_utc=utc(),source_sha256=pre['source_sha256'],protocol_sha256=sha(P),preflight_sha256=sha(B/'preflight.json'),inputs_sha256=pre['inputs_sha256'],prepared_sha256=pre['prepared_sha256'],completed=[],active=None,terminal_sha256={},fem_read=False,timing_valid_for_comparison=False)
    write(B/'manifest.json',manifest)
    try:
        for case,setting in p['cases'].items():
            z,blocks=cache[case]
            for method in p['methods']:
                first=None
                for arm in p['arms']:
                    name=f'{case}_seed{p["seed"]}_{method}_{arm}';out=B/name;out.mkdir();manifest['active']=name;write(B/'manifest.json',manifest);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    model=make_shared_model(method,p['seed'],z['centres'],z['halfwidths'],(z['uniform_centres'],z['uniform_halfwidths'])).float().cuda();trace=[];records=[];total=0;closures_total=0
                    for block in range(4):
                        points=blocks[block if arm=='uniform_refresh' else 0];start=time.perf_counter();pr=prepare(model,points);torch.cuda.synchronize();prepsec=time.perf_counter()-start;closures=0;start=time.perf_counter()
                        def observer(step,evaluations,loss,gradient):
                            assert np.isfinite([loss,gradient]).all();trace.append(dict(block=block,block_step=step,cumulative_step=total+step,closure_evaluations=closures_total+evaluations,loss=loss,max_gradient=gradient))
                        opt=ObservedLBFGS(model.parameters(),observer=observer,**p['optimizer'])
                        def closure():
                            nonlocal closures
                            opt.zero_grad(set_to_none=True);v=objective(parts(model,pr,setting));assert torch.isfinite(v);v.backward();closures+=1;assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in model.parameters());return v
                        opt.step(closure);torch.cuda.synchronize();sec=time.perf_counter()-start;assert opt.accepted_steps==400,(name,block,opt.stop_reason);total+=400;closures_total+=closures
                        state={k:v.detach().cpu() for k,v in model.state_dict().items()};torch.save(state,out/f'step_{total:04d}.pt')
                        if block==0:
                            if first is None:first=state
                            else:assert all(torch.equal(v,first[k]) for k,v in state.items())
                        with torch.no_grad():loss_parts={k:float(v) for k,v in parts(model,pr,setting).items()}
                        records.append(dict(block=block,cumulative_step=total,accepted_steps=400,closures=closures,stop_reason=opt.stop_reason,point_block=block if arm=='uniform_refresh' else 0,loss_parts=loss_parts,preparation_seconds=prepsec,optimization_seconds=sec));print(name,'FROZEN',total,'loss',trace[-1]['loss'],flush=True);del opt,pr,state
                    write(out/'trace.json',trace);write(out/'result.json',dict(case=case,method=method,arm=arm,seed=p['seed'],parameters=110705,accepted_steps=total,closure_evaluations=closures_total,blocks=records,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),fem_read=False,timing_valid_for_comparison=False))
                    manifest['completed'].append(name);manifest['terminal_sha256'][name]={f.name:sha(f) for f in out.iterdir() if f.is_file()};write(B/'manifest.json',manifest);del model;torch.cuda.empty_cache()
        manifest.update(status='fit_complete',active=None,finished_utc=utc());write(B/'manifest.json',manifest);print('ALL_12_TRANSFER_TRAJECTORIES_FROZEN_NO_FEM',flush=True)
    except BaseException as exc:
        manifest.update(status='failed',failure=repr(exc),traceback=traceback.format_exc());write(B/'manifest.json',manifest);raise
if __name__=='__main__':main()
