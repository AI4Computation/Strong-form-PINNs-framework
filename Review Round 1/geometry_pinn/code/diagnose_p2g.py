"""Registered frozen-field physical diagnosis: no FEM, fitting, or model selection."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,traceback
import numpy as np
import torch
from p2_components import residual
from shared_geometry_features import make_shared_model
from cavity_cover import normalized_domain
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2F_repetition_budget';OUT=ROOT/'results/R2_P2G_frozen_physics';P=ROOT/'protocols/R2_P2G_frozen_physics.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def stats(r,w=None):
    if r.ndim==1:r=r[:,None]
    e=np.sum(r*r,axis=1);w=np.ones(len(e))/len(e) if w is None else w/w.sum()
    order=np.argsort(e,kind='stable');quant=np.interp([.5,.9,.95,.99,1.],np.cumsum(w[order]),np.sqrt(e[order]))
    tail=np.argsort(-e,kind='stable');prior=np.cumsum(w[tail])-w[tail];fraction=np.minimum(w[tail],np.maximum(0.,.01-prior))
    return dict(total_mean=float(w@e),component_mean=(w[:,None]*r*r).sum(0).tolist(),norm_quantiles=quant.tolist(),top_one_percent_mass=float(fraction@e[tail]/max(w@e,1e-30)))
def regions(x,domain):
    d=np.full(len(x),np.inf);corners=[]
    for hole in domain.holes:
        if hole['kind']=='ellipse':
            assert abs(hole['axes'][0]-hole['axes'][1])<1e-12,'This diagnostic registers circles and polygons only'
            d=np.minimum(d,abs(np.linalg.norm(x-hole['center'],axis=1)-hole['axes'][0]))
        else:
            v=np.asarray(hole['vertices']);corners.extend(v)
            for a,b in zip(v,np.roll(v,-1,axis=0)):
                t=np.clip((x-a)@(b-a)/np.sum((b-a)**2),0,1);d=np.minimum(d,np.linalg.norm(x-a-t[:,None]*(b-a),axis=1))
    corner=np.zeros(len(x),bool) if not corners else np.min(np.linalg.norm(x[:,None]-np.asarray(corners)[None],axis=2),axis=1)<.02
    return dict(near_wall=d<=.05,far_wall=d>.05,corner=corner,away_corner=~corner)
def region_stats(r,masks):
    e=(r*r).sum(1);out={}
    for name,mask in masks.items():
        if mask.any():out[name]=dict(**stats(r[mask]),point_fraction=float(mask.mean()),residual_square_mass=float(e[mask].sum()/max(e.sum(),1e-30)))
    return out
@torch.no_grad()
def domain_residual(model,xy):
    out=np.empty((len(xy),5),float)
    for start in range(0,len(xy),1024):
        sl=slice(start,start+1024);q,j=model.sparse(model.prepare(xy[sl]));out[sl]=residual(q,j).cpu().numpy()
    assert np.isfinite(out).all();return out
@torch.no_grad()
def field(model,x):
    out=np.empty((len(x),5),float)
    for start in range(0,len(x),1024):
        sl=slice(start,start+1024);t=torch.as_tensor(x[sl],dtype=next(model.parameters()).dtype,device=next(model.parameters()).device);out[sl]=model(t).cpu().numpy()
    assert np.isfinite(out).all();return out
def boundary(model,points,lateral,top,weights):
    q={k:field(model,points[k]) for k in ['left','right','top','bottom','hole','gauge']};r={}
    for k in ['left','right']:r[k]=np.column_stack([q[k][:,2]-lateral,q[k][:,4]])
    r['top']=np.column_stack([q['top'][:,3]-top,q['top'][:,4]])
    r['bottom_shear']=q['bottom'][:,4,None];r['bottom_uy']=q['bottom'][:,1,None]
    r['gauge_ux']=q['gauge'][:,0,None];nx,ny=points['normal'].T
    r['hole']=np.column_stack([q['hole'][:,2]*nx+q['hole'][:,4]*ny,q['hole'][:,4]*nx+q['hole'][:,3]*ny])
    out={k:stats(v,weights.get(k)) for k,v in r.items()}
    traction=sum(out[k]['total_mean'] for k in ['left','right','top','bottom_shear','hole'])
    disp=out['bottom_uy']['total_mean']+out['gauge_ux']['total_mean']
    return dict(traction=traction,displacement=disp,constraints=out),r
def independent_boundary(domain):
    points={};weights={};holes=[]
    for curve in domain.curves:
        length=float(curve.quadrature(16)['w'].sum());q=curve.quadrature(max(1,int(np.ceil(length/.002))),3)
        if curve.hole:holes.append(q)
        else:points[curve.tag]=q['xy'];weights[curve.tag]=q['w']/q['w'].sum()
    points['hole']=np.vstack([q['xy'] for q in holes]);points['normal']=np.vstack([q['normal'] for q in holes]);w=np.concatenate([q['w'] for q in holes]);weights['hole']=w/w.sum()
    points['gauge']=np.array([[0.,-.5]]);weights['bottom_shear']=weights['bottom'];weights['bottom_uy']=weights['bottom']
    return points,weights
def numerical_audit(model,xy,r):
    idx=np.unique(np.r_[np.argsort(-np.sum(r*r,axis=1),kind='stable')[:8],np.random.default_rng(9274103).choice(len(xy),8,replace=False)])
    model.cpu().double()
    with torch.no_grad():q,j=model.sparse(model.prepare(xy[idx]));analytic=residual(q,j).numpy()
    with torch.enable_grad():
        x=torch.tensor(xy[idx],dtype=torch.float64,requires_grad=True);v=model(x)
        jac=torch.stack([torch.autograd.grad(v[:,i].sum(),x,retain_graph=True)[0] for i in range(5)],1)
    v=v.detach().numpy();j=jac.detach().numpy();mu=1.333/(2*(1+.3333));lam=1.333*.3333/((1+.3333)*(1-2*.3333))
    stress=np.column_stack([(lam+2*mu)*j[:,0,0]+lam*j[:,1,1],lam*j[:,0,0]+(lam+2*mu)*j[:,1,1],mu*(j[:,0,1]+j[:,1,0])])
    independent=np.column_stack([j[:,2,0]+j[:,4,1],j[:,4,0]+j[:,3,1],v[:,2:]-stress]);scale=max(1.,float(abs(independent).max()))
    e64=float(abs(analytic-independent).max()/scale);e32=float(abs(r[idx]-independent).max()/scale)
    assert e64<1e-9 and e32<3e-4,(e64,e32)
    return dict(float64_scaled_max_error=e64,float32_scaled_max_error=e32,passed=True),dict(indices=idx,xy=xy[idx],analytic_float64=analytic,autograd_float64=independent,saved_float32=r[idx])
def main():
    assert not OUT.exists();p=read(P);fp=read(ROOT/'protocols/R2_P2F_repetition_budget.json');f=read(B/'completion.json');m=read(B/'manifest.json')
    assert f['evaluation_fields']==81 and m['status']=='fit_complete'
    for path,h in f['supporting_files_sha256'].items():assert sha(path)==h
    for name,h in m['source_sha256'].items():assert sha(ROOT/'code'/name)==h
    for name,h in m['prepared_sha256'].items():assert sha(B/name)==h
    for run,files in m['terminal_sha256'].items():
        for file,h in files.items():assert sha(B/run/file)==h
    OUT.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=['diagnose_p2g.py','p2_components.py','sparse_mixed_pinn.py','shared_geometry_features.py','geometry_primitives.py','cavity_cover.py']
    report=dict(protocol_sha256=sha(P),source_sha256={v:sha(ROOT/'code'/v) for v in source},input_completion_sha256=sha(B/'completion.json'),geometry_sha256=sha(ROOT/'inputs/cavity_geometries.json'),fem_read=False,training_runs=0,formal_timing=False,results={},points_sha256={})
    geometries=read(ROOT/'inputs/cavity_geometries.json');cache={};completed=0
    for geometry in ['C1','L1']:
        domain,_,_=normalized_domain(geometries[geometry]);v=[domain.random_interior(p['independent_points']['count_per_set'],np.random.default_rng(seed)) for seed in p['independent_points']['seeds']]
        masks=[regions(x,domain) for x in v];bp,bw=independent_boundary(domain)
        arrays={f'validation{i}':x for i,x in enumerate(v)}
        for i,mask in enumerate(masks):arrays.update({f'validation{i}_{k}':val for k,val in mask.items()})
        arrays.update({'boundary_'+k:x for k,x in bp.items()});arrays.update({'weight_'+k:x for k,x in bw.items()})
        path=OUT/f'{geometry}_points.npz';np.savez_compressed(path,**arrays);report['points_sha256'][path.name]=sha(path);cache[geometry]=(v,masks,bp,bw)
    try:
        for case in p['cases']:
            setting=fp['case_settings'][case];geometry=setting['geometry'];valid,masks,bp,bw=cache[geometry];cover=load(B/f'{geometry}_covers.npz')
            for seed in p['seeds']:
                train=load(B/f'{geometry}_seed{seed}_points.npz');n=int(train['uniform_count']);tw={'hole':train['hole_weight']}
                for step in p['steps']:
                    for method in p['methods']:
                        name=f'{case}_seed{seed}_{method}';tag=name+f'_step{step:04d}';checkpoint=B/name/f'step_{step:04d}.pt'
                        write(OUT/'progress.json',dict(status='running',completed_fields=completed,expected_fields=81,active=tag))
                        model=make_shared_model(method,seed,cover['centres'],cover['halfwidths'],(cover['uniform_centres'],cover['uniform_halfwidths'])).float().cuda()
                        model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));model.eval()
                        rt=domain_residual(model,train['domain']);rv=[domain_residual(model,x) for x in valid]
                        tb,tr=boundary(model,train,setting['lateral'],setting['top'],tw);vb,vr=boundary(model,bp,setting['lateral'],setting['top'],bw)
                        tstats=stats(rt,train['domain_weight']);trainobj=tstats['total_mean']+10*tb['traction']+100*tb['displacement']
                        trace=read(B/name/'trace.json');recorded=next(row['loss'] for row in trace if row['accepted_step']==step)
                        assert abs(trainobj-recorded)<2e-5*max(.01,abs(recorded)),(tag,trainobj,recorded)
                        audit,aa=numerical_audit(model,valid[0],rv[0]);tu=stats(rt[:n]);tp=stats(rt[n:]);vs=[stats(v) for v in rv]
                        row=dict(case=case,seed=seed,step=step,method=method,checkpoint_sha256=sha(checkpoint),training_uniform=tu,training_probes=tp,training_weighted=tstats,
                            validation=vs,regions=[region_stats(r,mask) for r,mask in zip(rv,masks)],training_boundary=tb,independent_boundary=vb,
                            training_objective_recomputed=trainobj,training_trace_objective=recorded,training_objective_difference=abs(trainobj-recorded),
                            independent_objective=[v['total_mean']+10*vb['traction']+100*vb['displacement'] for v in vs],
                            independent_training_uniform_ratio=[v['total_mean']/tu['total_mean'] for v in vs],numerical_audit=audit)
                        raw=dict(training=rt,validation0=rv[0],validation1=rv[1]);raw.update({'training_boundary_'+k:x for k,x in tr.items()});raw.update({'independent_boundary_'+k:x for k,x in vr.items()})
                        raw.update({'audit_'+k:x for k,x in aa.items()});dest=OUT/f'{tag}_residuals.npz';np.savez_compressed(dest,**raw);row['raw_sha256']=sha(dest)
                        write(OUT/f'{tag}_metrics.json',row);report['results'][tag]=row;completed+=1
                        print(tag,'train/independent',tu['total_mean'],[v['total_mean'] for v in vs],'boundary',vb['traction'],flush=True)
                        del model;torch.cuda.empty_cache()
        assert completed==81;report['finished_utc']=datetime.now(timezone.utc).isoformat();write(OUT/'analysis.json',report)
        write(OUT/'progress.json',dict(status='complete',completed_fields=completed,expected_fields=81,active=None));print('ALL_81_PHYSICAL_DIAGNOSTICS_AND_NUMERICAL_AUDITS_COMPLETE',flush=True)
    except BaseException as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
