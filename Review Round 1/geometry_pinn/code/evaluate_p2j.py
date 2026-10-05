"""All frozen T1/S1 transfer endpoints, with material-aware reference/physics audits."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,traceback
import numpy as np
import torch
from shared_geometry_features import make_shared_model
from cavity_cover import normalized_domain
from p2_fem_interpolation import Field
from transfer_evaluation_helpers import from_elements,numerical_audit
from transfer_mechanics import distance_masks,wall_points,prepare,parts
from sparse_mixed_pinn import residual
from p2_components import objective
from diagnose_p2g import stats,region_stats,field
from evaluate_p2 import predict,metrics
from evaluate_p2d_engineering import score
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2I_shape_transfer';OUT=ROOT/'results/R2_P2J_reference_repair/evaluation';OLD=ROOT.parents[1]/'research';P=ROOT/'protocols/R2_P2I_shape_transfer.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}

@torch.no_grad()
def domain_residual(model,xy,setting):
    out=np.empty((len(xy),5),float)
    for start in range(0,len(xy),1024):
        sl=slice(start,start+1024);q,j=model.sparse(model.prepare(xy[sl]));out[sl]=residual(q,j,setting['E'],setting['nu']).cpu().numpy()
    assert np.isfinite(out).all();return out

def independent_boundary(domain,setting):
    points={};weights={};holes=[];tags=[]
    for curve in domain.curves:
        length=float(curve.quadrature(16)['w'].sum());q=curve.quadrature(max(1,int(np.ceil(length/.002))),3)
        if curve.hole:holes.append(q);tags.extend([curve.tag]*len(q['xy']))
        else:points[curve.tag]=q['xy'];weights[curve.tag]=q['w']/q['w'].sum()
    points.update(hole=np.vstack([q['xy'] for q in holes]),normal=np.vstack([q['normal'] for q in holes]),hole_tags=np.array(tags),hole_normal_stress=np.array([setting['hole_normal_stress'][t] for t in tags]),gauge=np.array([[0.,-.5]]))
    weights['hole']=np.concatenate([q['w'] for q in holes])
    for tag in set(tags):m=points['hole_tags']==tag;weights['hole'][m]/=weights['hole'][m].sum()
    weights['bottom_shear']=weights['bottom'];weights['bottom_uy']=weights['bottom'];return points,weights

def boundary(model,points,setting,weights):
    q={k:field(model,points[k]) for k in ['left','right','top','bottom','hole','gauge']};r={}
    for k in ['left','right']:r[k]=np.column_stack([q[k][:,2]-setting['lateral'],q[k][:,4]])
    r['top']=np.column_stack([q['top'][:,3]-setting['top'],q['top'][:,4]])
    r.update(bottom_shear=q['bottom'][:,4,None],bottom_uy=q['bottom'][:,1,None],gauge_ux=q['gauge'][:,0,None])
    nx,ny=points['normal'].T;h=q['hole'];pressure=points['hole_normal_stress']
    r['hole']=np.column_stack([h[:,2]*nx+h[:,4]*ny-pressure*nx,h[:,4]*nx+h[:,3]*ny-pressure*ny])
    out={k:stats(v,weights.get(k)) for k,v in r.items() if k!='hole'}
    for tag in setting['hole_normal_stress']:
        mask=points['hole_tags']==tag;out['hole_'+tag]=stats(r['hole'][mask],weights['hole'][mask])
    traction=sum(out[k]['total_mean'] for k in ['left','right','top','bottom_shear'])+sum(out['hole_'+t]['total_mean'] for t in setting['hole_normal_stress'])
    return dict(traction=traction,displacement=out['bottom_uy']['total_mean']+out['gauge_ux']['total_mean'],constraints=out),r

def reference(case,domain,setting):
    if case=='T1':paths=[OLD/'controlled_pinn/fem/tr3_tunnel_tq0p125_Load.npz']
    else:paths=[OLD/'validation_and_cost/shape_references/full_integration/tsf_square_g3_Load.npz',OLD/'validation_and_cost/shape_references/tsr_square_g3_mesh.npz']
    data=load(paths[0]);L,U,S=[setting[k] for k in ['length_scale','displacement_scale','stress_scale']]
    data['xy_u']=data['xy_u']/L;data['xy_s']=data['xy_s']/L;data['u']=data['u']/U;data['s']=data['s']/S;data['volume']=data['volume']/L**2
    if case=='T1':
        exactpath=OUT.parent/'T1_exact_mesh.npz';mesh=load(exactpath);assert np.array_equal(data['connectivity'],mesh['connectivity']);data['xy_u']=mesh['xy'];paths.append(exactpath)
    if case=='S1':
        mesh=load(paths[1]);assert np.array_equal(data['connectivity'],mesh['connectivity']-1);data['xy_u']=mesh['xy']
    ids=data['element_ids'];order=np.argsort(ids);elements=order[np.searchsorted(ids[order],data['ip_id'][:,0])];assert np.array_equal(ids[elements],data['ip_id'][:,0])
    uip,scheck,pos=from_elements(data,data['xy_s'],elements,setting['E'],setting['nu']);w=data['volume'];reconstruction=metrics(scheck,data['s'],w)['relative_l2_percent']
    preliminary=dict(case=case,source_sha256={str(p.resolve()):sha(p) for p in paths},stress_reconstruction_percent=reconstruction,maximum_position_defect=pos,area=float(w.sum()),geometry_area=float(domain.area),integration_points=len(w))
    write(OUT/f'{case}_reference_checks.json',preliminary)
    assert reconstruction<.02,(case,'stress reconstruction %',reconstruction)
    assert (w>0).all() and abs(w.sum()-domain.area)<1e-5,(case,'area discrepancy',w.sum()-domain.area)
    wall,normal,ww,tags,extrema,en=wall_points(domain);actual=np.vstack([wall,extrema]);normals=np.vstack([normal,en])
    if case=='T1':f=Field(data,setting['E'],setting['nu'])
    def boundary_ref(offset):
        query=actual-offset*normals
        if case=='T1':return f.evaluate(query,k=64)[0]
        i=np.searchsorted(mesh['x'],query[:,0],side='right')-1;j=np.searchsorted(mesh['y'],query[:,1],side='right')-1
        eid=mesh['grid_element_ids'][i,j];assert (eid>0).all();return from_elements(data,query,eid-1,setting['E'],setting['nu'])[0]
    delta=read(ROOT/'protocols/R2_P2J_reference_repair.json')['derived_offsets'][case]
    wu=boundary_ref(delta);shifted=boundary_ref(2*delta);halved=boundary_ref(delta/2);sensitivity=float(abs(wu-shifted).max());half_sensitivity=float(abs(wu-halved).max());assert max(sensitivity,half_sensitivity)<1e-5
    np.savez_compressed(OUT/f'{case}_wall_offset_audit.npz',analytic_points=actual,solid_outward_normals=normals,delta=np.array(delta),u_half_offset=halved,u_primary_offset=wu,u_double_offset=shifted)
    masks=distance_masks(data['xy_s'],domain);ref=dict(xy_ip=data['xy_s'],area_weight=w,u_ip=uip,s_ip=data['s'],xy_node=data['xy_u'],u_node=data['u'],wall=wall,wall_weight=ww,wall_tags=tags,wall_u=wu[:-4],extrema=extrema,extrema_u=wu[-4:],**masks)
    path=OUT/f'{case}_reference.npz';np.savez_compressed(path,**ref);preliminary.update(passed=True,path=str(path.resolve()),sha256=sha(path),wall_offset_sensitivity_max=sensitivity,half_offset_sensitivity_max=half_sensitivity,solid_side_offset=delta,normalization=dict(length=L,displacement=U,stress=S));write(OUT/f'{case}_reference_checks.json',preliminary)
    print('REFERENCE_PASS',case,'stress reconstruction %',reconstruction,'wall shift',sensitivity,flush=True);return ref,preliminary

def main():
    repair=read(ROOT/'protocols/R2_P2J_reference_repair.json');assert sha(B/'suspension.json')==repair['training_suspension_sha256'];assert sha(OUT.parent/'diagnosis.json')==repair['diagnosis_sha256'];assert sha(OUT.parent/'boundary_geometry.json')==repair['boundary_geometry_sha256']
    snapshot=read(B/'suspension.json')
    for f,h in snapshot['files_sha256'].items():assert sha(B/f)==h
    for f,h in snapshot['supporting_files_sha256'].items():assert sha(f)==h
    p=read(P);m=read(B/'manifest.json');assert m['status']=='fit_complete' and len(m['completed'])==12
    assert sha(P)==m['protocol_sha256'];pre=read(B/'preflight.json');assert pre['passed'] and sha(B/'preflight.json')==m['preflight_sha256']
    for k,h in m['source_sha256'].items():assert sha(ROOT/'code'/k)==h
    for k,h in m['prepared_sha256'].items():assert sha(B/k)==h
    for k,h in m['inputs_sha256'].items():assert sha(k)==h
    for k,h in pre['reference_file_sha256'].items():assert sha(k)==h
    for run,files in m['terminal_sha256'].items():
        for k,h in files.items():assert sha(B/run/k)==h
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=['evaluate_p2j.py','transfer_evaluation_helpers.py','transfer_mechanics.py','evaluate_p2.py','evaluate_p2d_engineering.py','diagnose_p2g.py','p2_fem_interpolation.py','shared_geometry_features.py']
    a=dict(repair_protocol_sha256=sha(ROOT/'protocols/R2_P2J_reference_repair.json'),diagnosis_sha256=sha(OUT.parent/'diagnosis.json'),geometry_check_sha256=sha(OUT.parent/'boundary_geometry.json'),protocol_sha256=sha(P),manifest_sha256=sha(B/'manifest.json'),source_sha256={f:sha(ROOT/'code'/f) for f in sources},all_12_fits_frozen_before_FEM=True,formal_timing=False,references={},results={},spatial={},points_sha256={});completed=0;geometries=read(ROOT/'inputs/cavity_geometries.json')
    try:
        # Complete both reference checks before evaluating any method ranking.
        for case,setting in p['cases'].items():
            domain,_,_=normalized_domain(geometries[case]);ref,a['references'][case]=reference(case,domain,setting);del ref
        for case,setting in p['cases'].items():
            ref=load(a['references'][case]['path']);domain,_,_=normalized_domain(geometries[case]);z=load(ROOT/f'results/R2_P2D_shared_features/{case}_covers.npz')
            valid=[domain.random_interior(24000,np.random.default_rng(s)) for s in [9296101,9296102]];masks=[distance_masks(x,domain) for x in valid];bp,bw=independent_boundary(domain,setting)
            arrays={f'validation{i}':x for i,x in enumerate(valid)}
            for i,mask in enumerate(masks):arrays.update({f'validation{i}_{k}':v for k,v in mask.items()})
            arrays.update({'boundary_'+k:v for k,v in bp.items()});arrays.update({'weight_'+k:v for k,v in bw.items()});pointspath=OUT/f'{case}_physics_points.npz';np.savez_compressed(pointspath,**arrays);a['points_sha256'][pointspath.name]=sha(pointspath)
            for step in p['evaluation_steps']:
                errors={}
                for method in p['methods']:
                    for arm in p['arms']:
                        run=f'{case}_seed{p["seed"]}_{method}_{arm}';tag=f'{run}_step{step:04d}';checkpoint=B/run/f'step_{step:04d}.pt'
                        write(OUT/'progress.json',dict(status='running',completed_fields=completed,expected_fields=24,active=tag))
                        train=load(B/f'{case}_block{step//400-1 if arm=="uniform_refresh" else 0}_points.npz');n=int(train['uniform_count'])
                        model=make_shared_model(method,p['seed'],z['centres'],z['halfwidths'],(z['uniform_centres'],z['uniform_halfwidths'])).float().cuda();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));model.eval()
                        rt=domain_residual(model,train['domain'],setting);rv=[domain_residual(model,x,setting) for x in valid]
                        tb,tr=boundary(model,train,setting,{'hole':train['hole_weight']});vb,vr=boundary(model,bp,setting,bw)
                        trainobj=stats(rt,train['domain_weight'])['total_mean']+10*tb['traction']+100*tb['displacement']
                        work=next(v for v in read(B/run/'trace.json') if v['cumulative_step']==step and v['block_step']==400);assert abs(trainobj-work['loss'])<2e-5*max(.01,abs(work['loss'])),(tag,trainobj,work['loss'])
                        with torch.no_grad():fast=float(objective(parts(model,prepare(model,train),setting)))
                        assert abs(fast-trainobj)<2e-5*max(.01,abs(fast))
                        q=predict(model,ref['xy_ip']);un=predict(model,ref['xy_node'])[:,:2];wall=predict(model,np.vstack([ref['wall'],ref['extrema']]))[:,:2]
                        pred=dict(q_ip=q,u_node=un,u_wall=wall[:-4],u_extrema=wall[-4:]);predpath=OUT/f'{tag}_predictions.npz';np.savez_compressed(predpath,**pred);values,err=score(pred,ref)
                        for label in np.unique(ref['wall_tags']):
                            mask=ref['wall_tags']==label;values['wall_u_'+label]=metrics(pred['u_wall'][mask],ref['wall_u'][mask],ref['wall_weight'][mask])
                        ep=OUT/f'{tag}_error_norms.npz';np.savez_compressed(ep,**err);errors[method+'_'+arm]=err
                        audit,aa=numerical_audit(model,valid[0],rv[0],setting['E'],setting['nu']);raw=dict(active_training=rt,validation0=rv[0],validation1=rv[1]);raw.update({'training_boundary_'+k:v for k,v in tr.items()});raw.update({'independent_boundary_'+k:v for k,v in vr.items()});raw.update({'audit_'+k:v for k,v in aa.items()});rp=OUT/f'{tag}_residuals.npz';np.savez_compressed(rp,**raw)
                        row=dict(case=case,seed=p['seed'],step=step,method=method,arm=arm,checkpoint_path=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),cumulative_closures=work['closure_evaluations'],training_objective_recomputed=trainobj,training_objective_recorded=work['loss'],training_uniform=stats(rt[:n]),training_probes=stats(rt[n:]),validation=[stats(v) for v in rv],regions=[region_stats(v,mask) for v,mask in zip(rv,masks)],training_boundary=tb,independent_boundary=vb,numerical_audit=audit,metrics=values,predictions_path=str(predpath.resolve()),predictions_sha256=sha(predpath),error_norms_sha256=sha(ep),raw_residuals_sha256=sha(rp))
                        write(OUT/f'{tag}_metrics.json',row);a['results'][tag]=row;completed+=1;print(tag,'U/S/wall',*[values[k]['relative_l2_percent'] for k in ['u_area','s_area','wall_u']],flush=True)
                        del model,pred,q,un,wall,rt,rv,raw;torch.cuda.empty_cache()
                w=ref['area_weight'];ww=ref['wall_weight'];spatial={name:{other:dict(u=float(w@(e['u']<oe['u'])/w.sum()),s=float(w@(e['s']<oe['s'])/w.sum()),wall_u=float(ww@(e['wall_u']<oe['wall_u'])/ww.sum())) for other,oe in errors.items() if other!=name} for name,e in errors.items()}
                a['spatial'][f'{case}_step{step:04d}']=spatial;write(OUT/f'{case}_step{step:04d}_spatial.json',spatial);del errors
            del ref
        assert completed==24;a['finished_utc']=datetime.now(timezone.utc).isoformat();write(OUT/'analysis.json',a);write(OUT/'progress.json',dict(status='complete',completed_fields=24,expected_fields=24,active=None));print('ALL_24_TRANSFER_ENDPOINTS_COMPLETE',flush=True)
    except BaseException as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed_fields=completed));write(OUT/'progress.json',dict(status='failed',completed_fields=completed,expected_fields=24));raise
if __name__=='__main__':main()
