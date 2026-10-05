"""All registered frozen P2H/continuous endpoints: common physical and FEM evaluation."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import torch
from shared_geometry_features import make_shared_model
from cavity_cover import normalized_domain
from diagnose_p2g import domain_residual,regions,region_stats,stats,boundary,independent_boundary,numerical_audit
from evaluate_p2 import predict
from evaluate_p2d_engineering import score
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2H_sampling_intervention';OLD=ROOT/'results/R2_P2F_repetition_budget';OUT=B/'evaluation';P=ROOT/'protocols/R2_P2H_sampling_intervention.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def main():
    p=read(P);m=read(B/'manifest.json');assert m['status']=='fit_complete' and len(m['completed'])==18
    assert sha(P)==m['protocol_sha256'];fc=read(OLD/'completion.json')
    for f,h in m['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in m['inputs_sha256'].items():assert sha(f)==h
    for f,h in m['points_sha256'].items():assert sha(B/f)==h
    for name,files in m['terminal_sha256'].items():
        for f,h in files.items():assert sha(B/name/f)==h
    for batch in ['R2_P2G_frozen_physics','R2_P2D_engineering_exploratory']:
        c=read(ROOT/'results'/batch/'completion.json')
        for f,h in c['supporting_files_sha256'].items():assert sha(f)==h
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    domain,_,_=normalized_domain(read(ROOT/'inputs/cavity_geometries.json')['L1'])
    valid=[domain.random_interior(24000,np.random.default_rng(s)) for s in [9285101,9285102]];masks=[regions(x,domain) for x in valid];bp,bw=independent_boundary(domain)
    arrays={f'validation{i}':x for i,x in enumerate(valid)}
    for i,mask in enumerate(masks):arrays.update({f'validation{i}_{k}':v for k,v in mask.items()})
    arrays.update({'boundary_'+k:v for k,v in bp.items()});arrays.update({'weight_'+k:v for k,v in bw.items()});np.savez_compressed(OUT/'L1_physics_points.npz',**arrays)
    prior=read(OLD/'evaluation/analysis.json');assert sha(OLD/'evaluation/analysis.json')==fc['files_sha256']['evaluation/analysis.json']
    refpath=Path(prior['references']['L1']['path']);assert sha(refpath)==prior['references']['L1']['sha256']
    for f,h in prior['references']['L1']['provenance']['source_sha256'].items():assert sha(f)==h
    ref=load(refpath);z=load(OLD/'L1_covers.npz');c,h,uc,uh=z['centres'],z['halfwidths'],z['uniform_centres'],z['uniform_halfwidths']
    sources=['evaluate_p2h.py','diagnose_p2g.py','evaluate_p2.py','evaluate_p2d_engineering.py','shared_geometry_features.py']
    a=dict(protocol_sha256=sha(P),manifest_sha256=sha(B/'manifest.json'),source_sha256={f:sha(ROOT/'code'/f) for f in sources},points_sha256=sha(OUT/'L1_physics_points.npz'),reference=prior['references']['L1'],prior_FEM_analysis_sha256=sha(OLD/'evaluation/analysis.json'),all_18_continuations_frozen_before_FEM=True,formal_timing=False,training_runs=0,results={},spatial={})
    completed=0;newfields=0
    for seed in p['seeds']:
        original=load(OLD/f'L1_seed{seed}_points.npz');n=int(original['uniform_count'])
        for step in p['evaluation_steps']:
            errors={};group={}
            for method in p['methods']:
                for arm in ['continuous']+p['arms']:
                    name=f'L1_seed{seed}_{method}';run=name+('_'+arm if arm!='continuous' else '');tag=run+f'_step{step:04d}'
                    checkpoint=(OLD/name if arm=='continuous' else B/run)/f'step_{step:04d}.pt'
                    expected=fc['files_sha256'][f'{name}/step_{step:04d}.pt'] if arm=='continuous' else m['terminal_sha256'][run][f'step_{step:04d}.pt'];assert sha(checkpoint)==expected
                    write(OUT/'progress.json',dict(status='running',completed_fields=completed,expected_fields=54,new_FEM_fields=newfields,active=tag))
                    active=load(B/f'L1_seed{seed}_refresh_block{step//400-2}_points.npz') if arm=='uniform_refresh' else original
                    model=make_shared_model(method,seed,c,h,(uc,uh)).float().cuda();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));model.eval()
                    ro=domain_residual(model,original['domain']);rt=domain_residual(model,active['domain']) if arm=='uniform_refresh' else ro;rv=[domain_residual(model,x) for x in valid]
                    tb,tr=boundary(model,active,-1.,-5.,{'hole':active['hole_weight']});vb,vr=boundary(model,bp,-1.,-5.,bw)
                    trainobj=stats(rt,active['domain_weight'])['total_mean']+10*tb['traction']+100*tb['displacement']
                    if arm=='continuous':
                        work=next(v for v in read(OLD/name/'trace.json') if v['accepted_step']==step);loss=work['loss'];closures=work['closure_evaluations']
                    else:
                        work=next(v for v in read(B/run/'trace.json') if v['cumulative_step']==step and v['block_step']==400);loss=work['loss'];closures=work['continuation_closures']+read(B/run/'result.json')['reused_prefix_closures']
                    assert abs(trainobj-loss)<2e-5*max(.01,abs(loss)),(tag,trainobj,loss)
                    if arm=='continuous':
                        priorrow=prior['cases']['L1'][str(seed)][str(step)][method];predpath=Path(priorrow['predictions_path']);assert sha(predpath)==priorrow['predictions_sha256'];pred=load(predpath)
                    else:
                        q=predict(model,ref['xy_ip']);un=predict(model,ref['xy_node'])[:,:2];wall=predict(model,np.vstack([ref['wall'],ref['extrema']]))[:,:2]
                        pred=dict(q_ip=q,u_node=un,u_wall=wall[:-4],u_extrema=wall[-4:]);predpath=OUT/f'{tag}_predictions.npz';np.savez_compressed(predpath,**pred);newfields+=1
                    metrics,err=score(pred,ref)
                    if arm=='continuous':
                        assert all(metrics[k]['relative_l2_percent']==priorrow['metrics'][k]['relative_l2_percent'] for k in ['u_area','s_area','wall_u'])
                    errorpath=OUT/f'{tag}_error_norms.npz';np.savez_compressed(errorpath,**err);errors[method+'_'+arm]=err
                    audit,aa=numerical_audit(model,valid[0],rv[0]);raw=dict(original_training=ro,active_training=rt,validation0=rv[0],validation1=rv[1]);raw.update({'training_boundary_'+k:v for k,v in tr.items()});raw.update({'independent_boundary_'+k:v for k,v in vr.items()});raw.update({'audit_'+k:v for k,v in aa.items()});rawpath=OUT/f'{tag}_residuals.npz';np.savez_compressed(rawpath,**raw)
                    row=dict(case='L1',seed=seed,step=step,method=method,arm=arm,checkpoint_path=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),cumulative_steps=step,cumulative_closures=closures,
                        training_objective_recomputed=trainobj,training_objective_recorded=loss,original_training_uniform=stats(ro[:n]),active_training_uniform=stats(rt[:n]),active_training_probes=stats(rt[n:]),
                        validation=[stats(v) for v in rv],regions=[region_stats(v,mask) for v,mask in zip(rv,masks)],training_boundary=tb,independent_boundary=vb,numerical_audit=audit,
                        independent_training_uniform_ratio=[stats(v)['total_mean']/stats(rt[:n])['total_mean'] for v in rv],metrics=metrics,predictions_path=str(predpath.resolve()),predictions_sha256=sha(predpath),reused_FEM_prediction=arm=='continuous',error_norms_sha256=sha(errorpath),raw_residuals_sha256=sha(rawpath))
                    write(OUT/f'{tag}_metrics.json',row);a['results'][tag]=row;group[method+'_'+arm]=tag;completed+=1
                    print(tag,'U/S/wall',*[metrics[k]['relative_l2_percent'] for k in ['u_area','s_area','wall_u']],'R',[v['total_mean'] for v in row['validation']],flush=True)
                    del model,pred,ro,rt,rv,raw;torch.cuda.empty_cache()
            # All paired spatial comparisons remain available, including within-arm and within-method.
            spatial={};w=ref['area_weight'];ww=ref['wall_weight']
            for name,err in errors.items():
                spatial[name]={other:dict(u=float(w@(err['u']<oe['u'])/w.sum()),s=float(w@(err['s']<oe['s'])/w.sum()),wall_u=float(ww@(err['wall_u']<oe['wall_u'])/ww.sum())) for other,oe in errors.items() if other!=name}
            a['spatial'][f'{seed}_step{step:04d}']=spatial;write(OUT/f'seed{seed}_step{step:04d}_spatial.json',spatial);del errors
    assert completed==54 and newfields==36;a['finished_utc']=datetime.now(timezone.utc).isoformat();write(OUT/'analysis.json',a);write(OUT/'progress.json',dict(status='complete',completed_fields=54,expected_fields=54,new_FEM_fields=36,active=None));print('ALL_54_PHYSICAL_AND_36_NEW_FEM_ENDPOINTS_COMPLETE',flush=True)
if __name__=='__main__':main()
