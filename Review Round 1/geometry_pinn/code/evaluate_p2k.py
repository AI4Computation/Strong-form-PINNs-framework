"""Evaluate all new replication states against unchanged verified references."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import traceback
import numpy as np
import torch
from evaluate_p2j import sha,read,write,load,domain_residual,boundary,numerical_audit,stats,region_stats,predict,score,metrics,objective,parts,prepare,make_shared_model,normalized_domain
ROOT=Path(__file__).resolve().parents[1];K=ROOT/'results/R2_P2K_transfer_replication';J=ROOT/'results/R2_P2J_reference_repair';MASTER_PROTOCOL=ROOT/'protocols/R2_P2K_transfer_replication.json'
def evaluate_seed(seed,prior):
    B=K/f'seed{seed}';OUT=B/'evaluation';P=ROOT/f'protocols/R2_P2K_seed{seed}.json'
    p=read(P);m=read(B/'manifest.json');assert m['status']=='fit_complete' and len(m['completed'])==12
    assert sha(P)==m['protocol_sha256'];pre=read(B/'preflight.json');assert pre['passed'] and sha(B/'preflight.json')==m['preflight_sha256']
    for k,h in m['source_sha256'].items():assert sha(ROOT/'code'/k)==h
    for k,h in m['prepared_sha256'].items():assert sha(B/k)==h
    for k,h in m['inputs_sha256'].items():assert sha(k)==h
    for k,h in pre['reference_file_sha256'].items():assert sha(k)==h
    for run,files in m['terminal_sha256'].items():
        for k,h in files.items():assert sha(B/run/k)==h
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=['evaluate_p2k.py','evaluate_p2j.py','transfer_evaluation_helpers.py','transfer_mechanics.py','evaluate_p2.py','evaluate_p2d_engineering.py','diagnose_p2g.py','p2_fem_interpolation.py','shared_geometry_features.py']
    a=dict(parent_protocol_sha256=sha(MASTER_PROTOCOL),prior_completion_sha256=sha(J/'completion.json'),protocol_sha256=sha(P),manifest_sha256=sha(B/'manifest.json'),source_sha256={f:sha(ROOT/'code'/f) for f in sources},all_24_new_fits_frozen_before_FEM=True,formal_timing=False,references=prior['references'],results={},spatial={},points_sha256={});completed=0;geometries=read(ROOT/'inputs/cavity_geometries.json')
    try:
        for case,setting in p['cases'].items():
            ref=load(a['references'][case]['path']);domain,_,_=normalized_domain(geometries[case]);z=load(ROOT/f'results/R2_P2D_shared_features/{case}_covers.npz')
            pointspath=J/f'evaluation/{case}_physics_points.npz';assert sha(pointspath)==prior['points_sha256'][pointspath.name];arrays=load(pointspath);valid=[arrays[f'validation{i}'] for i in range(2)];masks=[{k:arrays[f'validation{i}_{k}'] for k in ['near_wall','far_wall','corner','away_corner']} for i in range(2)];bp={k[len('boundary_'):]:v for k,v in arrays.items() if k.startswith('boundary_')};bw={k[len('weight_'):]:v for k,v in arrays.items() if k.startswith('weight_')};a['points_sha256'][str(pointspath.resolve())]=sha(pointspath)
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

def main():
    p=read(MASTER_PROTOCOL);m=read(K/'manifest.json');assert m['status']=='fit_complete' and m['completed_seeds']==p['seeds'];assert sha(MASTER_PROTOCOL)==m['protocol_sha256'];assert sha(ROOT/'code/train_p2k.py')==m['driver_sha256'];assert sha(ROOT/'code/train_p2i.py')==m['trainer_sha256']
    assert sha(J/'completion.json')==p['P2J_completion_sha256'];c=read(J/'completion.json')
    for f,h in c['files_sha256'].items():assert sha(J/f)==h
    for f,h in c['supporting_files_sha256'].items():assert sha(f)==h
    prior=read(J/'evaluation/analysis.json')
    for ref in prior['references'].values():
        assert sha(ref['path'])==ref['sha256']
        for f,h in ref['source_sha256'].items():assert sha(f)==h
    status=dict(status='running',completed_seeds=[],active_seed=None,analysis_sha256={},expected_fields=48,completed_fields=0)
    write(K/'evaluation_progress.json',status)
    for seed in p['seeds']:
        assert sha(K/f'seed{seed}/manifest.json')==m['seed_manifest_sha256'][str(seed)];status['active_seed']=seed;write(K/'evaluation_progress.json',status);evaluate_seed(seed,prior);status['completed_seeds'].append(seed);status['completed_fields']+=24;status['analysis_sha256'][str(seed)]=sha(K/f'seed{seed}/evaluation/analysis.json');write(K/'evaluation_progress.json',status)
    status.update(status='complete',active_seed=None);write(K/'evaluation_progress.json',status);print('ALL_48_REPLICATION_ENDPOINTS_COMPLETE',flush=True)
if __name__=='__main__':main()
