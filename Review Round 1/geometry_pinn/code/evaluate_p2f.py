"""Evaluate every registered seed/budget endpoint only after all fits freeze."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import torch
from shared_geometry_features import make_shared_model
from evaluate_p2 import predict
from evaluate_p2d_engineering import score
from cavity_cover import normalized_domain
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2F_repetition_budget';OUT=B/'evaluation';BASE=ROOT/'results/R2_P2A'
P=ROOT/'protocols/R2_P2F_repetition_budget.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}

def main():
    p=read(P);m=read(B/'manifest.json');assert m['status']=='fit_complete' and len(m['completed'])==27
    assert sha(P)==m['protocol_sha256']
    for name,files in m['terminal_sha256'].items():
        for f,h in files.items():assert sha(B/name/f)==h
    for f,h in m['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in m['prepared_sha256'].items():assert sha(B/f)==h
    # Verify the metric implementation used in the earlier closed engineering evaluation.
    ec=read(ROOT/'results/R2_P2D_engineering_exploratory/completion.json')
    for file,h in ec['supporting_files_sha256'].items():assert sha(file)==h
    bc=read(BASE/'completion.json');ba=read(BASE/'evaluation/analysis.json')
    assert sha(BASE/'evaluation/analysis.json')==bc['batch_files_sha256']['evaluation/analysis.json']
    assert sha(ROOT/'code/evaluate_p2.py')==ba['evaluator_sha256']
    assert not OUT.exists();OUT.mkdir()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(protocol_sha256=sha(P),training_manifest_sha256=sha(B/'manifest.json'),
        source_sha256={f:sha(ROOT/'code'/f) for f in ['evaluate_p2f.py','evaluate_p2d_engineering.py','evaluate_p2.py','shared_geometry_features.py']},
        all_27_trajectories_frozen_before_FEM=True,training_runs=0,formal_timing=False,references={},cases={})
    geom=read(ROOT/'inputs/cavity_geometries.json');complete=0
    write(OUT/'progress.json',dict(status='running',completed_fields=0,expected_fields=81,active=None))
    for case,setting in p['case_settings'].items():
        geometry=setting['geometry'];domain,_,_=normalized_domain(geom[geometry])
        path=BASE/f'evaluation/{case}_reference.npz';rh=sha(path)
        assert rh==ba['references'][case]['derived_sha256']==bc['batch_files_sha256'][f'evaluation/{case}_reference.npz']
        provenance=ba['references'][case]
        for f,h in provenance['source_sha256'].items():assert sha(f)==h
        assert provenance['stress_reconstruction_percent']<.02 and provenance['wall_offset_sensitivity_max']<1e-5
        ref=load(path);w=ref['area_weight'];assert (w>0).all() and abs(w.sum()-domain.area)<1e-5
        report['references'][case]=dict(path=str(path.resolve()),sha256=rh,provenance=provenance)
        z=load(B/f'{geometry}_covers.npz');c,h,uc,uh=z['centres'],z['halfwidths'],z['uniform_centres'],z['uniform_halfwidths']
        report['cases'][case]={}
        for seed in p['seeds']:
            report['cases'][case][str(seed)]={}
            for step in p['evaluation_steps']:
                rows={};errors={}
                for method in p['methods']:
                    name=f'{case}_seed{seed}_{method}';tag=name+f'_step{step:04d}'
                    result=read(B/name/'result.json');checkpoint=B/name/f'step_{step:04d}.pt';actual_step=step;state_status='registered_checkpoint'
                    if not checkpoint.exists():
                        assert result['accepted_steps']<step and result['stop_reason'] in ['gradient_tolerance','step_tolerance','loss_change_tolerance']
                        checkpoint=B/name/'terminal.pt';actual_step=result['accepted_steps'];state_status='unchanged_early_converged_terminal'
                    assert sha(checkpoint)==m['terminal_sha256'][name][checkpoint.name]
                    trace=read(B/name/'trace.json');work=next(v for v in trace if v['accepted_step']==actual_step)
                    write(OUT/'progress.json',dict(status='running',completed_fields=complete,expected_fields=81,active=tag))
                    model=make_shared_model(method,seed,c,h,(uc,uh)).float().cuda()
                    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));model.eval()
                    q=predict(model,ref['xy_ip']);un=predict(model,ref['xy_node'])[:,:2]
                    wall=predict(model,np.vstack([ref['wall'],ref['extrema']]))[:,:2]
                    pred=dict(q_ip=q,u_node=un,u_wall=wall[:-4],u_extrema=wall[-4:])
                    predpath=OUT/f'{tag}_predictions.npz';np.savez_compressed(predpath,**pred)
                    values,error=score(pred,ref);errpath=OUT/f'{tag}_error_norms.npz';np.savez_compressed(errpath,**error)
                    rows[method]=dict(metrics=values,predictions_path=str(predpath.resolve()),predictions_sha256=sha(predpath),
                        error_norms_sha256=sha(errpath),checkpoint_path=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),
                        nominal_step=step,actual_step=actual_step,endpoint_status=state_status,closure_evaluations=work['closure_evaluations'],
                        training_record_path=str((B/name/'result.json').resolve()))
                    errors[method]=error;complete+=1
                    print(tag,'U/S/wall %',*[values[k]['relative_l2_percent'] for k in ['u_area','s_area','wall_u']],flush=True)
                    del model,pred,q,un,wall;torch.cuda.empty_cache()
                for method in p['methods']:
                    rows[method]['area_fraction_lower_vector_error']={control:{field:float(np.sum(w*(errors[method][field]<errors[control][field]))/w.sum()) for field in ['u','s']} for control in p['methods'] if control!=method}
                    rows[method]['wall_fraction_lower_displacement_error']={control:float(np.sum(ref['wall_weight']*(errors[method]['wall_u']<errors[control]['wall_u']))/ref['wall_weight'].sum()) for control in p['methods'] if control!=method}
                report['cases'][case][str(seed)][str(step)]=rows
                write(OUT/f'{case}_seed{seed}_step{step:04d}_metrics.json',rows);del errors
        write(OUT/f'{case}_all_metrics.json',report['cases'][case]);del ref
    assert complete==81
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    write(OUT/'analysis.json',report);write(OUT/'progress.json',dict(status='complete',completed_fields=complete,expected_fields=81,active=None))
    print('ALL_81_REGISTERED_ENDPOINTS_EVALUATED',flush=True)
if __name__=='__main__':main()
