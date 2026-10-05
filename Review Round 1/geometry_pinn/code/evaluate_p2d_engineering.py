"""Separate, author-authorized post-hoc FEM evaluation of all P2D terminals.

Closed P2D and P2A directories are read-only inputs. No optimizer or selection.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
import torch
from shared_geometry_features import make_shared_model
from evaluate_p2 import metrics,predict
from cavity_cover import normalized_domain

ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT/'results/R2_P2D_shared_features'
BASE=ROOT/'results/R2_P2A';OUT=ROOT/'results/R2_P2D_engineering_exploratory'
PROTOCOL=ROOT/'protocols/R2_P2D_engineering_exploratory.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}

def tail_mass(error,w,fraction=.01):
    order=np.argsort(-error,kind='stable');sw=w[order];se=error[order]
    prior=np.cumsum(sw)-sw;use=np.minimum(sw,np.maximum(0.,fraction*w.sum()-prior))
    return float(np.sum(use*se**2)/max(np.sum(w*error**2),1e-30))

def score(q,ref):
    assert all(np.isfinite(v).all() for v in q.values())
    w=ref['area_weight'];values={};errors={}
    for name,cols,truth in [('u',[0,1],ref['u_ip']),('s',[2,3,4],ref['s_ip'])]:
        pred=q['q_ip'][:,cols];e=np.linalg.norm(pred-truth,axis=1);errors[name]=e
        values[name+'_area']=metrics(pred,truth,w)
        values[name+'_point']=metrics(pred,truth,np.ones(len(w)))
        values[name+'_area']['worst_one_percent_area_squared_error_share']=tail_mass(e,w)
        for region in ['near_wall','far_wall','corner','away_corner']:
            mask=ref[region]
            if mask.any():values[name+'_'+region]=metrics(pred[mask],truth[mask],w[mask])
    values['u_original_nodes']=metrics(q['u_node'],ref['u_node'],np.ones(len(ref['u_node'])))
    values['wall_u']=metrics(q['u_wall'],ref['wall_u'],ref['wall_weight'])
    convergence=lambda u:np.array([u[0,1]-u[1,1],u[2,0]-u[3,0]])
    actual=q['u_extrema'];truth=ref['extrema_u'];ac=convergence(actual);tc=convergence(truth)
    values['engineering']=dict(extrema_order=['crown','invert','right','left'],extrema_prediction=actual.tolist(),
        extrema_reference=truth.tolist(),extrema_absolute_error=abs(actual-truth).tolist(),
        extrema_vector_error=np.linalg.norm(actual-truth,axis=1).tolist(),
        convergence_order=['vertical','horizontal'],convergence_prediction=ac.tolist(),
        convergence_reference=tc.tolist(),convergence_absolute_error=abs(ac-tc).tolist())
    errors['wall_u']=np.linalg.norm(q['u_wall']-ref['wall_u'],axis=1)
    return values,errors

def main():
    assert not OUT.exists(),'Inspect existing evaluation instead of overwriting'
    p=read(PROTOCOL);tp=read(ROOT/'protocols/R2_P2D_shared_features.json');bp=read(ROOT/'protocols/R2_P2A_development.json')
    m=read(TRAIN/'manifest.json');closed=read(TRAIN/'completion.json');bc=read(BASE/'completion.json')
    assert m['status']=='fit_complete' and len(m['completed'])==21
    assert p['methods']==tp['methods'] and p['cases']==list(tp['cases'])
    assert tp['cases']==bp['cases'] and tp['seed']==bp['seed']
    for name,h in closed['files_sha256'].items():assert sha(TRAIN/name)==h
    for name,h in closed['supporting_files_sha256'].items():assert sha(name)==h
    for name,h in m['source_sha256'].items():assert sha(ROOT/'code'/name)==h
    assert sha(ROOT/'protocols/R2_P2D_shared_features.json')==m['protocol_sha256']
    base_analysis=BASE/'evaluation/analysis.json';assert sha(base_analysis)==bc['batch_files_sha256']['evaluation/analysis.json']
    ba=read(base_analysis)
    assert sha(ROOT/'code/evaluate_p2.py')==ba['evaluator_sha256']
    assert sha(ROOT/'code/p2_fem_interpolation.py')==ba['interpolation_sha256']
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    OUT.mkdir()
    report=dict(id=p['id'],type='post_hoc_exploratory',protocol_sha256=sha(PROTOCOL),
        source_sha256={name:sha(ROOT/'code'/name) for name in ['evaluate_p2d_engineering.py','evaluate_p2.py','shared_geometry_features.py','p2_components.py']},
        train_manifest_sha256=sha(TRAIN/'manifest.json'),train_completion_sha256=sha(TRAIN/'completion.json'),
        baseline_completion_sha256=sha(BASE/'completion.json'),original_P2D_gate_unchanged=True,
        training_runs=0,formal_timing=False,all_terminals_previously_frozen=True,references={},cases={},plain_reproduction={})
    geometries=read(ROOT/'inputs/cavity_geometries.json')
    write(OUT/'progress.json',dict(status='running',completed=[],active=None))
    for case in p['cases']:
        geometry=tp['cases'][case]['geometry'];domain,_,_=normalized_domain(geometries[geometry])
        path=BASE/f'evaluation/{case}_reference.npz';rh=sha(path)
        assert rh==bc['batch_files_sha256'][f'evaluation/{case}_reference.npz']==ba['references'][case]['derived_sha256']
        provenance=ba['references'][case]
        assert provenance['stress_reconstruction_percent']<.02 and provenance['wall_offset_sensitivity_max']<1e-5
        for source,h in provenance['source_sha256'].items():assert sha(source)==h
        ref=load(path);w=ref['area_weight'];assert (w>0).all() and abs(w.sum()-domain.area)<1e-5
        report['references'][case]=dict(path=str(path.resolve()),sha256=rh,provenance=provenance)
        covers=load(TRAIN/f'{geometry}_covers.npz');c,h=covers['centres'],covers['halfwidths'];uniform=(covers['uniform_centres'],covers['uniform_halfwidths'])
        rows={};errs={}
        for method in p['methods']:
            name=case+'_'+method;write(OUT/'progress.json',dict(status='running',completed=list(report['cases']),active=name))
            terminal=TRAIN/name/'terminal.pt';assert sha(terminal)==m['terminal_sha256'][name]['terminal.pt']
            state=torch.load(terminal,map_location='cpu',weights_only=True)
            start=time.perf_counter();reused=method in ['fourier_half','anchored','independent_marginal']
            if reused:
                original=torch.load(BASE/name/'terminal.pt',map_location='cpu',weights_only=True)
                assert set(original)==set(state) and all(torch.equal(original[k],state[k]) for k in state)
                predpath=BASE/f'evaluation/{name}_predictions.npz';ph=sha(predpath)
                assert ph==bc['batch_files_sha256'][f'evaluation/{name}_predictions.npz']==ba['cases'][case][method]['predictions_sha256']
                prediction=load(predpath);report['plain_reproduction'][name]=True
            else:
                model=make_shared_model(method,tp['seed'],c,h,uniform).float().cuda();model.load_state_dict(state);model.eval()
                q=predict(model,ref['xy_ip']);un=predict(model,ref['xy_node'])[:,:2]
                boundary=predict(model,np.vstack([ref['wall'],ref['extrema']]))[:,:2]
                prediction=dict(q_ip=q,u_node=un,u_wall=boundary[:-4],u_extrema=boundary[-4:])
                predpath=OUT/f'{name}_predictions.npz';np.savez_compressed(predpath,**prediction);ph=sha(predpath)
                del model;torch.cuda.empty_cache()
            values,error=score(prediction,ref)
            if reused:
                for key in ['u_area','s_area','wall_u']:
                    assert values[key]['relative_l2_percent']==ba['cases'][case][method]['metrics'][key]['relative_l2_percent']
            errorspath=OUT/f'{name}_error_norms.npz';np.savez_compressed(errorspath,**error)
            rows[method]=dict(metrics=values,predictions_path=str(predpath.resolve()),predictions_sha256=ph,
                reused_frozen_prediction=reused,error_norms_sha256=sha(errorspath),
                training_record_path=str((TRAIN/name/'result.json').resolve()),
                development_evaluation_seconds=time.perf_counter()-start)
            errs[method]=error
            print(case,method,'U/S/wall %',*[values[k]['relative_l2_percent'] for k in ['u_area','s_area','wall_u']],flush=True)
            del prediction,state
        for method in p['methods']:
            rows[method]['area_fraction_lower_vector_error']={control:{field:float(np.sum(w*(errs[method][field]<errs[control][field]))/w.sum()) for field in ['u','s']} for control in p['methods'] if control!=method}
            rows[method]['wall_fraction_lower_displacement_error']={control:float(np.sum(ref['wall_weight']*(errs[method]['wall_u']<errs[control]['wall_u']))/ref['wall_weight'].sum()) for control in p['methods'] if control!=method}
        report['cases'][case]=rows;write(OUT/f'{case}_metrics.json',rows)
        del ref,errs
    report['contrasts']={case:{control:{key:rows['geometry_rbf_fourier']['metrics'][key]['relative_l2_percent']/rows[control]['metrics'][key]['relative_l2_percent'] for key in ['u_area','s_area','wall_u']} for control in ['uniform_rbf_fourier','fourier_half']} for case,rows in report['cases'].items()}
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    write(OUT/'analysis.json',report);write(OUT/'progress.json',dict(status='complete',completed=p['cases'],active=None))
    print(json.dumps(report['contrasts']),flush=True)

if __name__=='__main__':main()
