"""Frozen-field physical residual generalization audit; no FEM or optimization."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from p2_components import make_model,residual
from cavity_cover import normalized_domain

ROOT=Path(__file__).resolve().parents[1];BATCH=ROOT/'results/R2_P2A';OUT=BATCH/'diagnostic'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')


def near_wall(xy,domain):
    h=domain.holes[0]
    if h['kind']=='ellipse':return abs(np.linalg.norm(xy,axis=1)-h['axes'][0])<=.05
    v=np.asarray(h['vertices']);d=np.full(len(xy),np.inf)
    for a,b in zip(v,np.roll(v,-1,axis=0)):
        t=np.clip((xy-a)@(b-a)/np.sum((b-a)**2),0,1)
        d=np.minimum(d,np.linalg.norm(xy-a-t[:,None]*(b-a),axis=1))
    return d<=.05


def stats(r):
    norm=np.linalg.norm(r,axis=1)
    return dict(equilibrium_mean=float(np.mean(np.sum(r[:,:2]**2,axis=1))),
                constitutive_mean=float(np.mean(np.sum(r[:,2:]**2,axis=1))),
                total_mean=float(np.mean(norm**2)),norm_quantiles=np.quantile(norm,[.5,.9,.95,.99,1]).tolist())


@torch.no_grad()
def main():
    m=read(BATCH/'manifest.json');assert m['status']=='fit_complete' and len(m['completed'])==15
    assert not OUT.exists();OUT.mkdir()
    for name,files in m['terminal_sha256'].items():
        for f,h in files.items():assert sha(BATCH/name/f)==h
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=read(ROOT/'protocols/R2_P2A_development.json');geometries=read(ROOT/'inputs/cavity_geometries.json')
    report=dict(protocol_sha256=sha(ROOT/'protocols/R2_P2A_diagnostic.json'),source_sha256=sha(__file__),
                terminal_manifest_sha256=sha(BATCH/'manifest.json'),fem_read=False,optimization_runs=0,results={})
    for case,setting in p['cases'].items():
        geometry=setting['geometry'];domain,_,_=normalized_domain(geometries[geometry])
        with np.load(BATCH/f'{geometry}_points.npz') as z:training=z['domain'];n_uniform=int(z['uniform_count'])
        with np.load(BATCH/f'{geometry}_covers.npz') as z:covers=[(z['centres'],z['halfwidths']),(z['uniform_centres'],z['uniform_halfwidths'])]
        independent=domain.random_interior(24000,np.random.default_rng(9262026))
        near=near_wall(independent,domain);xy=np.vstack([training,independent])
        if not (OUT/f'{geometry}_points.npz').exists():np.savez_compressed(OUT/f'{geometry}_points.npz',training=training,independent=independent,near_wall=near)
        for method in p['methods']:
            name=case+'_'+method;model=make_model(method,p['seed'],covers,'cuda').float()
            model.load_state_dict(torch.load(BATCH/name/'terminal.pt',map_location='cuda',weights_only=True))
            r=np.empty((len(xy),5),dtype=np.float64)
            for start in range(0,len(xy),1024):
                sl=slice(start,start+1024);q,j=model.sparse(model.prepare(xy[sl]));r[sl]=residual(q,j).cpu().numpy()
            a,b=r[:len(training)],r[len(training):]
            values=dict(training_uniform=stats(a[:n_uniform]),training_probes=stats(a[n_uniform:]),independent=stats(b),independent_near=stats(b[near]),independent_far=stats(b[~near]))
            values['independent_to_training_uniform_total_ratio']=values['independent']['total_mean']/values['training_uniform']['total_mean']
            values['training_weighted_total']=.9*values['training_uniform']['total_mean']+.1*values['training_probes']['total_mean']
            original=read(BATCH/name/'result.json')['loss_parts'];expected=original['equilibrium']+original['constitutive']
            assert abs(values['training_weighted_total']-expected)<2e-5*max(1,expected)
            np.savez_compressed(OUT/f'{name}_residuals.npz',training=a,independent=b)
            report['results'][name]=values
            print(name,'independent/train residual ratio',values['independent_to_training_uniform_total_ratio'],flush=True)
            del model;torch.cuda.empty_cache()
    write(OUT/'analysis.json',report)
    print('DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__':main()
