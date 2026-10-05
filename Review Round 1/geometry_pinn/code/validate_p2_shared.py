"""Frozen-terminal, pre-registered physical screening. No FEM reads."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from p2_components import residual
from cavity_cover import normalized_domain
from shared_geometry_features import make_shared_model

ROOT=Path(__file__).resolve().parents[1]; B=ROOT/'results/R2_P2D_shared_features'; OUT=B/'physical_validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def stats(r):
    e=np.sum(r*r,axis=1);k=max(1,int(np.ceil(len(e)*.01)))
    return dict(total_mean=float(e.mean()),component_mean=np.mean(r*r,axis=0).tolist(),
        norm_quantiles=np.quantile(np.sqrt(e),[.5,.9,.95,.99,1]).tolist(),
        top_one_percent_mass=float(np.sort(e)[-k:].sum()/max(e.sum(),1e-30)))

@torch.no_grad()
def main():
    m=read(B/'manifest.json');assert m['status']=='fit_complete' and len(m['completed'])==21
    for name,files in m['terminal_sha256'].items():
        for f,h in files.items():assert sha(B/name/f)==h
    for name,h in m['source_sha256'].items():assert sha(ROOT/'code'/name)==h
    for name,h in m['prepared_sha256'].items():assert sha(B/name)==h
    protocol=ROOT/'protocols/R2_P2D_shared_features.json';assert sha(protocol)==m['protocol_sha256']
    p=read(protocol);geometries=read(ROOT/'inputs/cavity_geometries.json')
    assert not OUT.exists();OUT.mkdir()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(protocol_sha256=sha(protocol),source_sha256=sha(__file__),manifest_sha256=sha(B/'manifest.json'),
                results={},plain_model_reproduction={},fem_read=False,training_runs=0)
    size=p['validation']['points_per_set']
    for case,setting in p['cases'].items():
        geometry=setting['geometry'];domain,_,_=normalized_domain(geometries[geometry])
        valid=[domain.random_interior(size,np.random.default_rng(seed)) for seed in p['validation']['seeds']]
        if not (OUT/f'{geometry}_points.npz').exists():np.savez_compressed(OUT/f'{geometry}_points.npz',validation0=valid[0],validation1=valid[1])
        with np.load(B/f'{geometry}_covers.npz') as z:c,h,uc,uh=z['centres'],z['halfwidths'],z['uniform_centres'],z['uniform_halfwidths']
        with np.load(B/f'{geometry}_points.npz') as z:train=z['domain'];n=int(z['uniform_count'])
        for method in p['methods']:
            name=case+'_'+method
            state=torch.load(B/name/'terminal.pt',map_location='cpu',weights_only=True)
            if method in ['fourier_half','anchored','independent_marginal']:
                old=torch.load(ROOT/'results/R2_P2A'/name/'terminal.pt',map_location='cpu',weights_only=True)
                equal=set(old)==set(state) and all(torch.equal(old[k],state[k]) for k in old)
                report['plain_model_reproduction'][name]=equal
                assert equal, 'Plain-control terminal no longer exactly reproduces P2A'
            model=make_shared_model(method,p['seed'],c,h,(uc,uh)).float().cuda();model.load_state_dict(state)
            xy=np.vstack([train]+valid);r=np.empty((len(xy),5),dtype=np.float64)
            for start in range(0,len(xy),1024):
                sl=slice(start,start+1024);q,j=model.sparse(model.prepare(xy[sl]));r[sl]=residual(q,j).cpu().numpy()
            assert np.isfinite(r).all()
            tr=r[:len(train)];v0=r[len(train):len(train)+size];v1=r[len(train)+size:]
            result=dict(training_uniform=stats(tr[:n]),training_probes=stats(tr[n:]),validation=[stats(v0),stats(v1)])
            result['independent_training_ratios']=[v['total_mean']/result['training_uniform']['total_mean'] for v in result['validation']]
            expected=read(B/name/'result.json')['loss_parts'];target=expected['equilibrium']+expected['constitutive']
            weighted=.9*result['training_uniform']['total_mean']+.1*result['training_probes']['total_mean']
            assert abs(weighted-target)<2e-5*max(1.,target)
            np.savez_compressed(OUT/f'{name}_residuals.npz',training=tr,validation0=v0,validation1=v1)
            report['results'][name]=result
            print(name,[v['total_mean'] for v in result['validation']],flush=True)
            del model,state;torch.cuda.empty_cache()
    rows=report['results'];cases={}
    for case in p['cases']:
        g=rows[case+'_geometry_rbf_fourier'];u=rows[case+'_uniform_rbf_fourier'];f=rows[case+'_fourier_half']
        ru=[g['validation'][i]['total_mean']/u['validation'][i]['total_mean'] for i in range(2)]
        rf=[g['validation'][i]['total_mean']/f['validation'][i]['total_mean'] for i in range(2)]
        rt=g['independent_training_ratios']
        cases[case]=dict(geometry_over_uniform=ru,geometry_over_fourier=rf,independent_over_training=rt,
                        independent_gain=max(ru)<=.9,competitive=max(rf)<=1.1,reliability=max(rt)<=10)
    report['gates']=dict(cases=cases,advance_to_FEM=all(v['independent_gain'] and v['competitive'] and v['reliability'] for v in cases.values()))
    write(OUT/'analysis.json',report);print(json.dumps(report['gates']),flush=True)

if __name__=='__main__':main()
