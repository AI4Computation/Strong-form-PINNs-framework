"""Audit every new field, pooled contrasts and frozen ancestry before closure."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2K_transfer_replication';J=ROOT/'results/R2_P2J_reference_repair'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    assert not (B/'completion.json').exists();p=read(ROOT/'protocols/R2_P2K_transfer_replication.json');s=read(B/'summary.json');m=read(B/'manifest.json');assert m['status']=='fit_complete' and read(B/'evaluation_progress.json')['completed_fields']==48 and len(s['results'])==72
    assert sha(ROOT/'code/train_p2k.py')==m['driver_sha256'];assert sha(ROOT/'code/report_p2k.py')==s['report_source_sha256'];assert sha(ROOT/'protocols/R2_P2K_transfer_replication.json')==m['protocol_sha256']==s['protocol_sha256']
    for f,h in s['source_analysis_sha256'].items():assert sha(f)==h
    checked=0;pairs=0;maxerror=0.
    for seed in p['seeds']:
        train=B/f'seed{seed}';E=train/'evaluation';sm=read(train/'manifest.json');a=read(E/'analysis.json');assert len(sm['completed'])==12 and len(a['results'])==24 and sha(train/'manifest.json')==m['seed_manifest_sha256'][str(seed)]==a['manifest_sha256']
        assert sha(ROOT/f'protocols/R2_P2K_seed{seed}.json')==sm['protocol_sha256']
        for f,h in sm['source_sha256'].items():assert sha(ROOT/'code'/f)==h
        for f,h in a['source_sha256'].items():assert sha(ROOT/'code'/f)==h
        for f,h in sm['prepared_sha256'].items():assert sha(train/f)==h
        for f,h in sm['inputs_sha256'].items():assert sha(f)==h
        for f,h in a['points_sha256'].items():assert sha(f)==h
        for run,files in sm['terminal_sha256'].items():
            for f,h in files.items():assert sha(train/run/f)==h
            rr=read(train/run/'result.json');assert rr['accepted_steps']==1600 and rr['parameters']==110705 and not rr['fem_read'];assert all(v['accepted_steps']==400 and v['stop_reason']=='max_iter' for v in rr['blocks'])
        for case in p['cases']:
            base=load(train/f'{case}_block0_points.npz');n=int(base['uniform_count'])
            for block in [1,2,3]:
                fresh=load(train/f'{case}_block{block}_points.npz');assert not np.array_equal(fresh['domain'][:n],base['domain'][:n]);assert np.array_equal(fresh['domain'][n:],base['domain'][n:]);assert all(np.array_equal(v,base[k]) for k,v in fresh.items() if k!='domain')
            for method in p['methods']:
                states=[torch.load(train/f'{case}_seed{seed}_{method}_{arm}/step_0400.pt',map_location='cpu',weights_only=True) for arm in p['arms']];assert all(torch.equal(v,states[1][k]) for k,v in states[0].items());pairs+=1
            ref=load(a['references'][case]['path']);assert sha(a['references'][case]['path'])==a['references'][case]['sha256'];pts=load(J/f'evaluation/{case}_physics_points.npz')
            for step in p['evaluation_steps']:
                errors={}
                for method in p['methods']:
                 for arm in p['arms']:
                    tag=f'{case}_seed{seed}_{method}_{arm}_step{step:04d}';r=a['results'][tag];assert s['results'][tag]['metrics']==r['metrics'];assert sha(r['predictions_path'])==r['predictions_sha256'];assert sha(E/f'{tag}_error_norms.npz')==r['error_norms_sha256'];assert sha(E/f'{tag}_residuals.npz')==r['raw_residuals_sha256']
                    pred=load(r['predictions_path']);err=load(E/f'{tag}_error_norms.npz');raw=load(E/f'{tag}_residuals.npz');errors[method+'_'+arm]=err
                    for i in range(2):
                        v=raw[f'validation{i}'];assert v.shape==(24000,5) and np.isfinite(v).all();np.testing.assert_allclose((v*v).mean(0),r['validation'][i]['component_mean'],rtol=1e-10,atol=1e-12)
                        for region,rs in r['regions'][i].items():
                            mask=pts[f'validation{i}_{region}'];np.testing.assert_allclose((v[mask]*v[mask]).sum(1).mean(),rs['total_mean'],rtol=1e-10,atol=1e-12)
                    for name,est,true,w,key in [('u',pred['q_ip'][:,:2],ref['u_ip'],ref['area_weight'],'u_area'),('s',pred['q_ip'][:,2:],ref['s_ip'],ref['area_weight'],'s_area'),('wall_u',pred['u_wall'],ref['wall_u'],ref['wall_weight'],'wall_u')]:
                        delta=est-true;norm=np.sqrt((delta*delta).sum(1));np.testing.assert_allclose(norm,err[name],rtol=1e-12,atol=1e-14);value=100*np.sqrt(np.sum(w*(delta*delta).sum(1))/np.sum(w*(true*true).sum(1)));stored=r['metrics'][key]['relative_l2_percent'];np.testing.assert_allclose(value,stored,rtol=1e-12,atol=1e-12);maxerror=max(maxerror,abs(value-stored))
                        np.testing.assert_allclose(w@norm/w.sum(),r['metrics'][key]['vector_mae'],rtol=1e-12,atol=1e-14)
                    conv=lambda u:np.array([u[0,1]-u[1,1],u[2,0]-u[3,0]])
                    np.testing.assert_allclose(abs(conv(pred['u_extrema'])-conv(ref['extrema_u'])),r['metrics']['engineering']['convergence_absolute_error'],rtol=1e-12,atol=1e-14);assert r['numerical_audit']['passed'];checked+=1;del pred,raw
                for arm in p['arms']:
                 for control in ['fourier_half','uniform_rbf_fourier']:
                  for name,w in [('u',ref['area_weight']),('s',ref['area_weight']),('wall_u',ref['wall_weight'])]:
                    g=errors['geometry_rbf_fourier_'+arm][name];f=errors[control+'_'+arm][name];fraction=float(w@(g<f)/w.sum());np.testing.assert_allclose(fraction,a['spatial'][f'{case}_step{step:04d}']['geometry_rbf_fourier_'+arm][control+'_'+arm][name],rtol=1e-12,atol=1e-14)
                del errors;print('AUDIT_FIELDS',checked,flush=True)
            del ref,pts
    assert checked==48 and pairs==12
    for group,metrics in s['contrasts'].items():
        for metric,r in metrics.items():
            v=np.array(r['ratios']);np.testing.assert_allclose(np.prod(v)**(1/3),r['geometric_mean'],rtol=1e-12,atol=1e-12);assert int((v<1).sum())==r['favorable_count'] and int((v[1:]<1).sum())==r['prospective_favorable_count'] and bool((v[1:]<1).all())==r['both_new_seeds_favorable']
    old={}
    for folder in sorted((ROOT/'results').iterdir()):
        cp=folder/'completion.json'
        if folder==B or not cp.exists():continue
        c=read(cp);files=c.get('files_sha256',c.get('batch_files_sha256',{}))
        for f,h in files.items():assert sha(folder/f)==h,(folder,f)
        for f,h in c.get('supporting_files_sha256',{}).items():assert sha(f)==h,f
        old[folder.name]=dict(verified_files=len(files),completion_sha256=sha(cp),unchanged=True)
    stopped=ROOT/'results/R2_P2I_shape_transfer';snap=read(stopped/'suspension.json')
    for f,h in snap['files_sha256'].items():assert sha(stopped/f)==h
    for f,h in snap['supporting_files_sha256'].items():assert sha(f)==h
    first=read(ROOT.parent/'00_基线与规则/第一轮保全核验.json');assert first['passed'] and first['files_checked']==9923
    audit=dict(new_fields_checked=48,prior_verified_fields_reused=24,pooled_endpoint_count=72,exact_first_block_pairs=12,raw_error_physics_spatial_engineering_checks=True,paired_summary_arithmetic_checks=True,maximum_primary_L2_difference=maxerror);write(B/'audit.json',audit)
    supports=[ROOT/'P2K_Transfer_Replication_Report.md']+[ROOT/'protocols'/f for f in ['R2_P2K_transfer_replication.json','R2_P2K_seed260931.json','R2_P2K_seed260932.json']]+[ROOT/'code'/f for f in ['train_p2k.py','evaluate_p2k.py','report_p2k.py','audit_close_p2k.py']]
    completion=dict(completed_utc=datetime.now(timezone.utc).isoformat(),training_trajectories=24,new_evaluated_fields=48,descriptive_pooled_fields=72,formal_timing=False,files_sha256={f.relative_to(B).as_posix():sha(f) for f in sorted(B.rglob('*')) if f.is_file()},supporting_files_sha256={str(f.resolve()):sha(f) for f in supports},prior_batches_preserved=old,original_P2I_suspension_unchanged=True,first_round_preservation=first,audit=audit);write(B/'completion.json',completion);print('P2K_CLOSED',len(completion['files_sha256']),'files',flush=True)
if __name__=='__main__':main()
