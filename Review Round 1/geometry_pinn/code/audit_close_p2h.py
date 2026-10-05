"""Independent endpoint arithmetic, paired-input checks, and immutable closure."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2H_sampling_intervention';E=B/'evaluation';F=ROOT/'results/R2_P2F_repetition_budget'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def main():
    assert not (B/'completion.json').exists();a=read(E/'analysis.json');s=read(E/'summary.json');m=read(B/'manifest.json');p=read(ROOT/'protocols/R2_P2H_sampling_intervention.json')
    assert m['status']=='fit_complete' and len(m['completed'])==18 and len(a['results'])==54
    assert read(E/'progress.json')['new_FEM_fields']==36
    assert sha(E/'analysis.json')==s['analysis_sha256'] and sha(ROOT/'code/report_p2h.py')==s['report_source_sha256']
    assert sha(B/'manifest.json')==a['manifest_sha256'] and sha(ROOT/'protocols/R2_P2H_sampling_intervention.json')==m['protocol_sha256']
    for f,h in m['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in a['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in m['inputs_sha256'].items():assert sha(f)==h
    for f,h in m['points_sha256'].items():assert sha(B/f)==h
    for name,files in m['terminal_sha256'].items():
        for f,h in files.items():assert sha(B/name/f)==h
        r=read(B/name/'result.json');assert r['continuation_steps']==1200 and r['parameters']==110705 and len(r['blocks'])==3
        assert all(v['accepted_steps']==400 and v['stop_reason']=='max_iter' for v in r['blocks'])
        if r['arm']=='fixed_reset':
            old=next(v['loss'] for v in read(F/f"L1_seed{r['seed']}_{r['method']}/trace.json") if v['accepted_step']==400)
            start=read(B/name/'trace.json')[0]['loss'];assert abs(start-old)<2e-5*max(.01,abs(old))
    for seed in p['seeds']:
        base=load(F/f'L1_seed{seed}_points.npz');n=int(base['uniform_count'])
        for block in range(3):
            fresh=load(B/f'L1_seed{seed}_refresh_block{block}_points.npz')
            assert np.array_equal(fresh['domain'][n:],base['domain'][n:]) and not np.array_equal(fresh['domain'][:n],base['domain'][:n])
            assert all(np.array_equal(v,base[k]) for k,v in fresh.items() if k!='domain')
    ref=load(a['reference']['path']);assert sha(a['reference']['path'])==a['reference']['sha256'];points=load(E/'L1_physics_points.npz');assert sha(E/'L1_physics_points.npz')==a['points_sha256']
    checked=0;max_error=0.
    for tag,r in a['results'].items():
        rawpath=E/f'{tag}_residuals.npz';errpath=E/f'{tag}_error_norms.npz'
        assert sha(rawpath)==r['raw_residuals_sha256'] and sha(errpath)==r['error_norms_sha256'] and sha(r['predictions_path'])==r['predictions_sha256']
        raw=load(rawpath);err=load(errpath);pred=load(r['predictions_path'])
        for i in range(2):
            values=raw[f'validation{i}'];assert values.shape==(24000,5) and np.isfinite(values).all()
            np.testing.assert_allclose(np.mean(values*values,axis=0),r['validation'][i]['component_mean'],rtol=1e-10,atol=1e-12)
            for region,rs in r['regions'][i].items():
                mask=points[f'validation{i}_{region}'];v=(values*values).sum(1)
                np.testing.assert_allclose(v[mask].mean(),rs['total_mean'],rtol=1e-10,atol=1e-12)
        for field,estimate,truth,w,key in [('u',pred['q_ip'][:,:2],ref['u_ip'],ref['area_weight'],'u_area'),('s',pred['q_ip'][:,2:],ref['s_ip'],ref['area_weight'],'s_area'),('wall_u',pred['u_wall'],ref['wall_u'],ref['wall_weight'],'wall_u')]:
            delta=estimate-truth;norm=np.sqrt(np.sum(delta*delta,axis=1));np.testing.assert_allclose(norm,err[field],rtol=1e-12,atol=1e-14)
            value=100*np.sqrt(np.sum(w*np.sum(delta*delta,axis=1))/np.sum(w*np.sum(truth*truth,axis=1)));stored=r['metrics'][key]['relative_l2_percent']
            np.testing.assert_allclose(value,stored,rtol=1e-12,atol=1e-12);max_error=max(max_error,abs(value-stored))
            sg=s['groups'][f"{r['step']}_{r['method']}_{r['arm']}"][key+'.relative_l2_percent']['values'];assert sg[p['seeds'].index(r['seed'])]==stored
        assert r['numerical_audit']['passed'];checked+=1
        del raw,err,pred
        if checked%9==0:print('AUDIT_FIELDS',checked,flush=True)
    assert checked==54
    old={}
    for folder in sorted((ROOT/'results').iterdir()):
        cp=folder/'completion.json'
        if folder==B or not cp.exists():continue
        c=read(cp);files=c.get('files_sha256',c.get('batch_files_sha256',{}))
        for f,h in files.items():assert sha(folder/f)==h,(folder,f)
        for f,h in c.get('supporting_files_sha256',{}).items():assert sha(f)==h,f
        old[folder.name]=dict(verified_files=len(files),completion_sha256=sha(cp),unchanged=True)
    first=read(ROOT.parent/'00_基线与规则/第一轮保全核验.json');assert first['passed'] and first['files_checked']==9923
    audit=dict(checked_fields=54,new_FEM_fields=36,continuous_reused_FEM_fields=18,raw_component_region_and_L2_checks=True,paired_points_and_checkpoint_checks=True,maximum_primary_L2_difference=max_error)
    (E/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    supports=[ROOT/'protocols/R2_P2H_sampling_intervention.json',ROOT/'P2H_Sampling_Intervention_Report.md']+[ROOT/'code'/f for f in ['train_p2h.py','evaluate_p2h.py','report_p2h.py','audit_close_p2h.py']]
    c=dict(completed_utc=datetime.now(timezone.utc).isoformat(),training_continuations=18,reused_prefix_steps_per_continuation=400,physical_fields=54,new_FEM_fields=36,formal_timing=False,
        files_sha256={f.relative_to(B).as_posix():sha(f) for f in sorted(B.rglob('*')) if f.is_file()},supporting_files_sha256={str(f.resolve()):sha(f) for f in supports},prior_batches_preserved=old,first_round_preservation=first,audit=audit)
    (B/'completion.json').write_text(json.dumps(c,indent=2,ensure_ascii=False),encoding='utf-8');print('P2H_CLOSED',len(c['files_sha256']),'files; all 54 endpoints audited; prior batches unchanged',flush=True)
if __name__=='__main__':main()
