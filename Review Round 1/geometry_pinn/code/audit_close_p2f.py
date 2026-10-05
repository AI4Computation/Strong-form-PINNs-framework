"""Independent raw-array arithmetic checks and immutable P2F closure."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2F_repetition_budget';E=B/'evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def main():
    assert not (B/'completion.json').exists()
    p=read(ROOT/'protocols/R2_P2F_repetition_budget.json');m=read(B/'manifest.json');a=read(E/'analysis.json');s=read(E/'summary.json')
    assert m['status']=='fit_complete' and len(m['completed'])==27
    assert read(E/'progress.json')['completed_fields']==81
    assert sha(E/'analysis.json')==s['source_analysis_sha256']
    assert sha(ROOT/'code/summarize_p2f.py')==s['summary_code_sha256']
    assert sha(B/'manifest.json')==a['training_manifest_sha256']
    assert sha(ROOT/'protocols/R2_P2F_repetition_budget.json')==m['protocol_sha256']
    for f,h in m['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in a['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in m['prepared_sha256'].items():assert sha(B/f)==h
    for run,files in m['terminal_sha256'].items():
        for f,h in files.items():assert sha(B/run/f)==h
        r=read(B/run/'result.json');assert r['parameters']==110705 and r['accepted_steps']==1600
        assert r['stop_reason']=='max_iter' and not r['fem_read'] and not r['timing_valid_for_comparison']
    checked=0;max_difference=0.
    for case,seeds in a['cases'].items():
        ref=load(a['references'][case]['path']);assert sha(a['references'][case]['path'])==a['references'][case]['sha256']
        for seed,steps in seeds.items():
            for step,methods in steps.items():
                for method,row in methods.items():
                    pred=load(row['predictions_path']);assert sha(row['predictions_path'])==row['predictions_sha256']
                    ep=E/(Path(row['predictions_path']).name.replace('_predictions.npz','_error_norms.npz'))
                    err=load(ep);assert sha(ep)==row['error_norms_sha256']
                    for field,estimate,truth,weights,metric in [
                        ('u',pred['q_ip'][:,:2],ref['u_ip'],ref['area_weight'],'u_area'),
                        ('s',pred['q_ip'][:,2:],ref['s_ip'],ref['area_weight'],'s_area'),
                        ('wall_u',pred['u_wall'],ref['wall_u'],ref['wall_weight'],'wall_u')]:
                        delta=estimate-truth;norm=np.sqrt(np.sum(delta*delta,axis=1))
                        np.testing.assert_allclose(norm,err[field],rtol=1e-12,atol=1e-14)
                        calculated=100*np.sqrt(np.sum(weights*np.sum(delta*delta,axis=1))/np.sum(weights*np.sum(truth*truth,axis=1)))
                        recorded=row['metrics'][metric]['relative_l2_percent'];max_difference=max(max_difference,abs(calculated-recorded))
                        np.testing.assert_allclose(calculated,recorded,rtol=1e-12,atol=1e-12)
                        vals=s['groups'][f'{case}_step{int(step):04d}'][method][metric+'.relative_l2_percent']['values']
                        assert vals[p['seeds'].index(int(seed))]==recorded
                    de=pred['u_extrema']-ref['extrema_u']
                    # Stored convergence subtracts float32 predictions before the float64 reference;
                    # the independent double-precision difference must allow this rounding.
                    ct=8*np.finfo(np.float32).eps*max(1.,float(abs(pred['u_extrema']).max()))
                    np.testing.assert_allclose(np.abs([de[0,1]-de[1,1],de[2,0]-de[3,0]]),row['metrics']['engineering']['convergence_absolute_error'],rtol=1e-9,atol=ct)
                    checked+=1
                    del pred,err
                print('RAW_ARRAY_AUDIT',case,seed,step,'checked',checked,flush=True)
        del ref
    assert checked==81
    prior={}
    for folder in sorted((ROOT/'results').iterdir()):
        cp=folder/'completion.json'
        if folder==B or not cp.exists():continue
        old=read(cp);files=old.get('files_sha256',old.get('batch_files_sha256',{}))
        for f,h in files.items():assert sha(folder/f)==h,(folder,f)
        for f,h in old.get('supporting_files_sha256',{}).items():assert sha(f)==h,f
        prior[folder.name]=dict(completion_sha256=sha(cp),verified_files=len(files),unchanged=True)
    first=read(ROOT.parent/'00_基线与规则/第一轮保全核验.json');assert first['passed'] and first['files_checked']==9923
    report=ROOT/'P2F_Repetition_Budget_Report.md';assert report.exists()
    supports=[ROOT/'protocols/R2_P2F_repetition_budget.json',report]+[ROOT/'code'/f for f in ['prepare_p2f.py','train_p2f.py','evaluate_p2f.py','summarize_p2f.py','report_p2f.py','audit_close_p2f.py']]
    audit=dict(checked_fields=checked,independent_primary_L2_max_absolute_difference=max_difference,
        full_prediction_to_saved_error_norm_checks=True,engineering_convergence_checks=True,summary_values_checked=True,
        convergence_audit_tolerance='8 float32 epsilon times max(1, max absolute endpoint displacement); independently recomputed in float64')
    (E/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    files={f.relative_to(B).as_posix():sha(f) for f in sorted(B.rglob('*')) if f.is_file()}
    close=dict(completed_utc=datetime.now(timezone.utc).isoformat(),training_runs=27,evaluation_fields=81,formal_timing=False,
        all_training_frozen_before_FEM=True,files_sha256=files,supporting_files_sha256={str(f.resolve()):sha(f) for f in supports},
        referenced_input_sha256={v['path']:v['sha256'] for v in a['references'].values()},prior_batches_preserved=prior,
        first_round_preservation=first,audit=audit)
    (B/'completion.json').write_text(json.dumps(close,indent=2,ensure_ascii=False),encoding='utf-8')
    print('P2F_CLOSED',len(files),'files; all 81 raw fields checked; prior batches and first round unchanged',flush=True)
if __name__=='__main__':main()
