"""Independent raw-residual arithmetic and preservation checks before closure."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2G_frozen_physics'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    assert not (B/'completion.json').exists();a=read(B/'analysis.json');s=read(B/'summary.json');p=ROOT/'protocols/R2_P2G_frozen_physics.json'
    assert read(B/'progress.json')['completed_fields']==81 and len(a['results'])==81 and not a['fem_read']
    assert sha(p)==a['protocol_sha256'] and sha(B/'analysis.json')==s['diagnostic_analysis_sha256']
    for f,h in a['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in a['points_sha256'].items():assert sha(B/f)==h
    assert sha(ROOT/'code/report_p2g.py')==s['report_code_sha256']
    checked=0;maxdiff=0.
    for tag,row in a['results'].items():
        raw=B/f'{tag}_residuals.npz';assert sha(raw)==row['raw_sha256']
        with np.load(raw) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            for i in range(2):
                r=z[f'validation{i}'];assert r.shape==(24000,5)
                comp=np.mean(r*r,axis=0);value=float(np.sum(r*r)/len(r));v=row['validation'][i]
                np.testing.assert_allclose(comp,v['component_mean'],rtol=1e-11,atol=1e-12)
                np.testing.assert_allclose(value,v['total_mean'],rtol=1e-11,atol=1e-12);maxdiff=max(maxdiff,abs(value-v['total_mean']))
                pointfile=B/('C1_points.npz' if row['case'] in ['C1','C8'] else 'L1_points.npz')
                with np.load(pointfile) as coords:
                    for name,rs in row['regions'][i].items():
                        mask=coords[f'validation{i}_{name}'];vals=(r*r).sum(1)
                        np.testing.assert_allclose(vals[mask].mean(),rs['total_mean'],rtol=1e-11,atol=1e-12)
                        np.testing.assert_allclose(vals[mask].sum()/vals.sum(),rs['residual_square_mass'],rtol=1e-11,atol=1e-12)
            assert row['numerical_audit']['passed']
            assert row['training_objective_difference']<2e-5*max(.01,abs(row['training_trace_objective']))
        checked+=1
    first=read(ROOT.parent/'00_基线与规则/第一轮保全核验.json');assert first['passed'] and first['files_checked']==9923
    old={}
    for folder in sorted((ROOT/'results').iterdir()):
        cp=folder/'completion.json'
        if folder==B or not cp.exists():continue
        c=read(cp);files=c.get('files_sha256',c.get('batch_files_sha256',{}))
        for f,h in files.items():assert sha(folder/f)==h,(folder,f)
        for f,h in c.get('supporting_files_sha256',{}).items():assert sha(f)==h,f
        old[folder.name]=dict(verified_files=len(files),completion_sha256=sha(cp),unchanged=True)
    audit=dict(checked_fields=checked,independent_raw_component_total_and_region_checks_passed=True,max_absolute_mean_square_difference=maxdiff)
    (B/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    supports=[p,ROOT/'P2G_Frozen_Physics_Report.md']+[ROOT/'code'/n for n in ['diagnose_p2g.py','report_p2g.py','audit_close_p2g.py']]
    closure=dict(completed_utc=datetime.now(timezone.utc).isoformat(),diagnostic_fields=81,independent_sets_per_field=2,points_per_set=24000,training_runs=0,formal_timing=False,
        files_sha256={f.relative_to(B).as_posix():sha(f) for f in sorted(B.rglob('*')) if f.is_file()},supporting_files_sha256={str(f.resolve()):sha(f) for f in supports},prior_batches_preserved=old,first_round_preservation=first,audit=audit)
    (B/'completion.json').write_text(json.dumps(closure,indent=2,ensure_ascii=False),encoding='utf-8')
    print('P2G_CLOSED',len(closure['files_sha256']),'files,',checked,'raw-field audits,',len(old),'prior batches unchanged')
if __name__=='__main__':main()
