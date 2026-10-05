"""Audit complete recovered P2I fields and freeze P2J without altering P2I."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2J_reference_repair';E=B/'evaluation';TRAIN=ROOT/'results/R2_P2I_shape_transfer'
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
    assert not (B/'completion.json').exists();a=read(E/'analysis.json');s=read(E/'summary.json');m=read(TRAIN/'manifest.json');rp=ROOT/'protocols/R2_P2J_reference_repair.json'
    assert len(a['results'])==24 and read(E/'progress.json')['status']=='complete';assert sha(E/'analysis.json')==s['analysis_sha256'] and sha(ROOT/'code/report_p2j.py')==s['report_source_sha256']
    assert sha(rp)==a['repair_protocol_sha256'] and sha(TRAIN/'manifest.json')==a['manifest_sha256']
    for f,h in a['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    snapshot=read(TRAIN/'suspension.json')
    for f,h in snapshot['files_sha256'].items():assert sha(TRAIN/f)==h
    for f,h in snapshot['supporting_files_sha256'].items():assert sha(f)==h
    diag=read(B/'diagnosis.json')
    for f,h in diag['source_sha256'].items():assert sha(f)==h
    assert sha(ROOT/'code/diagnose_p2j_reference.py')==diag['source_code_sha256']
    geom=read(B/'boundary_geometry.json');assert sha(ROOT/'code/inspect_p2j_boundary.py')==geom['source_code_sha256']
    checked=0;max_difference=0.;sensitivity={}
    for case,reference in a['references'].items():
        assert sha(reference['path'])==reference['sha256']
        for f,h in reference['source_sha256'].items():assert sha(f)==h
        ref=load(reference['path']);pts=load(E/f'{case}_physics_points.npz');assert sha(E/f'{case}_physics_points.npz')==a['points_sha256'][f'{case}_physics_points.npz']
        offsets=load(E/f'{case}_wall_offset_audit.npz');assert float(offsets['delta'])==read(rp)['derived_offsets'][case]
        errs=[float(abs(offsets['u_primary_offset']-offsets[k]).max()) for k in ['u_half_offset','u_double_offset']];assert max(errs)<1e-5
        assert errs[0]==reference['half_offset_sensitivity_max'] and errs[1]==reference['wall_offset_sensitivity_max'];sensitivity[case]=errs
        for tag,row in a['results'].items():
            if row['case']!=case:continue
            assert sha(row['predictions_path'])==row['predictions_sha256'] and sha(E/f'{tag}_error_norms.npz')==row['error_norms_sha256'] and sha(E/f'{tag}_residuals.npz')==row['raw_residuals_sha256']
            pred=load(row['predictions_path']);err=load(E/f'{tag}_error_norms.npz');raw=load(E/f'{tag}_residuals.npz')
            for i in range(2):
                v=raw[f'validation{i}'];assert v.shape==(24000,5) and np.isfinite(v).all();np.testing.assert_allclose((v*v).mean(0),row['validation'][i]['component_mean'],rtol=1e-10,atol=1e-12)
                for region,rs in row['regions'][i].items():
                    mask=pts[f'validation{i}_{region}'];np.testing.assert_allclose((v[mask]*v[mask]).sum(1).mean(),rs['total_mean'],rtol=1e-10,atol=1e-12)
            for name,estimate,truth,w,key in [('u',pred['q_ip'][:,:2],ref['u_ip'],ref['area_weight'],'u_area'),('s',pred['q_ip'][:,2:],ref['s_ip'],ref['area_weight'],'s_area'),('wall_u',pred['u_wall'],ref['wall_u'],ref['wall_weight'],'wall_u')]:
                d=estimate-truth;norm=np.sqrt((d*d).sum(1));np.testing.assert_allclose(norm,err[name],rtol=1e-12,atol=1e-14)
                value=100*np.sqrt(np.sum(w*(d*d).sum(1))/np.sum(w*(truth*truth).sum(1)));stored=row['metrics'][key]['relative_l2_percent'];np.testing.assert_allclose(value,stored,rtol=1e-12,atol=1e-12);max_difference=max(max_difference,abs(value-stored))
                for region in ['near_wall','far_wall','corner','away_corner'] if name!='wall_u' else []:
                    mask=ref[region]
                    if mask.any():
                        value=100*np.sqrt(np.sum(w[mask]*(d[mask]*d[mask]).sum(1))/np.sum(w[mask]*(truth[mask]*truth[mask]).sum(1)));np.testing.assert_allclose(value,row['metrics'][name+'_'+region]['relative_l2_percent'],rtol=1e-12,atol=1e-12)
            for label in np.unique(ref['wall_tags']):
                mask=ref['wall_tags']==label;w=ref['wall_weight'][mask];d=pred['u_wall'][mask]-ref['wall_u'][mask];true=ref['wall_u'][mask];value=100*np.sqrt(np.sum(w*(d*d).sum(1))/np.sum(w*(true*true).sum(1)));np.testing.assert_allclose(value,row['metrics']['wall_u_'+label]['relative_l2_percent'],rtol=1e-12,atol=1e-12)
            conv=lambda u:np.array([u[0,1]-u[1,1],u[2,0]-u[3,0]])
            np.testing.assert_allclose(abs(conv(pred['u_extrema'])-conv(ref['extrema_u'])),row['metrics']['engineering']['convergence_absolute_error'],rtol=1e-12,atol=1e-14)
            assert row['numerical_audit']['passed'];checked+=1;del pred,err,raw
            if checked%6==0:print('AUDIT_FIELDS',checked,flush=True)
        del ref,pts
    assert checked==24;old={}
    for folder in sorted((ROOT/'results').iterdir()):
        cp=folder/'completion.json'
        if folder==B or not cp.exists():continue
        c=read(cp);files=c.get('files_sha256',c.get('batch_files_sha256',{}))
        for f,h in files.items():assert sha(folder/f)==h,(folder,f)
        for f,h in c.get('supporting_files_sha256',{}).items():assert sha(f)==h,f
        old[folder.name]=dict(verified_files=len(files),completion_sha256=sha(cp),unchanged=True)
    first=read(ROOT.parent/'00_基线与规则/第一轮保全核验.json');assert first['passed'] and first['files_checked']==9923
    audit=dict(checked_fields=24,complete_registered_endpoints=True,raw_L2_region_wall_engineering_and_physics_checks=True,maximum_primary_L2_difference=max_difference,reference_half_double_offset_differences=sensitivity,original_P2I_snapshot_unchanged=True)
    write(E/'audit.json',audit)
    supports=[rp,ROOT/'P2J_Shape_Transfer_Evaluation_Report.md']+[ROOT/'code'/f for f in ['diagnose_p2j_reference.py','inspect_p2j_boundary.py','evaluate_p2j.py','report_p2j.py','audit_close_p2j.py']]
    completion=dict(completed_utc=datetime.now(timezone.utc).isoformat(),training_runs=0,reused_P2I_training_trajectories=12,physical_and_FEM_fields=24,formal_timing=False,files_sha256={f.relative_to(B).as_posix():sha(f) for f in sorted(B.rglob('*')) if f.is_file()},supporting_files_sha256={str(f.resolve()):sha(f) for f in supports},prior_batches_preserved=old,P2I_suspension_sha256=sha(TRAIN/'suspension.json'),first_round_preservation=first,audit=audit)
    write(B/'completion.json',completion);print('P2J_CLOSED',len(completion['files_sha256']),'files; all 24 fields audited; original P2I and prior batches unchanged',flush=True)
if __name__=='__main__':main()
