"""Independent exact-state, timing/target arithmetic and preservation checks."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import json,hashlib,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2L_formal_cost'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    assert not (B/'completion.json').exists();m=read(B/'manifest.json');s=read(B/'summary.json');p=read(ROOT/'protocols/R2_P2L_formal_cost.json');assert m['status']=='timing_complete' and len(m['completed'])==18 and len(s['rows'])==36
    assert sha(B/'manifest.json')==s['timing_manifest_sha256'];assert sha(ROOT/'protocols/R2_P2L_formal_cost.json')==m['protocol_sha256']==s['protocol_sha256']
    for f,h in m['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in m['inputs_sha256'].items():assert sha(f)==h
    for g in m['gate_records']:assert g['passed'] and g['ac']==1 and g['system_cpu_percent']<=15 and float(g['gpu'].split(',')[0])<=10
    assert [v['index'] for v in m['run_records']]==list(range(18))
    matches=0;pointsets={};querysets={};peak_cpu=[]
    for rr in m['run_records']:
        folder=B/rr['folder'];r=read(folder/'result.json');src=Path(r['source_scientific_run']);job=p['jobs'][rr['index']]
        assert all(r[k]==v for k,v in job.items());assert r['parameters']==110705 and r['accepted_steps']==1600 and r['formal_timing']
        original=read(src/'result.json');assert r['closure_evaluations']==original['closure_evaluations'] and [v['closures'] for v in r['blocks']]==[v['closures'] for v in original['blocks']]
        assert read(folder/'trace.json')==read(src/'trace.json')
        for i,step in enumerate([400,800,1200,1600]):
            a=torch.load(src/f'step_{step:04d}.pt',map_location='cpu',weights_only=True);b=torch.load(folder/f'step_{step:04d}.pt',map_location='cpu',weights_only=True);assert set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items());matches+=1
            block=r['blocks'][i];assert block['accepted_steps']==400 and block['cumulative_step']==step
            for k,v in block.items():
                if k.endswith('_seconds'):assert math.isfinite(v) and v>=0
            assert block['elapsed_before_checkpoint_io_seconds']<=block['end_to_end_seconds']
        assert r['components']['solve_end_to_end_seconds']>=r['blocks'][-1]['end_to_end_seconds']
        assert rr['process_envelope_seconds']>=r['components']['solve_end_to_end_seconds']+r['warmup_seconds']
        samples=read(folder/'resource_samples.json');assert samples and all(v['ac']==1 for v in samples);assert r['peak_sampled_rss_bytes']==max(v['rss'] for v in samples);peak_cpu.append(max(v['system_cpu_percent'] for v in samples))
        assert r['peak_reserved_bytes']>=r['peak_allocated_bytes']>0
        with np.load(folder/'output_queries.npz') as q:
            xy,qv=q['xy'],q['q'];assert xy.shape==(10000,2) and qv.shape==(10000,5) and np.isfinite(qv).all()
            if r['case'] in querysets:assert np.array_equal(xy,querysets[r['case']])
            else:querysets[r['case']]=xy.copy()
        assert sha(folder/'result.json')==s['runs'][r['name']]['result_sha256']
    for v in s['targets'].values():
        count=0;last=0
        for seed,q in v['by_seed'].items():
            rows=[s['rows'][f"{v['case']}_seed{seed}_{v['method']}_uniform_refresh_step{step:04d}"] for step in [800,1600]]
            hits=[all(z['metrics'][k]['relative_l2_percent']<=threshold for k,threshold in zip(['u_area','s_area','wall_u'],v['target'])) for z in rows];assert q['hits']==hits
            first=next((i for i,x in enumerate(hits) if x),None);assert q['first_observed_seconds']==(None if first is None else rows[first]['elapsed_seconds']);assert q['censor_seconds']==(rows[1]['elapsed_seconds'] if first is None else None);assert q['lost_at1600']==(hits[0] and not hits[1]);count+=first is not None;last+=hits[1]
        assert count==v['ever_count'] and last==v['at1600_count']
    for key,q in s['contrasts'].items():
        case,step,control=key.split('_',2);ratios=[]
        for seed in p['seeds']:
            g=s['rows'][f'{case}_seed{seed}_geometry_rbf_fourier_uniform_refresh_step{int(step):04d}'];c=s['rows'][f'{case}_seed{seed}_{control}_uniform_refresh_step{int(step):04d}'];ratios.append(g['elapsed_seconds']/c['elapsed_seconds'])
        np.testing.assert_allclose(ratios,q['elapsed_ratios'],rtol=1e-14);np.testing.assert_allclose(np.prod(ratios)**(1/3),q['geometric_mean'],rtol=1e-14)
    old={}
    for folder in sorted((ROOT/'results').iterdir()):
        cp=folder/'completion.json'
        if folder==B or not cp.exists():continue
        closed=read(cp);files=closed.get('files_sha256',closed.get('batch_files_sha256',{}))
        for f,h in files.items():assert sha(folder/f)==h,(folder,f)
        for f,h in closed.get('supporting_files_sha256',{}).items():assert sha(f)==h,f
        old[folder.name]=dict(verified_files=len(files),completion_sha256=sha(cp),unchanged=True)
    stopped=ROOT/'results/R2_P2I_shape_transfer';snap=read(stopped/'suspension.json')
    for f,h in snap['files_sha256'].items():assert sha(stopped/f)==h
    for f,h in snap['supporting_files_sha256'].items():assert sha(f)==h
    first=read(ROOT.parent/'00_基线与规则/第一轮保全核验.json');assert first['passed'] and first['files_checked']==9923
    audit=dict(formal_runs_checked=18,exact_checkpoints=matches,identical_accepted_step_traces=18,accuracy_endpoints=36,target_and_cost_arithmetic_checked=True,all_idle_and_AC_gates_pass=True,worker_peak_system_cpu_percent=peak_cpu,common_reference_free_queries=True,torch_version=torch.__version__,cuda_runtime=torch.version.cuda,first_round_preserved=True);write(B/'audit.json',audit)
    supports=[ROOT/'P2L_Formal_Cost_Report.md',ROOT/'protocols/R2_P2L_formal_cost.json']+[ROOT/'code'/f for f in ['bench_p2l.py','bench_p2l_worker.py','report_p2l.py','audit_close_p2l.py']]
    completion=dict(completed_utc=datetime.now(timezone.utc).isoformat(),formal_timing_replays=18,new_scientific_conditions=0,exact_checkpoints=72,joined_accuracy_endpoints=36,files_sha256={f.relative_to(B).as_posix():sha(f) for f in sorted(B.rglob('*')) if f.is_file()},supporting_files_sha256={str(f.resolve()):sha(f) for f in supports},prior_batches_preserved=old,original_P2I_suspension_unchanged=True,first_round_preservation=first,audit=audit);write(B/'completion.json',completion);print('P2L_CLOSED',len(completion['files_sha256']),'files',flush=True)
if __name__=='__main__':main()
