"""Preserve a reference-blocked batch without declaring its evaluation complete."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import torch
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2I_shape_transfer'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    assert not (B/'suspension.json').exists();m=read(B/'manifest.json');p=read(ROOT/'protocols/R2_P2I_shape_transfer.json');assert m['status']=='fit_complete' and len(m['completed'])==12
    assert sha(ROOT/'protocols/R2_P2I_shape_transfer.json')==m['protocol_sha256']
    for f,h in m['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    for f,h in m['prepared_sha256'].items():assert sha(B/f)==h
    for f,h in m['inputs_sha256'].items():assert sha(f)==h
    for name,files in m['terminal_sha256'].items():
        for f,h in files.items():assert sha(B/name/f)==h
        r=read(B/name/'result.json');assert r['accepted_steps']==1600 and r['parameters']==110705 and len(r['blocks'])==4 and not r['fem_read']
        assert all(v['accepted_steps']==400 and v['stop_reason']=='max_iter' for v in r['blocks'])
    pairs=0
    for case in p['cases']:
      for method in p['methods']:
        states=[torch.load(B/f'{case}_seed{p["seed"]}_{method}_{arm}'/'step_0400.pt',map_location='cpu',weights_only=True) for arm in p['arms']]
        assert set(states[0])==set(states[1]) and all(torch.equal(v,states[1][k]) for k,v in states[0].items());pairs+=1
    progress=read(B/'evaluation/progress.json');assert progress['status']=='failed' and progress['completed_fields']==0
    old={}
    for folder in sorted((ROOT/'results').iterdir()):
        cp=folder/'completion.json'
        if folder==B or not cp.exists():continue
        c=read(cp);files=c.get('files_sha256',c.get('batch_files_sha256',{}))
        for f,h in files.items():assert sha(folder/f)==h,(folder,f)
        for f,h in c.get('supporting_files_sha256',{}).items():assert sha(f)==h,f
        old[folder.name]=dict(verified_files=len(files),completion_sha256=sha(cp),unchanged=True)
    first=read(ROOT.parent/'00_基线与规则/第一轮保全核验.json');assert first['passed'] and first['files_checked']==9923
    supports=[ROOT/'protocols/R2_P2I_shape_transfer.json',ROOT/'P2I_Shape_Transfer_Report.md']+[ROOT/'code'/f for f in ['train_p2i.py','transfer_mechanics.py','evaluate_p2i.py','transfer_evaluation_helpers.py','report_p2i.py','freeze_p2i_reference_stop.py']]
    result=dict(status='reference_blocked_evaluation_incomplete',recorded_utc=datetime.now(timezone.utc).isoformat(),training_trajectories=12,accepted_steps_per_trajectory=1600,frozen_checkpoints=48,exact_first_block_pairs=pairs,evaluated_fields=0,registered_fields_remaining=24,formal_timing=False,reason=read(B/'evaluation/failure.json')['error'],files_sha256={f.relative_to(B).as_posix():sha(f) for f in sorted(B.rglob('*')) if f.is_file()},supporting_files_sha256={str(f.resolve()):sha(f) for f in supports},prior_batches_preserved=old,first_round_preservation=first,next='Read-only mesh/analytic-ellipse boundary consistency diagnosis; no threshold relaxation, no method ranking and no retraining. Any repaired evaluator belongs to a separately registered output, keeping this snapshot intact.')
    (B/'suspension.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print('P2I_TRAINING_PRESERVED_REFERENCE_STOP',len(result['files_sha256']),'files; 12 complete fits; 0/24 evaluated; prior batches',len(old))
if __name__=='__main__':main()
