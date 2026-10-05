"""Serial two-seed repetition using the byte-verified frozen P2I trainer."""
from pathlib import Path
import traceback
import train_p2i as t
ROOT=t.ROOT;OUT=ROOT/'results/R2_P2K_transfer_replication';P=ROOT/'protocols/R2_P2K_transfer_replication.json'
def main():
    assert not OUT.exists();p=t.read(P);j=ROOT/'results/R2_P2J_reference_repair';assert t.sha(j/'completion.json')==p['P2J_completion_sha256'];assert t.sha(ROOT/'code/train_p2i.py')==p['P2I_training_source_sha256']
    c=t.read(j/'completion.json')
    for f,h in c['files_sha256'].items():assert t.sha(j/f)==h
    for f,h in c['supporting_files_sha256'].items():assert t.sha(f)==h
    OUT.mkdir();m=dict(status='running',created_utc=t.utc(),protocol_sha256=t.sha(P),driver_sha256=t.sha(__file__),trainer_sha256=t.sha(ROOT/'code/train_p2i.py'),completed_seeds=[],active_seed=None,seed_manifest_sha256={},fem_solution_arrays_read=False,formal_timing=False)
    t.write(OUT/'manifest.json',m)
    try:
        for seed in p['seeds']:
            t.B=OUT/f'seed{seed}';t.P=ROOT/f'protocols/R2_P2K_seed{seed}.json';m['active_seed']=seed;t.write(OUT/'manifest.json',m);t.main()
            s=t.read(t.B/'manifest.json');assert s['status']=='fit_complete' and len(s['completed'])==12;m['completed_seeds'].append(seed);m['seed_manifest_sha256'][str(seed)]=t.sha(t.B/'manifest.json');t.write(OUT/'manifest.json',m)
        m.update(status='fit_complete',active_seed=None,finished_utc=t.utc());t.write(OUT/'manifest.json',m);print('ALL_24_REPLICATION_TRAJECTORIES_FROZEN_NO_FEM',flush=True)
    except BaseException as e:
        m.update(status='failed',error=repr(e),traceback=traceback.format_exc());t.write(OUT/'manifest.json',m);raise
if __name__=='__main__':main()
