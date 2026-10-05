"""Check complete coverage, immutable inputs, paired samples, and accepted traces."""
import runtime
from runtime import torch
from models import ROOT
from train import code_hashes
import numpy as np,json,hashlib

def audit(require_complete=True):
    manifest=json.loads((ROOT/'config/run_manifest.json').read_text());hashes=code_hashes()
    reference=json.loads((ROOT/'config/reference_selection.json').read_text())
    samples={};checked=0
    for c in manifest:
        folder=ROOT/'runs'/c['id'];path=folder/'result.json'
        if not path.exists():
            if require_complete:raise FileNotFoundError(path)
            continue
        r=json.loads(path.read_text(encoding='utf-8'));checked+=1
        assert r['configuration']==c and r['code_hashes']==hashes
        assert r['reference_sha256']==reference[c['geometry']]['reference_sha256'][c['case']+'.npz']
        assert r['status']=='complete' and not r['pilot']
        assert len(r['closure_losses'])==r['closure_evaluations']
        assert len(r['accepted_history'])==r['accepted_steps']+1
        assert [x['accepted_step'] for x in r['accepted_history']]==list(range(r['accepted_steps']+1))
        assert r['metric_trace'][-1]['accepted_step']==r['accepted_steps']
        assert r['stop_reason'] in ('max_iter','max_eval','gradient_tolerance','loss_change_tolerance',
                                  'step_tolerance','directional_derivative_tolerance','initial_gradient_tolerance')
        assert (np.diff([t['optimization_s'] for t in r['metric_trace']])>=-1e-6).all()
        assert (np.diff([t['available_wall_s'] for t in r['metric_trace']])>=-1e-6).all()
        assert abs(r['total_wall_s']-r['optimization_s']-r['diagnostic_and_checkpoint_s'])<1.
        key=(c['geometry'],c['seed'])
        if key in samples:assert samples[key]==r['collocation_hashes']
        else:samples[key]=r['collocation_hashes']
        for metric in ('u_vector_pct','s_vector_pct'):
            for target in [5.,2.,1.]:
                hits=[t for t in r['metric_trace'] if t[metric] is not None and t[metric]<=target]
                claimed=r['thresholds'][f'{metric}_le_{target:g}']
                assert claimed['reached']==bool(hits)
                if hits:assert claimed['accepted_step']==hits[0]['accepted_step']
        assert (folder/f"step_{r['accepted_steps']:05d}.pt").exists()
    paired_checks=0
    for case in ('C1','C8'):
        for seed in range(41,49):
            paths=[ROOT/'runs'/f'{case}_{method}_s{seed}'/'step_00000.pt' for method in
                   ['anchored','independent_gaussian','independent_marginal','fourier','fourier_half','fourier_double']]
            if not all(p.exists() for p in paths):continue
            state=[torch.load(p,map_location='cpu',weights_only=True)['state_dict'] for p in paths]
            for other in state[1:]:
                for k,v in state[0].items():
                    if k.startswith('net.'):assert torch.equal(v,other[k])
            assert torch.equal(state[0]['W'],state[1]['W']) and torch.equal(state[0]['W'],state[2]['W'])
            assert torch.equal(state[3]['B']*.5,state[4]['B']) and torch.equal(state[3]['B']*2,state[5]['B'])
            paired_checks+=1
    report={'passed':True,'complete':checked==len(manifest),'runs_checked':checked,'planned':len(manifest),
            'paired_initial_state_groups_checked':paired_checks,'collocation_hashes_agree':True,
            'accepted_steps_and_thresholds_consistent':True,'sources_and_reference_hashes_match':True}
    (ROOT/'checks/formal_results_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    import sys
    print(json.dumps(audit('--partial' not in sys.argv),indent=2))
