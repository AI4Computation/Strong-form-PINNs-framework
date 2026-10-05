"""One planned conditional bandwidth extension; preserve all base-run sources."""
import runtime
from runtime import torch
import train
from models import ROOT
import json,hashlib,datetime,argparse
import numpy as np
from summarize import write_csv

ORIGINAL_BUILD=train.build_model
ORIGINAL_HASHES=train.code_hashes

def build_extended(method,seed,geometry='circle'):
    if method!='fourier_quarter':return ORIGINAL_BUILD(method,seed,geometry)
    model=ORIGINAL_BUILD('fourier',seed,geometry)
    with torch.no_grad():model.B.mul_(.25)
    return model

def extension_hashes():
    return {**ORIGINAL_HASHES(),'conditional_extension':hashlib.sha256(__file_bytes()).hexdigest()}

def __file_bytes():
    from pathlib import Path
    return Path(__file__).read_bytes()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify_initialization():
    checks=[]
    for seed in range(41,49):
        quarter=build_extended('fourier_quarter',seed)
        normal=ORIGINAL_BUILD('fourier',seed)
        assert torch.equal(quarter.B,normal.B*.25)
        assert all(torch.equal(v,normal.net.state_dict()[k]) for k,v in quarter.net.state_dict().items())
        checks.append({'seed':seed,'same_downstream_initial_state':True,'B_exactly_default_divided_by_4':True})
    return checks

def summarize_extension(manifest):
    import csv
    with (ROOT/'summary/fourier_bandwidth_sensitivity.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    reports=[];audits=[];rng=np.random.default_rng(20260920)
    for c in manifest:
        folder=ROOT/'runs'/c['id'];r=json.loads((folder/'result.json').read_text())
        default_folder=ROOT/'runs'/f"{c['case']}_fourier_s{c['seed']}"
        default=json.loads((default_folder/'result.json').read_text())
        assert r['collocation_hashes']==default['collocation_hashes']
        assert r['reference_sha256']==default['reference_sha256']
        assert r['code_hashes']==extension_hashes()
        assert {k:v for k,v in r['configuration'].items() if k not in ('id','method')}=={k:v for k,v in default['configuration'].items() if k not in ('id','method')}
        initial=torch.load(folder/'step_00000.pt',map_location='cpu',weights_only=True)['state_dict']
        paired=torch.load(default_folder/'step_00000.pt',map_location='cpu',weights_only=True)['state_dict']
        assert torch.equal(initial['B'],paired['B']*.25)
        assert all(torch.equal(v,paired[k]) for k,v in initial.items() if k.startswith('net.'))
        assert len(r['accepted_history'])==r['accepted_steps']+1 and r['metric_trace'][-1]['accepted_step']==r['accepted_steps']
        rows.append({'id':c['id'],'case':c['case'],'seed':c['seed'],'sigma_over_default':.25,
            'sigma_default':20/(2*np.pi),**r['metrics'],'optimization_s':r['optimization_s'],'total_wall_s':r['total_wall_s']})
        audits.append(c['id'])
    write_csv('fourier_bandwidth_extended.csv',rows)
    for case in ['C1','C8']:
        chosen=sorted([c for c in manifest if c['case']==case],key=lambda c:c['seed'])
        indices=rng.integers(0,8,(10000,8))
        for metric in ['s_vector_pct','u_vector_pct','s_near_pct']:
            values=[];anchor=[]
            for c in chosen:
                r=json.loads((ROOT/'runs'/c['id']/'result.json').read_text())
                a=json.loads((ROOT/'runs'/f"{case}_anchored_s{c['seed']}"/'result.json').read_text())
                values.append(r['metrics'][metric]);anchor.append(a['metrics'][metric])
            values=np.array(values);difference=np.array(anchor)-values
            lo,hi=np.quantile(difference[indices].mean(1),[.025,.975])
            reports.append({'case':case,'metric':metric,'n':8,'fourier_quarter_mean':float(values.mean()),
                'fourier_quarter_sd':float(values.std(ddof=1)), 'mean_anchored_minus_quarter_pp':float(difference.mean()),
                'ci95_low':float(lo),'ci95_high':float(hi),
                'raw_differences_pp':difference.tolist()})
    train.write_json(ROOT/'checks/fourier_extension_audit.json',{'passed':True,'n_runs':len(audits),
        'checks':'Reference, code, config, initial B/downstream parameters, samples, accepted counts and final trace verified.',
        'runs':audits})
    train.write_json(ROOT/'summary/fourier_extension_summary.json',{'results':reports,
        'scope':'Conditional one-step extension of the prespecified sensitivity scan; all original results retained. This is not an independent test of a globally optimal bandwidth.'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    base=json.loads((ROOT/'runs/C1_anchored_s41/result.json').read_text())
    assert ORIGINAL_HASHES()==base['code_hashes']
    checks=verify_initialization()
    train.write_json(ROOT/'checks/fourier_extension_initialization.json',{'passed':True,'checks':checks})
    if args.verify_only:print('Quarter-bandwidth initialization verified.');return
    manifest=json.loads((ROOT/'config/fourier_extension_manifest.json').read_text());assert len(manifest)==16
    train.build_model=build_extended;train.code_hashes=extension_hashes
    frozen=extension_hashes();started=now();done=sum((ROOT/'runs'/c['id']/'result.json').exists() for c in manifest)
    try:
        for c in manifest:
            assert extension_hashes()==frozen
            train.write_json(ROOT/'进度.json',{'status':'conditional_fourier_extension','updated_utc':now(),
                'extension_started_utc':started,'base_completed':334,'base_planned':334,
                'extension_completed':done,'extension_planned':16,'completed':334+done,'planned':350,'current':c['id']})
            existed=(ROOT/'runs'/c['id']/'result.json').exists()
            print('EXTENSION '+c['id'],flush=True);train.run_one(c)
            if not existed:done+=1
        summarize_extension(manifest)
        train.write_json(ROOT/'进度.json',{'status':'computed_pending_scientific_and_visual_review','updated_utc':now(),
            'base_completed':334,'base_planned':334,'extension_completed':16,'extension_planned':16,
            'completed':350,'planned':350,'note':'Base protocol and one conditional bandwidth extension complete; scientific review and figure QA remain.'})
    except Exception:
        import traceback
        train.write_json(ROOT/'进度.json',{'status':'failed','updated_utc':now(),'base_completed':334,
            'extension_completed':done,'extension_planned':16,'error':traceback.format_exc()})
        raise

if __name__=='__main__':main()
