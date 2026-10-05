"""Read-only DEM trajectory and saved-checkpoint diagnosis; no retraining."""
from pathlib import Path
import csv, datetime, hashlib, json, statistics, sys

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT.parent/'controlled_pinn'

def write_csv(name,rows):
    folder=ROOT/'summary';folder.mkdir(parents=True,exist_ok=True)
    with (folder/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)

def main():
    manifest=json.loads((RESEARCH/'config/run_manifest.json').read_text())
    configs=sorted([c for c in manifest if c['method']=='dem'],key=lambda c:c['id'])
    assert len(configs)==46
    rows=[];trajectories=[];result_hashes={};snapshots=[]
    for c in configs:
        path=RESEARCH/'runs'/c['id']/'result.json'
        r=json.loads(path.read_text());result_hashes[c['id']]=hashlib.sha256(path.read_bytes()).hexdigest()
        trace=r['metric_trace'];best=min(trace,key=lambda t:t['s_vector_pct']);best_u=min(trace,key=lambda t:t['u_vector_pct'])
        final=trace[-1]
        rows.append({'id':c['id'],'case':c['case'],'seed':c['seed'],
            'best_observed_s_step':best['accepted_step'],'best_observed_s_pct':best['s_vector_pct'],
            'u_pct_at_best_observed_s':best['u_vector_pct'],'loss_at_best_observed_s':best['loss'],
            'best_observed_u_step':best_u['accepted_step'],'best_observed_u_pct':best_u['u_vector_pct'],
            'final_s_pct':final['s_vector_pct'],'final_u_pct':final['u_vector_pct'],'final_loss':final['loss'],
            'final_over_best_observed_s':final['s_vector_pct']/best['s_vector_pct'],
            'best_observed_s_optimization_s':best['optimization_s'],
            'final_optimization_s':final['optimization_s'],
            'selection_scope':'Retrospective FEM-assisted diagnostic only; not a deployable stopping rule.'})
        for t in trace:
            trajectories.append({'id':c['id'],'case':c['case'],'seed':c['seed'],**{k:t[k] for k in
                ['accepted_step','loss','u_vector_pct','s_vector_pct','optimization_s','available_wall_s']}})
        for step in [100,250,500,750,1000]:
            t=next(v for v in trace if v['accepted_step']==step)
            snapshots.append({'id':c['id'],'case':c['case'],'seed':c['seed'],**{k:t[k] for k in
                ['accepted_step','loss','u_vector_pct','s_vector_pct','optimization_s']}})
    write_csv('dem_history_per_run.csv',rows)
    write_csv('dem_error_energy_history.csv',trajectories)
    write_csv('dem_fixed_step_diagnostics.csv',snapshots)
    cases=[]
    for case in sorted({r['case'] for r in rows}):
        group=[r for r in rows if r['case']==case]
        cases.append({'case':case,'n':len(group),
            'best_observed_s_mean':statistics.mean(r['best_observed_s_pct'] for r in group),
            'best_observed_s_sd':statistics.stdev(r['best_observed_s_pct'] for r in group),
            'best_observed_s_step_median':statistics.median(r['best_observed_s_step'] for r in group),
            'best_observed_u_mean':statistics.mean(r['best_observed_u_pct'] for r in group),
            'final_s_mean':statistics.mean(r['final_s_pct'] for r in group),
            'n_final_more_than_twice_best_s':sum(r['final_over_best_observed_s']>2 for r in group)})
    write_csv('dem_history_by_case.csv',cases)

    # Check the same pre-existing snapshots for every core-case seed. No weights at
    # other minima are reconstructed or claimed to have been retained.
    sys.path.insert(0,str(RESEARCH/'code'))
    from check_dem_quadrature import torch,legacy,rule,energy
    rules={n:rule(n) for n in [79,158,316]}
    with (RESEARCH/'summary/dem_quadrature_audit.csv').open(encoding='utf-8-sig') as f:
        endpoint=list(csv.DictReader(f))
    energy_rows=[];checkpoint_hashes={};done=0
    for c in [c for c in configs if c['case'] in ['C1','C8']]:
        for step in [100,500,1000]:
            folder=RESEARCH/'runs'/c['id'];cp=folder/f'step_{step:05d}.pt'
            checkpoint_hashes[str(cp.relative_to(RESEARCH))]=hashlib.sha256(cp.read_bytes()).hexdigest()
            t=next(t for t in trajectories if t['id']==c['id'] and t['accepted_step']==step)
            if step<1000:
                model=legacy.DEMNet().to('cpu')
                model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict'])
            for order,q in rules.items():
                if step==1000:
                    v=next(v for v in endpoint if v['id']==c['id'] and int(v['order'])==order)
                    internal,work,potential=[float(v[k]) for k in ['internal_energy','external_work','potential_energy']]
                else:
                    internal,work,potential=energy(model,q,c['p_lateral'],c['p_top'])
                energy_rows.append({'id':c['id'],'case':c['case'],'seed':c['seed'],
                    'accepted_step':step,'order':order,'internal_energy':internal,
                    'external_work':work,'potential_energy':potential,
                    'recorded_training_loss':t['loss'],'s_vector_pct':t['s_vector_pct'],'u_vector_pct':t['u_vector_pct']})
            if step<1000:
                done+=1
                if done%8==0:print(f'CHECKPOINTS {done}/32',flush=True)
    assert len(energy_rows)==144
    write_csv('dem_saved_checkpoint_quadrature.csv',energy_rows)
    report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'n_runs':46,'n_minimum_before_final':sum(r['best_observed_s_step']<1000 for r in rows),
        'n_final_s_more_than_twice_best':sum(r['final_over_best_observed_s']>2 for r in rows),
        'n_final_s_more_than_ten_times_best':sum(r['final_over_best_observed_s']>10 for r in rows),
        'case_statistics':cases,'core_checkpoints':48,'new_checkpoint_evaluations':32,
        'quadrature_orders':[79,158,316],
        'scope':'All 46 DEM histories; common stored steps 100, 500, 1000 for all 16 C1/C8 runs. Endpoint quadrature reused unchanged. No new training, no checkpoint selected for formal performance reporting.',
        'limits':'Retrospective minimum errors use FEM and are diagnostic only. Error observations every 50 steps plus early step 10; unsaved minimum-step weights are unavailable. Dense quadrature is not a certified converged integral.',
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'result_sha256':result_hashes,'checkpoint_sha256':checkpoint_hashes}
    (ROOT/'summary/dem_history_diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['n_runs','n_final_s_more_than_twice_best','n_final_s_more_than_ten_times_best','core_checkpoints']}),flush=True)

if __name__=='__main__':main()
