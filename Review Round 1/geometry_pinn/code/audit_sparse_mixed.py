"""No optimizer: CPU sparse/dense, spatial and parameter derivative audit."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import hashlib,json,sys
from datetime import datetime
import numpy as np
import torch
from sparse_mixed_pinn import SparseMixedPINN,residual

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'geometry_pinn/protocols/R2_P1B_sparse_operator.json'
OUTPUT=ROOT/'geometry_pinn/results/R2_P1B_sparse_operator.json'


def main():
    assert not OUTPUT.exists(),'Audit is frozen; do not overwrite.'
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));torch.set_num_threads(p['cpu_threads'])
    report={'id':p['id'],'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('sparse_mixed_pinn.py')]},
            'python':sys.executable,'torch':torch.__version__,'device':'CPU','cases':{},'training_runs':0,'fem_access':False,'timing_valid_for_comparison':False}
    for case in p['cases']:
        with np.load(ROOT/f'geometry_pinn/results/R2_P1/{case}_cover.npz') as data:
            c,h=data['centres'],data['halfwidths'];xy=data['interior_probes'][:p['audit_points_per_geometry']]
            # One interior point per patch exercises every local parameter group.
            all_probes=data['interior_probes']
        model=SparseMixedPINN(c,h,p['target_trainable_parameters'],p['global_hidden_width'],p['seed'])
        prepared=model.prepare(xy);q,jac=model.sparse(prepared)
        x=torch.tensor(xy,dtype=torch.float64,requires_grad=True);dense=model.dense(x)
        dense_jac=torch.stack([torch.autograd.grad(dense[:,i].sum(),x,retain_graph=True)[0] for i in range(5)],dim=1)
        values={'field':float((q-dense).abs().max().detach()),'jacobian':float((jac-dense_jac).abs().max().detach()),
                'residual':float((residual(q,jac)-residual(dense,dense_jac)).abs().max().detach())}
        # A larger independent solid sample exercises nearly all local supports;
        # mandatory active-expert coverage is measured rather than assumed.
        wide=model.prepare(all_probes)
        full,full_jac=model.sparse(wide);loss=residual(full,full_jac).square().mean()+.1*full.square().mean()
        gradients=torch.autograd.grad(loss,tuple(model.parameters()))
        all_finite=all(bool(torch.all(torch.isfinite(g))) for g in gradients)
        nonzero_tensors=all(bool(torch.any(g!=0)) for g in gradients)
        active_experts=[len(torch.unique(e['expert'])) for e in wide['edges']]
        expected_experts=[len(i) for i in model.bank_indices]
        torch.manual_seed(p['seed']+1)
        directions=[torch.randn_like(v) for v in model.parameters()]
        norm=torch.sqrt(sum((d*d).sum() for d in directions));directions=[d/norm for d in directions]
        analytic=sum((g*d).sum() for g,d in zip(gradients,directions)).item()
        step=1e-5
        def move(amount):
            with torch.no_grad():
                for v,d in zip(model.parameters(),directions):v.add_(amount*d)
        def value():
            with torch.no_grad():
                f,j=model.sparse(wide)
                return (residual(f,j).square().mean()+.1*f.square().mean()).item()
        move(step);plus=value();move(-2*step);minus=value();move(step)
        fd=(plus-minus)/(2*step)
        values['parameter_directional_relative']=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-8)
        count=sum(v.numel() for v in model.parameters());delta=p['target_trainable_parameters']-count
        # With local outputs zero, all local fields and their Jacobians vanish.
        with torch.no_grad():
            for bank in model.banks:bank.w3.zero_();bank.b3.zero_()
        no_local,_=model.sparse(prepared)
        coarse=model.coarse.evaluate(2*prepared['x'],torch.zeros(len(xy),dtype=torch.long),torch.full_like(prepared['x'],2),False)
        checks={'sparse_field':values['field']<=p['tolerances']['field_absolute'],
                'spatial_derivative':values['jacobian']<=p['tolerances']['spatial_derivative_absolute'],
                'mixed_residual':values['residual']<=p['tolerances']['residual_absolute'],
                'parameter_gradient':values['parameter_directional_relative']<=p['tolerances']['parameter_directional_relative'],
                'all_parameter_tensors_active':nonzero_tensors and all_finite,
                'all_local_experts_exercised':active_experts==expected_experts,
                'parameter_budget':0<=delta<=p['target_trainable_parameters']*p['tolerances']['parameter_shortfall_fraction'],
                'sparse_local_edges':sum(len(e['point']) for e in wide['edges'])<=4*len(all_probes),
                'zero_correction':bool(torch.equal(no_local,coarse))}
        row={'passed':all(checks.values()),'checks':checks,'errors':values,'parameters':count,'target_shortfall':delta,
             'width_counts':{str(w):int(np.sum(model.widths==w)) for w in set(model.widths.tolist())},
             'local_edges':sum(len(e['point']) for e in wide['edges']),'dense_point_expert_pairs':len(all_probes)*len(c),
             'active_experts':active_experts,'expected_experts':expected_experts}
        report['cases'][case]=row;print(case,json.dumps(row),flush=True)
    report['passed']=all(r['passed'] for r in report['cases'].values());report['completed_local']=datetime.now().astimezone().isoformat()
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
