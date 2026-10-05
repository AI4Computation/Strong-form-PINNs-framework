"""Bounded CPU-only geometry/representation audit, without optimization or FEM."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import json,hashlib,copy
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from cavity_cover import normalized_domain,construct
from shared_geometry_features import make_shared_model,uniform_centres
from shared_geometry_budget import make_budget_compatible
R=Path(__file__).resolve().parents[1];B=R/'results/R2_P2M_representation_audit';P=R/'protocols/R2_P2M_representation_audit.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def check(model,xy):
    x=torch.tensor(xy,dtype=torch.float64,requires_grad=True);prep=model.prepare(x)
    jf=torch.autograd.functional.jacobian(lambda y:model.features(y[None])[0],x[0],vectorize=True,strategy='forward-mode')
    fe=float((jf-prep['df'][0]).abs().max())/max(1,float(jf.abs().max()))
    q,j=model.sparse(prep);ind=model(x);ja=torch.stack([torch.autograd.grad(ind[:,k].sum(),x,retain_graph=True,create_graph=True)[0] for k in range(5)],1)
    je=float((ja-j).detach().abs().max())/max(1,float(ja.detach().abs().max()));qe=float((q-ind).detach().abs().max());assert max(fe,je,qe)<1e-10
    loss=q.square().mean()+j.square().mean();loss.backward();assert all(v.requires_grad and v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    assert prep['features'].shape==(len(xy),1000) and sum(v.numel() for v in model.parameters())==110705
    return dict(feature_derivative_relative_error=fe,output_jacobian_relative_error=je,output_difference=qe,parameters=110705,features=1000,finite_parameter_gradients=True)
def main():
    assert not B.exists();p=read(P);g=read(R/'inputs/cavity_geometries.json');rule=read(R/'protocols/R2_P1_geometry.json');torch.set_num_threads(2);B.mkdir()
    source=[Path(__file__),R/'code/shared_geometry_budget.py',R/'code/shared_geometry_features.py',R/'code/cavity_cover.py',R/'code/p2_components.py',R/'code/geometry_primitives.py',P,R/'protocols/R2_P1_geometry.json',R/'inputs/cavity_geometries.json']
    hashes={str(f.resolve()):sha(f) for f in source};out=dict(protocol_sha256=sha(P),source_sha256=hashes,original={},geometry_only={},odd_fixture={},fem_access=False,training_runs=0,formal_timing=False)
    for case in p['original_geometries']:
        domain,_,_=normalized_domain(g[case]);cover=construct(domain,rule);c,h=cover['centres'],cover['halfwidths']
        with np.load(R/f'results/R2_P2D_shared_features/{case}_covers.npz') as z:
            assert np.array_equal(c,z['centres']) and np.array_equal(h,z['halfwidths']);u=(z['uniform_centres'],z['uniform_halfwidths'])
            results={};xy=domain.random_interior(7,np.random.default_rng(442))
            for method in ['geometry_rbf_fourier','uniform_rbf_fourier']:
                a=make_shared_model(method,p['seed'],c,h,u);b=make_budget_compatible(method,p['seed'],c,h,u);assert set(a.state_dict())==set(b.state_dict()) and all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items());assert torch.equal(a(torch.tensor(xy)),b(torch.tensor(xy)));results[method]=dict(bitwise_unchanged=True,**check(b,xy))
        out['original'][case]=dict(K=len(c),checks=results);print('ORIGINAL_UNCHANGED',case,len(c),flush=True)
    for v in p['additional_geometry_only_cases']:
        geo=copy.deepcopy(g['C1']);geo['holes']=[dict(kind='ellipse',tag='cavity',center=v['center'],axes=v['axes'],angle=v['angle'])];domain,_,_=normalized_domain(geo);cover=construct(domain,rule);c,h=cover['centres'],cover['halfwidths'];model=make_budget_compatible('geometry_rbf_fourier',p['seed'],c,h)
        changed=copy.deepcopy(geo);factor=7.3;shift=np.array([1.2,-2.4]);changed['outer']=(np.asarray(changed['outer'])*factor+shift).tolist();hole=changed['holes'][0];hole['center']=(np.array(hole['center'])*factor+shift).tolist();hole['axes']=(np.array(hole['axes'])*factor).tolist();nd,_,_=normalized_domain(changed);cv=construct(nd,rule)
        assert c.shape==cv['centres'].shape;dev=max(float(abs(c-cv['centres']).max()),float(abs(h-cv['halfwidths']).max()));assert dev<=1e-12
        np.savez_compressed(B/f"{v['name']}_geometry.npz",centres=c,widths=h,boundary=cover['boundary'],target=cover['target']);out['geometry_only'][v['name']]=dict(input=geo,K=len(c),normalization_equivariance_error=dev,**check(model,domain.random_interior(7,np.random.default_rng(443))))
        print('GEOMETRY_ONLY_PASS',v['name'],len(c),flush=True)
    with np.load(R/'results/R2_P2D_shared_features/T1_covers.npz') as z:c,h=z['centres'][:245],z['halfwidths'][:245]
    legacy_rejected=False
    try:make_shared_model('geometry_rbf_fourier',p['seed'],c,h)
    except AssertionError:legacy_rejected=True
    assert legacy_rejected;fixed=make_budget_compatible('geometry_rbf_fourier',p['seed'],c,h);d,_,_=normalized_domain(g['T1']);out['odd_fixture']=dict(K=245,legacy_rejected=True,new_constructed=True,**check(fixed,d.random_interior(7,np.random.default_rng(444))))
    for k in [0,1000,1001]:
        try:make_budget_compatible('geometry_rbf_fourier',p['seed'],np.zeros((k,2)),np.ones((k,2)))
        except ValueError:pass
        else:raise AssertionError(('budget must reject',k))
    out['explicit_budget_rejections']=[0,1000,1001];out['passed']=True
    for f,h in hashes.items():assert sha(f)==h
    write(B/'audit.json',out)
    lines=['# P2M: geometry representation and compatibility audit','',
       'This is a CPU-only construction and differential-consistency audit. It adds no trained solution, FEM evaluation, time comparison or PDE generalization result. All frozen scientific code and results are unchanged.','',
       'The frozen shared feature constructor requires an even Gaussian count because it assumes complete sine/cosine pairs. The new wrapper delegates every even-count model to that exact constructor. For odd K it uses M=1000-K global channels: ceil(M/2) sine and floor(M/2) cosine terms from the same deterministic frequency stream, followed by K Gaussian terms. No zero padding, trainable-layer enlargement, feature deletion or per-shape manual choice occurs. All110705 network parameters remain trainable.','',
       '| Input | Gaussian count | Scope | Result |','|---|---:|---|---|']
    for k,v in out['original'].items():lines.append(f"| {k} | {v['K']} | Frozen geometry; both representations | Exact state/output preservation; derivative/gradient checks pass |")
    for k,v in out['geometry_only'].items():lines.append(f"| {k} | {v['K']} | New geometry construction only | Derivatives and normalized scaling/translation check pass |")
    lines+=['| Odd algebraic fixture | 245 | First245 frozen T1 features, not a new PDE | Old constructor rejects; wrapper passes |','',
       'The feature map remains a single global network with fixed, unnormalized Gaussian inputs of infinite support. It is not the compact normalized window/subnetwork scheme used in earlier unsuccessful candidates. Geometry changes positions, widths and the local/global feature allocation before training. This is geometry adaptation, not residual-driven online enrichment.','',
       'The finite-budget applicability condition remains1<=K<1000. Empty or over-budget covers are explicitly rejected. No arbitrary-complexity or rotation-equivariance guarantee is established. Geometry normalization gives translation/isotropic-scale invariance of the normalized representation for nondegenerate decisions; geometric tolerances and refinement thresholds can matter near decision boundaries.','',
       'All previous accuracy and timing evidence uses even K (244/360/376/460) and is unchanged. Odd-count compatibility has numerical validation only, with no new accuracy claim. The two new offset shapes also carry no FEM accuracy claim. The synthetic odd fixture is identified separately to avoid counting it as independent geometry validation.','',
       'Raw arrays and exact errors: `results/R2_P2M_representation_audit`; protocol: `protocols/R2_P2M_representation_audit.json`; implementation: `code/shared_geometry_budget.py`; audit: `code/audit_p2m_representation.py`.','']
    report=R/'P2M_Representation_Audit.md';report.write_text('\n'.join(lines),encoding='utf-8')
    write(B/'completion.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),scientific_training_runs=0,formal_timing=False,files_sha256={f.name:sha(f) for f in B.iterdir() if f.is_file()},supporting_files_sha256=dict(hashes,**{str(report.resolve()):sha(report)})));print('P2M_CLOSED',flush=True)
if __name__=='__main__':main()
