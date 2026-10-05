"""Fresh paired sampling and numerical preflight, without training or FEM."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import torch
from cavity_cover import normalized_domain
from p2_components import shared_points,prepare_loss,parts_loss,objective
from prepare_p2 import independent_loss
from shared_geometry_features import make_shared_model
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2F_repetition_budget';P=ROOT/'protocols/R2_P2F_repetition_budget.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    assert not B.exists();B.mkdir()
    p=read(P);prior=ROOT/'results/R2_P2D_shared_features';closed=read(prior/'completion.json');pm=read(prior/'manifest.json')
    for name,h in pm['source_sha256'].items():assert sha(ROOT/'code'/name)==h
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    geom=read(ROOT/'inputs/cavity_geometries.json');inputs=[P,ROOT/'inputs/cavity_geometries.json',prior/'completion.json',prior/'manifest.json']
    report=dict(passed=False,optimization_runs=0,fem_read=False,protocol_sha256=sha(P),points={},numeric={})
    bc=read(ROOT/'results/R2_P2A/completion.json')
    for geometry in ['C1','L1']:
        path=prior/f'{geometry}_covers.npz';assert sha(path)==closed['files_sha256'][path.name]
        inputs.append(path);shutil.copyfile(path,B/path.name)
        with np.load(path) as z:c,h,uc,uh=z['centres'],z['halfwidths'],z['uniform_centres'],z['uniform_halfwidths']
        support=ROOT/f'results/R2_P2A/{geometry}_covers.npz';assert sha(support)==bc['batch_files_sha256'][support.name];inputs.append(support)
        with np.load(support) as z:covers=[(z['centres'],z['halfwidths']),(z['uniform_centres'],z['uniform_halfwidths'])]
        domain,_,_=normalized_domain(geom[geometry])
        for seed in p['seeds']:
            points=shared_points(domain,covers,seed);name=f'{geometry}_seed{seed}'
            assert domain.contains(points['domain']).all() and len(points['domain'])==6000
            assert abs(points['domain_weight'].sum()-1)<1e-12 and abs(points['hole_weight'].sum()-1)<1e-12
            assert np.max(abs(np.linalg.norm(points['normal'],axis=1)-1))<1e-10
            response={}
            for label,centres,widths in [('geometry',c,h),('uniform',uc,uh)]:
                maxima=np.exp(-.5*(((points['domain'][:,None]-centres[None])/widths[None])**2).sum(2)).max(0)
                response[label]=float(maxima.min());assert maxima.min()>.1
            np.savez_compressed(B/f'{name}_points.npz',**points)
            report['points'][name]=dict(uniform=int(points['uniform_count']),probes=int(points['probe_count']),minimum_max_response=response)
            small={k:v.copy() for k,v in points.items()}
            for key in ['domain','left','right','top','bottom','hole','normal']:small[key]=small[key][:9]
            small['domain_weight']=np.full(9,1/9);small['hole_weight']=np.full(9,1/9)
            initial=None;fullB=None
            for method in p['methods']:
                model=make_shared_model(method,seed,c,h,(uc,uh));assert sum(v.numel() for v in model.parameters())==110705
                weights=[v.detach().clone() for v in model.net.parameters()]
                if initial is None:initial=weights;fullB=model.B.clone()
                else:
                    assert all(torch.equal(x,y) for x,y in zip(initial,weights))
                    assert torch.equal(model.B,fullB[:len(model.B)])
                prepared=prepare_loss(model,small);fast=objective(parts_loss(model,prepared,-1.,-5.));auto=independent_loss(model,small,-1.,-5.)
                fg=torch.autograd.grad(fast,tuple(model.parameters()));ag=torch.autograd.grad(auto,tuple(model.parameters()))
                le=abs(fast.item()-auto.item())/max(1.,abs(auto.item()))
                ge=max(float((x-y).abs().max()) for x,y in zip(fg,ag))/max(1.,max(float(x.abs().max()) for x in ag))
                assert max(le,ge)<1e-9
                model=model.float().cuda();value=objective(parts_loss(model,prepare_loss(model,points),-1.,-5.));value.backward()
                assert torch.isfinite(value) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
                report['numeric'][name+'_'+method]=dict(parameters=110705,objective_relative=le,parameter_gradient_relative=ge,full_gpu_gradients_finite=True)
                print(name,method,'PASS',flush=True)
                del model,prepared,fast,auto,fg,ag,value;torch.cuda.empty_cache()
    sources=['prepare_p2f.py','shared_geometry_features.py','p2_components.py','prepare_p2.py','p2_observed_lbfgs.py','cavity_cover.py','geometry_primitives.py','support_probes.py','sparse_mixed_pinn.py']
    report['source_sha256']={name:sha(ROOT/'code'/name) for name in sources};report['inputs_sha256']={str(x.resolve()):sha(x) for x in inputs}
    report['prepared_sha256']={x.name:sha(x) for x in B.glob('*.npz')};report['passed']=True
    write(B/'preflight.json',report);print('18_PREFLIGHTS_PASS_NO_TRAINING_NO_FEM',flush=True)
if __name__=='__main__':main()
