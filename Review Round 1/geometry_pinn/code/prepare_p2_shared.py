"""Construction and derivative audit only; no optimizer or FEM access."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import copy, hashlib, json, shutil
import numpy as np
import torch
from cavity_cover import normalized_domain
from p2_components import shared_points, uniform_cover, prepare_loss, parts_loss, objective
from prepare_p2 import independent_loss
from shared_geometry_features import make_shared_model, uniform_centres

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'results/R2_P2D_shared_features'
PROTOCOL=ROOT/'protocols/R2_P2D_shared_features.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d): Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    assert not (BATCH/'preflight.json').exists() and not (BATCH/'manifest.json').exists(), 'Never overwrite a prepared or started batch'
    BATCH.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    geometries=json.loads((ROOT/'inputs/cavity_geometries.json').read_text(encoding='utf-8'))
    report=dict(passed=False,optimization_runs=0,fem_read=False,geometry={},numeric={},protocol_sha256=sha(PROTOCOL))
    inputs=[PROTOCOL,ROOT/'inputs/cavity_geometries.json']
    for geometry in ['C1','T1','S1','L1']:
        path=ROOT/f'results/R2_P1/{geometry}_cover.npz';inputs.append(path)
        with np.load(path) as z: c,h=z['centres'],z['halfwidths']
        domain,origin,length=normalized_domain(geometries[geometry])
        transformed=copy.deepcopy(geometries[geometry]); scale=7.3; shift=np.array([2.1,-3.7])
        transformed['outer']=(np.asarray(transformed['outer'])*scale+shift).tolist()
        for hole in transformed['holes']:
            if hole['kind']=='polygon': hole['vertices']=(np.asarray(hole['vertices'])*scale+shift).tolist()
            else:
                hole['center']=(np.asarray(hole['center'])*scale+shift).tolist()
                hole['axes']=(np.asarray(hole['axes'])*scale).tolist()
        _,o2,l2=normalized_domain(transformed)
        physical=c*length+origin
        invariant_error=float(np.max(np.abs((physical*scale+shift-o2)/l2-c)))
        assert invariant_error<1e-14
        uc,uh=uniform_centres(len(c),domain)
        np.savez_compressed(BATCH/f'{geometry}_covers.npz',centres=c,halfwidths=h,uniform_centres=uc,uniform_halfwidths=uh)
        if geometry in ['C1','L1']:
            src=ROOT/f'results/R2_P2A/{geometry}_points.npz';inputs.append(src)
            shutil.copyfile(src,BATCH/src.name)
            with np.load(src) as z: points={k:z[k] for k in z.files}
        else:
            pc,ph,_=uniform_cover(domain,len(c))
            points=shared_points(domain,[(c,h),(pc,ph)],protocol['seed'])
            np.savez_compressed(BATCH/f'{geometry}_points.npz',**points)
        small={k:v.copy() for k,v in points.items()}
        for key in ['domain','left','right','top','bottom','hole','normal']: small[key]=small[key][:7]
        small['domain_weight']=np.full(7,1/7);small['hole_weight']=np.full(7,1/7)
        responses={}
        for label,centres,widths in [('geometry',c,h),('uniform',uc,uh)]:
            rbf=np.exp(-.5*(((points['domain'][:,None]-centres[None])/widths[None])**2).sum(2))
            responses[label]={'min_max_response':float(rbf.max(0).min()),'columns_max_below_0p1':int((rbf.max(0)<.1).sum())}
            assert responses[label]['columns_max_below_0p1']==0
        report['geometry'][geometry]=dict(gaussians=len(c),global_features=1000-len(c),normalization_error=invariant_error,responses=responses)
        initial=None
        for method in protocol['methods']:
            model=make_shared_model(method,protocol['seed'],c,h,(uc,uh))
            count=sum(p.numel() for p in model.parameters());assert count==110705
            values=[p.detach().clone() for p in model.net.parameters()]
            if initial is None: initial=values
            else: assert all(torch.equal(a,b) for a,b in zip(initial,values))
            prepared=prepare_loss(model,small)
            assert prepared['domain']['features'].shape==(7,1000)
            x=torch.tensor(small['domain'],dtype=torch.float64,requires_grad=True)
            out=model(x)
            jac=torch.stack([torch.autograd.grad(out[:,i].sum(),x,retain_graph=True)[0] for i in range(5)],1)
            fast,dj=model.sparse(prepared['domain'])
            field_error=float((fast-out).abs().max().detach())
            jac_error=float((dj-jac).abs().max().detach())/max(1.,float(jac.abs().max()))
            analytic=objective(parts_loss(model,prepared,-1.,-5.))
            automatic=independent_loss(model,small,-1.,-5.)
            ga=torch.autograd.grad(analytic,tuple(model.parameters()))
            gb=torch.autograd.grad(automatic,tuple(model.parameters()))
            loss_error=abs(analytic.item()-automatic.item())/max(1.,abs(automatic.item()))
            gradient_error=max(float((a-b).abs().max()) for a,b in zip(ga,gb))/max(1.,max(float(b.abs().max()) for b in gb))
            assert max(field_error,jac_error,loss_error,gradient_error)<1e-9
            model=model.float().cuda();full=objective(parts_loss(model,prepare_loss(model,points),-1.,-5.));full.backward()
            assert torch.isfinite(full) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            gf=objective(parts_loss(model,prepare_loss(model,small),-1.,-5.));gd=independent_loss(model,small,-1.,-5.)
            gpu_error=abs(gf.item()-gd.item())/max(1.,abs(gd.item()));assert gpu_error<2e-5
            report['numeric'][geometry+'_'+method]=dict(parameters=count,field_absolute=field_error,jacobian_relative=jac_error,
                objective_relative=loss_error,gradient_relative=gradient_error,gpu_objective_relative=gpu_error,full_gpu_gradients_finite=True)
            print(geometry,method,'PASS',flush=True)
            del model,prepared,out,jac,fast,dj,full,gf,gd,ga,gb,analytic,automatic
            torch.cuda.empty_cache()
    report['inputs_sha256']={str(f.resolve()):sha(f) for f in inputs}
    sources=['shared_geometry_features.py','prepare_p2_shared.py','prepare_p2.py','p2_components.py','p2_observed_lbfgs.py',
             'sparse_mixed_pinn.py','cavity_cover.py','geometry_primitives.py','support_probes.py']
    report['source_sha256']={name:sha(ROOT/'code'/name) for name in sources}
    report['prepared_sha256']={f.name:sha(f) for f in BATCH.glob('*.npz')}
    report['passed']=True
    write(BATCH/'preflight.json',report)
    print('ALL_FOUR_GEOMETRIES_PASS_NO_TRAINING_NO_FEM',flush=True)

if __name__=='__main__': main()
