"""Create common points and controls, then independently preflight before fitting."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from cavity_cover import normalized_domain,windows
from p2_components import *
from p2_observed_lbfgs import ObservedLBFGS

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'results/R2_P2A'
PROTOCOL=ROOT/'protocols/R2_P2A_development.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def independent_loss(model,points,lateral,top):
    ref=next(model.parameters())
    x=torch.tensor(points['domain'],dtype=ref.dtype,device=ref.device,requires_grad=True)
    call=model.dense if isinstance(model,SparseMixedPINN) else model.forward
    q=call(x)
    j=torch.stack([torch.autograd.grad(q[:,i].sum(),x,create_graph=True,retain_graph=True)[0] for i in range(5)],dim=1)
    r=residual(q,j);w=torch.tensor(points['domain_weight'],dtype=ref.dtype,device=ref.device)
    result=(w*r.square().sum(1)).sum()
    out={k:call(torch.tensor(points[k],dtype=ref.dtype,device=ref.device)) for k in ['left','right','top','bottom','hole','gauge']}
    t=0
    for key,component,target in [('left',2,lateral),('right',2,lateral),('top',3,top)]:
        t+=((out[key][:,component]-target).square()+out[key][:,4].square()).mean()
    t+=out['bottom'][:,4].square().mean()
    n=torch.tensor(points['normal'],dtype=ref.dtype,device=ref.device)
    h=out['hole'];traction=torch.stack([h[:,2]*n[:,0]+h[:,4]*n[:,1],h[:,4]*n[:,0]+h[:,3]*n[:,1]],1)
    hw=torch.tensor(points['hole_weight'],dtype=ref.dtype,device=ref.device)
    t+=(hw*traction.square().sum(1)).sum()
    result+=10*t+100*(out['bottom'][:,1].square().mean()+out['gauge'][0,0].square())
    return result


def main():
    assert not BATCH.exists(),'Do not overwrite a prepared batch.'
    BATCH.mkdir()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    geometries=json.loads((ROOT/'inputs/cavity_geometries.json').read_text(encoding='utf-8'))
    report={'protocol_sha256':sha(PROTOCOL),'geometry':{},'numeric':{},'passed':False,'optimization_runs':0}
    input_files=[PROTOCOL,ROOT/'inputs/cavity_geometries.json']
    for geometry in ['C1','L1']:
        source=ROOT/f'results/R2_P1/{geometry}_cover.npz';input_files.append(source)
        with np.load(source) as z:c,h=z['centres'],z['halfwidths']
        domain,_,_=normalized_domain(geometries[geometry])
        uc,uh,n=uniform_cover(domain,len(c))
        covers=[(c,h),(uc,uh)]
        points=shared_points(domain,covers,p['seed'])
        np.savez_compressed(BATCH/f'{geometry}_points.npz',**points)
        np.savez_compressed(BATCH/f'{geometry}_covers.npz',centres=c,halfwidths=h,uniform_centres=uc,uniform_halfwidths=uh)
        audit={}
        for method,cover in zip(['geometry_local','uniform_local'],covers):
            w,tot,overlap=windows(points['domain'],*cover)
            audit[method]={'patches':len(cover[0]),'inactive_patches':int(np.sum(~np.any(w>0,axis=0))),
                           'maximum_overlap':int(overlap.max()),'minimum_raw_window_sum':float(tot.min())}
            assert audit[method]['inactive_patches']==0
        report['geometry'][geometry]=dict(uniform_grid=n,probes=int(points['probe_count']),uniform=int(points['uniform_count']),representations=audit)
        small={k:v.copy() for k,v in points.items()}
        for key in ['domain','left','right','top','bottom','hole','normal']:
            small[key]=small[key][:7]
        small['domain_weight']=np.full(7,1/7);small['hole_weight']=np.full(7,1/7)
        # All representations share actual W/downstream initialization where applicable.
        a=make_model('anchored',p['seed'],covers);b=make_model('independent_marginal',p['seed'],covers)
        assert torch.equal(a.W,b.W)
        assert all(torch.equal(x,y) for x,y in zip(a.net.parameters(),b.net.parameters()))
        del a,b
        for method in p['methods']:
            model=make_model(method,p['seed'],covers)
            count=sum(v.numel() for v in model.parameters())
            assert 0<=110705-count<=111
            prepared=prepare_loss(model,small)
            analytic=objective(parts_loss(model,prepared,-1.,-5.))
            automatic=independent_loss(model,small,-1.,-5.)
            ga=torch.autograd.grad(analytic,tuple(model.parameters()))
            gb=torch.autograd.grad(automatic,tuple(model.parameters()))
            loss_error=float(abs(analytic.item()-automatic.item())/max(1,abs(automatic.item())))
            gradient_error=max(float((x-y).abs().max()) for x,y in zip(ga,gb))/max(1,max(float(v.abs().max()) for v in gb))
            assert max(loss_error,gradient_error)<1e-9,(method,loss_error,gradient_error)
            model=model.float().cuda();prepared=prepare_loss(model,points)
            full=objective(parts_loss(model,prepared,-1.,-5.));full.backward()
            assert torch.isfinite(full) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
            # GPU float32 small objective against full-autograd independent evaluation.
            gs=objective(parts_loss(model,prepare_loss(model,small),-1.,-5.))
            gd=independent_loss(model,small,-1.,-5.)
            gpu_error=float(abs(gs.item()-gd.item())/max(1,abs(gd.item())))
            assert gpu_error<2e-5,(method,gpu_error)
            report['numeric'][geometry+'_'+method]=dict(parameters=count,cpu_loss_relative=loss_error,cpu_parameter_gradient_relative=gradient_error,
                                                       gpu_loss_relative=gpu_error,gpu_full_loss=float(full.detach()),gpu_gradients_finite=True)
            print(geometry,method,'preflight passed',flush=True)
            del model,prepared,full,gs,gd,ga,gb,analytic,automatic
            torch.cuda.empty_cache()
    # Instrumentation must not change the installed optimizer's numerical result.
    endpoints=[]
    for cls in [torch.optim.LBFGS,ObservedLBFGS]:
        x=torch.nn.Parameter(torch.tensor([3.,-4.,1.],dtype=torch.float64))
        opt=cls([x],max_iter=20,max_eval=30,line_search_fn='strong_wolfe',tolerance_change=1e-12)
        def closure():
            opt.zero_grad();loss=((x-torch.tensor([1.,2.,3.]))**2*torch.tensor([1.,2.,5.])).sum();loss.backward();return loss
        opt.step(closure);endpoints.append(x.detach().clone())
    assert torch.equal(*endpoints)
    report['optimizer_exact_quadratic_match']=True
    report['inputs_sha256']={str(f.resolve()):sha(f) for f in input_files}
    code_names=['prepare_p2.py','p2_components.py','p2_observed_lbfgs.py','sparse_mixed_pinn.py','cavity_cover.py','geometry_primitives.py','support_probes.py']
    report['source_sha256']={name:sha(ROOT/'code'/name) for name in code_names}
    report['prepared_sha256']={f.name:sha(f) for f in BATCH.glob('*.npz')}
    report['passed']=True
    write(BATCH/'preflight.json',report)
    print('P2_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':main()
