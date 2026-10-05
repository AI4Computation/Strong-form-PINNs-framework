"""Original DEM architecture with exact spatial Jacobians and audited quadrature."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['MKL_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.pop('KMP_DUPLICATE_LIB_OK',None)
import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from slice_quadrature import SliceQuadrature
from geometry import Domain

ROOT=Path(__file__).resolve().parents[2]
LEGACY=ROOT/'code/benchmark/benchmark_suite.py'
spec=importlib.util.spec_from_file_location('audited_dem_legacy',LEGACY)
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
spec=importlib.util.spec_from_file_location('audited_dem_lbfgs',ROOT/'code/observed_lbfgs.py')
optimizer_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(optimizer_module)
ObservedLBFGS=optimizer_module.ObservedLBFGS
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def utc():return datetime.now(timezone.utc).isoformat()
def snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def build(seed,device='cuda'):
    assert torch.get_default_dtype()==torch.float32
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(200000+seed)
        model=legacy.DEMNet()
    return model.double().to(device)


def field(model,xy):
    """Same displacement map as legacy.DEMNet; analytic coordinate derivatives."""
    h=xy;gx=torch.zeros_like(xy);gy=torch.zeros_like(xy);gx[:,0]=1.;gy[:,1]=1.
    for layer in model.net:
        if isinstance(layer,torch.nn.Linear):
            h=layer(h);gx=gx@layer.weight.T;gy=gy@layer.weight.T
        elif isinstance(layer,torch.nn.Tanh):
            h=torch.tanh(h);d=1-h.square();gx=gx*d;gy=gy*d
        else:raise TypeError(type(layer))
    gauge=model.raw(xy.new_tensor([[0.,-.5]]))[0,0]
    distance=xy[:,1]+.5
    uv=torch.stack([h[:,0]-gauge,distance*h[:,1]],1)
    exx=gx[:,0];eyy=h[:,1]+distance*gy[:,1];exy=.5*(gy[:,0]+distance*gx[:,1])
    trace=exx+eyy
    stress=torch.stack([legacy.LAM*trace+2*legacy.MU*exx,legacy.LAM*trace+2*legacy.MU*eyy,2*legacy.MU*exy],1)
    density=.5*legacy.LAM*trace.square()+legacy.MU*(exx.square()+eyy.square()+2*exy.square())
    return torch.cat([uv,stress],1),density


def rule(base,order=5):
    domain=Domain([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]],
        [dict(kind='ellipse',tag='cavity',center=[0.,0.],axes=[.1,.1])],
        ['bottom','right','top','left'])
    sample=SliceQuadrature(domain,base_depth=base,order=order).points()
    nodes,w=np.polynomial.legendre.leggauss(order)
    n=2**base
    t=(-.5+(np.arange(n)[:,None]+(nodes+1)/2)/n).ravel();mass=np.tile(w/(2*n),n)
    boundary=np.vstack([np.column_stack([np.full(len(t),-.5),t]),np.column_stack([np.full(len(t),.5),t]),np.column_stack([t,np.full(len(t),.5)])])
    sample.update(boundary=boundary,boundary_weights=mass,base_depth=base,order=order)
    return sample


def energy(model,sample,p,pt,backward=False,chunk=4096):
    device=next(model.parameters()).device
    if backward:model.zero_grad(set_to_none=True)
    internal=0.
    with torch.enable_grad() if backward else torch.no_grad():
        for start in range(0,len(sample['xy']),chunk):
            xy=torch.as_tensor(sample['xy'][start:start+chunk],dtype=torch.float64,device=device)
            w=torch.as_tensor(sample['w'][start:start+chunk],dtype=torch.float64,device=device)
            _,density=field(model,xy);value=(w*density).sum()
            internal+=float(value.detach())
            if backward:value.backward()
        xy=torch.as_tensor(sample['boundary'],dtype=torch.float64,device=device)
        uv=model(xy);w=torch.as_tensor(sample['boundary_weights'],dtype=torch.float64,device=device);n=len(w)
        work=(w*(-p*uv[:n,0]+p*uv[n:2*n,0]+pt*uv[2*n:,1])).sum()
        if backward:(-work).backward()
        external=float(work.detach())
    result=dict(internal_energy=internal,external_work=external,potential_energy=internal-external)
    if backward:
        result['gradient']=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()]).detach().cpu().numpy().copy()
    return result


def numerical_check(model,fit,validation,final,config,protocol,previous_energy=None):
    p,pt=config['p_lateral'],config['p_top']
    a=energy(model,fit,p,pt,True);b=energy(model,validation,p,pt,True);c=energy(model,final,p,pt)
    scale=max(abs(c['internal_energy']),abs(c['external_work']),1.)
    difference=max(abs(a['internal_energy']-c['internal_energy']),abs(b['internal_energy']-c['internal_energy']),
                   abs(a['external_work']-c['external_work']),abs(b['external_work']-c['external_work']))
    energy_tolerance=protocol['energy_component_relative_tolerance']*scale
    gradient_difference=float(np.linalg.norm(a['gradient']-b['gradient']))
    gradient_scale=max(float(np.linalg.norm(b['gradient'])),protocol['gradient_scale_floor']*scale)
    gradient_relative=gradient_difference/gradient_scale
    nonincrease=previous_energy is None or c['potential_energy']<=previous_energy+energy_tolerance
    passed=bool(np.isfinite(difference) and np.isfinite(gradient_relative) and difference<=energy_tolerance and
        gradient_relative<=protocol['gradient_relative_tolerance'] and nonincrease)
    for record in [a,b]:record.pop('gradient')
    return dict(passed=passed,fit=a,validation=b,final=c,component_difference=difference,component_tolerance=energy_tolerance,
        gradient_relative_difference=gradient_relative,gradient_difference_norm=gradient_difference,
        gradient_scale=gradient_scale,validation_energy_nonincrease=bool(nonincrease))


def predict(model,xy,chunk=4096):
    out=[];device=next(model.parameters()).device
    with torch.no_grad():
        for start in range(0,len(xy),chunk):
            value,_=field(model,torch.as_tensor(xy[start:start+chunk],dtype=torch.float64,device=device))
            out.append(value.cpu().numpy())
    return np.concatenate(out)
