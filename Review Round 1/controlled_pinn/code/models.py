"""Controlled stage-3 models. Legacy DEM/XPINN definitions are read unchanged."""
import runtime
from runtime import torch, DEVICE, DTYPE
from pathlib import Path
import importlib.util, sys, json, math, hashlib
import numpy as np
from torch import nn
from matplotlib.path import Path as Polygon

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
legacy_path=PROJECT/'代码源文件/审稿修改-基线对比/benchmark_suite.py'
spec=importlib.util.spec_from_file_location('tust_legacy_circle',legacy_path)
legacy=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=legacy
spec.loader.exec_module(legacy)
GEOMETRY=json.loads((ROOT/'config/geometry.json').read_text())
POLYGON=np.asarray(GEOMETRY['tunnel_vertices_normalized'])

def generator(seed,stream):
    return torch.Generator().manual_seed(stream+seed)

def shared_init(seed,constructor):
    # CPU-only initialization stream, restored after use.
    state=torch.random.get_rng_state()
    torch.random.default_generator.manual_seed(200000+seed)
    result=constructor()
    torch.random.set_rng_state(state)
    return result

class FixedFeatures(nn.Module):
    def __init__(self,method,seed,width):
        super().__init__()
        self.method=method
        if method.startswith('fourier'):
            factor={'fourier':1.,'fourier_half':.5,'fourier_double':2.}[method]
            B=torch.randn((500,2),generator=generator(seed,300000))*20/(2*math.pi)*factor
            self.register_buffer('B',B)
        else:
            W=torch.rand((1000,2),generator=generator(seed,300000))*40-20
            C=torch.rand((1000,2),generator=generator(seed,400000))-.5
            if method=='anchored':b=-(W*C).sum(1)
            elif method=='independent_gaussian':
                b=torch.randn(1000,generator=generator(seed,500000))*math.sqrt(2*20**2*.5**2/9)
            elif method=='independent_marginal':
                wp=torch.rand((1000,2),generator=generator(seed,500000))*40-20
                cp=torch.rand((1000,2),generator=generator(seed,600000))-.5
                b=-(wp*cp).sum(1)
            else:raise ValueError(method)
            self.register_buffer('W',W);self.register_buffer('b',b)
        self.net=shared_init(seed,lambda:legacy.mlp(1000,[width,width],5))
    def features(self,x):
        if hasattr(self,'B'):
            phase=2*math.pi*(x@self.B.T)
            return torch.cat([torch.sin(phase),torch.cos(phase)],1)
        return torch.tanh(x@self.W.T+self.b)
    def forward(self,x):return self.net(self.features(x))

def build_model(method,seed,geometry='circle'):
    if method in ('anchored','independent_gaussian','independent_marginal') or method.startswith('fourier'):
        model=FixedFeatures(method,seed,100 if geometry=='circle' else 128)
    elif method=='vanilla':model=shared_init(seed,legacy.VanillaMixed)
    elif method=='vanilla_matched':
        width=233 if geometry=='circle' else 267
        model=shared_init(seed,lambda:legacy.mlp(2,[width]*3,5))
    elif method=='xpinn':model=shared_init(seed,legacy.XPINNMixed)
    elif method=='dem':model=shared_init(seed,legacy.DEMNet)
    else:raise ValueError(method)
    return model.to(DEVICE)

def ellipse_points(n):
    theta=np.linspace(0,2*np.pi,n)
    ca,sa=np.cos(-np.pi/4),np.sin(-np.pi/4)
    rot=np.array([[ca,-sa],[sa,ca]])
    xy=np.column_stack([.1*np.cos(theta),.05*np.sin(theta)])@rot.T+.2
    local=(xy-.2)@rot
    normal=-(local/np.array([.1**2,.05**2]))@rot.T
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    return xy,normal

def in_rock(points,geometry):
    points=np.asarray(points)
    inside=(np.abs(points)<.5+1e-12).all(1)
    if geometry=='circle':return inside & (np.linalg.norm(points,axis=1)>.1)
    ca,sa=np.cos(-np.pi/4),np.sin(-np.pi/4)
    local=(points-.2)@np.array([[ca,-sa],[sa,ca]])
    ellipse=(local[:,0]/.1)**2+(local[:,1]/.05)**2<=1
    return inside & ~Polygon(POLYGON).contains_points(points) & ~ellipse

def tunnel_samples(seed,n_domain=8000,n_boundary=500):
    rng=generator(seed,100000)
    parts=[]
    budgets=[int(.8*n_domain),int(.1*n_domain)]
    budgets.append(n_domain-sum(budgets))
    for kind,n in enumerate(budgets):
        accepted=[];remaining=n
        while remaining:
            batchsize=max(remaining*4,128)
            if kind==0:xy=torch.rand((batchsize,2),generator=rng)-.5
            elif kind==1:xy=torch.randn((batchsize,2),generator=rng)*.08
            else:xy=torch.randn((batchsize,2),generator=rng)*.04+.2
            # Preserve original candidate clamping; exact per-stratum truncation fixes overshoot.
            xy=xy.clamp(-.5,.5)
            xy=xy[torch.as_tensor(in_rock(xy.numpy(),'tunnel'))][:remaining]
            accepted.append(xy);remaining-=len(xy)
        parts.append(torch.cat(accepted))
    domain=torch.cat(parts)
    domain=domain[torch.randperm(n_domain,generator=rng)]
    data={'domain':domain}
    for key,axis,value in [('left',0,-.5),('right',0,.5),('top',1,.5),('bottom',1,-.5)]:
        xy=torch.rand((n_boundary,2),generator=rng)-.5
        xy[:,axis]=value;data[key]=xy
    # The submitted geometry includes its repeated closing vertex; keep its boundary weighting.
    nodes=POLYGON[:-1]
    tangent=np.roll(nodes,-1,0)-np.roll(nodes,1,0)
    normal=np.column_stack([tangent[:,1],-tangent[:,0]])
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    normal=np.vstack([normal,normal[0]])
    data['hole']=torch.tensor(POLYGON,dtype=DTYPE)
    data['hole_normals']=torch.tensor(normal,dtype=DTYPE)
    xy,n=ellipse_points(200)
    data['ellipse']=torch.tensor(xy,dtype=DTYPE)
    data['ellipse_normals']=torch.tensor(n,dtype=DTYPE)
    return {k:v.to(DEVICE) for k,v in data.items()}

def samples_for(seed,geometry,n_domain=None,n_boundary=500):
    if geometry=='tunnel':return tunnel_samples(seed,n_domain or 8000,n_boundary)
    sample=legacy.generate_samples(n_domain or 6000,n_boundary,seed)
    return {k:getattr(sample,k) for k in sample.__dataclass_fields__}

def parts_loss(model,s,p,pt,geometry='circle',p_water=-1.):
    E,nu=(1.333,.3333) if geometry=='circle' else (1.,.26)
    lam=E*nu/((1+nu)*(1-2*nu));mu=E/(2*(1+nu))
    xy=s['domain'].detach().clone().requires_grad_(True)
    out=model(xy)
    grad=lambda i:torch.autograd.grad(out[:,i].sum(),xy,create_graph=True)[0]
    gu,gv=grad(0),grad(1)
    tr=gu[:,0]+gv[:,1]
    const=((out[:,2]-lam*tr-2*mu*gu[:,0]).square()
          +(out[:,3]-lam*tr-2*mu*gv[:,1]).square()
          +(out[:,4]-mu*(gu[:,1]+gv[:,0])).square()).mean()
    gxx,gyy,gxy=grad(2),grad(3),grad(4)
    eq=(gxx[:,0]+gxy[:,1]).square().mean()+(gxy[:,0]+gyy[:,1]).square().mean()
    ol,orr,ot,ob,oh=[model(s[key]) for key in ('left','right','top','bottom','hole')]
    traction=((ol[:,2]-p).square()+ol[:,4].square()).mean()
    traction+=((orr[:,2]-p).square()+orr[:,4].square()).mean()
    traction+=((ot[:,3]-pt).square()+ot[:,4].square()).mean()+ob[:,4].square().mean()
    normal=-s['hole']/.1 if geometry=='circle' else s['hole_normals']
    nx,ny=normal[:,0],normal[:,1]
    traction+=((oh[:,2]*nx+oh[:,4]*ny).square()+(oh[:,4]*nx+oh[:,3]*ny).square()).mean()
    if geometry=='tunnel':
        oe=model(s['ellipse']);en=s['ellipse_normals'];nx,ny=en[:,0],en[:,1]
        tn=oe[:,2]*nx.square()+oe[:,3]*ny.square()+2*nx*ny*oe[:,4]
        tt=(oe[:,3]-oe[:,2])*nx*ny+(nx.square()-ny.square())*oe[:,4]
        traction+=((tn-p_water).square()+tt.square()).mean()
    anchor=torch.tensor([[0.,-.5]],device=DEVICE,dtype=DTYPE)
    disp=ob[:,1].square().mean()+model(anchor)[0,0].square()
    result={'equilibrium':eq,'constitutive':const,'traction':traction,'displacement':disp}
    if isinstance(model,legacy.XPINNMixed):
        interface=eq.new_zeros(())
        for key,a,b in [('interface_v_low',0,1),('interface_v_high',2,3),
                        ('interface_h_left',0,2),('interface_h_right',1,3)]:
            interface+=(model.forward_subdomain(a,s[key])-model.forward_subdomain(b,s[key])).square().mean()
        result['interface']=interface
    return result

WEIGHTS={'equilibrium':1.,'constitutive':1.,'traction':10.,'displacement':100.,'interface':10.}

def objective(model,s,config):
    p,pt=config['p_lateral'],config['p_top']
    if config['method']=='dem':return legacy.dem_loss(model,legacy.SampleSet(**s),p,pt)
    parts=parts_loss(model,s,p,pt,config['geometry'],config.get('p_water',-1.))
    return sum(WEIGHTS[k]*v for k,v in parts.items())

def array_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
