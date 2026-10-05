"""Shared feature representations, physical objective and collocation rules."""
import math
import numpy as np
import torch
from torch import nn
from cavity_cover import windows
from support_probes import support_probes
from sparse_mixed_pinn import SparseMixedPINN, residual


def uniform_cover(domain, candidate_count, overlap=1.5):
    # Match the approximate patch count through geometry alone; no FEM search.
    n = int(np.floor(np.sqrt(candidate_count) + .5))
    centres, halfwidths = [], []
    for i in range(n):
        for j in range(n):
            lo = domain.lo + (domain.hi-domain.lo)*np.array([i,j])/n
            hi = lo + (domain.hi-domain.lo)/n
            centre = (lo+hi)/2
            if domain.contains(centre[None])[0] or any(c.cuts_box(lo,hi) for c in domain.curves):
                centres.append(centre)
                halfwidths.append((hi-lo)*overlap/2)
    return np.asarray(centres),np.asarray(halfwidths),n


def shared_points(domain, covers, seed, n_domain=6000, n_outer=500, n_hole=1000):
    probes = [support_probes(domain,c,h)[0] for c,h in covers]
    probes = np.unique(np.vstack(probes),axis=0)
    assert len(probes)<n_domain//3
    rng=np.random.default_rng(100000+seed)
    uniform=domain.random_interior(n_domain-len(probes),rng)
    xy=np.vstack([uniform,probes])
    weights=np.r_[np.full(len(uniform),.9/len(uniform)),np.full(len(probes),.1/len(probes))]
    result=dict(domain=xy,domain_weight=weights,uniform_count=np.array(len(uniform)),
                probe_count=np.array(len(probes)))
    for tag in ['left','right','top','bottom']:
        curves=[c for c in domain.curves if c.tag==tag and not c.hole]
        assert len(curves)==1
        x,normal,_,jac=curves[0].evaluate((np.arange(n_outer)+rng.random(n_outer))/n_outer)
        result[tag]=x
    holes=[c for c in domain.curves if c.hole]
    lengths=np.array([c.quadrature(32,5)['w'].sum() for c in holes])
    desired=n_hole*lengths/lengths.sum()
    counts=np.floor(desired).astype(int)
    counts[np.argsort(-(desired-counts),kind='stable')[:n_hole-counts.sum()]]+=1
    assert np.all(counts>0)
    x,n,w=[],[],[]
    for curve,count in zip(holes,counts):
        xy,normal,_,jac=curve.evaluate((np.arange(count)+rng.random(count))/count)
        x.append(xy);n.append(normal);w.append(jac/count)
    result['hole']=np.vstack(x);result['normal']=np.vstack(n)
    result['hole_weight']=np.concatenate(w)/sum(v.sum() for v in w)
    result['gauge']=np.array([[0.,-.5]])
    return result


class FeatureMixedPINN(nn.Module):
    """Original feature laws and initialization streams; explicit first derivatives."""
    def __init__(self,method,seed):
        super().__init__()
        self.method=method
        gen=lambda offset:torch.Generator().manual_seed(offset+seed)
        if method=='fourier_half':
            B=torch.randn(500,2,generator=gen(300000))*20/(2*math.pi)*.5
            self.register_buffer('B',B)
        else:
            W=torch.rand(1000,2,generator=gen(300000))*40-20
            C=torch.rand(1000,2,generator=gen(400000))-.5
            b=-(W*C).sum(1)
            if method=='independent_marginal':
                wp=torch.rand(1000,2,generator=gen(500000))*40-20
                cp=torch.rand(1000,2,generator=gen(600000))-.5
                b=-(wp*cp).sum(1)
            elif method!='anchored':raise ValueError(method)
            self.register_buffer('W',W);self.register_buffer('b',b)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(200000+seed)
            self.net=nn.Sequential(nn.Linear(1000,100),nn.Tanh(),nn.Linear(100,100),nn.Tanh(),nn.Linear(100,5))
        self.double()

    def features(self,x):
        if hasattr(self,'B'):
            phase=2*math.pi*x@self.B.T
            return torch.cat([torch.sin(phase),torch.cos(phase)],dim=1)
        return torch.tanh(x@self.W.T+self.b)

    def forward(self,x):return self.net(self.features(x))

    def prepare(self,xy):
        first=next(self.parameters())
        x=torch.as_tensor(xy,dtype=first.dtype,device=first.device)
        f=self.features(x)
        if hasattr(self,'B'):
            phase=2*math.pi*x@self.B.T
            df=2*math.pi*torch.cat([torch.cos(phase)[:,:,None]*self.B[None],
                                   -torch.sin(phase)[:,:,None]*self.B[None]],dim=1)
        else:df=(1-f*f)[:,:,None]*self.W[None]
        return dict(x=x,features=f,df=df)

    def sparse(self,p):
        h1=torch.tanh(self.net[0](p['features']))
        d1=(1-h1*h1)[:,:,None]*torch.einsum('of,nfd->nod',self.net[0].weight,p['df'])
        h2=torch.tanh(self.net[2](h1))
        d2=(1-h2*h2)[:,:,None]*torch.einsum('of,nfd->nod',self.net[2].weight,d1)
        return self.net[4](h2),torch.einsum('of,nfd->nod',self.net[4].weight,d2)


def make_model(method,seed,covers,device='cpu'):
    if method in ['geometry_local','uniform_local']:
        c,h=covers[0 if method=='geometry_local' else 1]
        model=SparseMixedPINN(c,h,seed=seed)
    else:model=FeatureMixedPINN(method,seed)
    return model.to(device)


def prepare_loss(model,points):
    p={key:model.prepare(points[key]) for key in ['domain','left','right','top','bottom','hole','gauge']}
    ref=next(model.parameters())
    for key in ['normal','domain_weight','hole_weight']:
        p[key]=torch.as_tensor(points[key],dtype=ref.dtype,device=ref.device)
    return p


def parts_loss(model,p,lateral,top):
    q,j=model.sparse(p['domain']);r=residual(q,j)
    eq=(r[:,:2].square().sum(1)*p['domain_weight']).sum()
    const=(r[:,2:].square().sum(1)*p['domain_weight']).sum()
    left,right,upper,bottom,hole,gauge=[model.sparse(p[k])[0] for k in ['left','right','top','bottom','hole','gauge']]
    traction=((left[:,2]-lateral).square()+left[:,4].square()).mean()
    traction+=((right[:,2]-lateral).square()+right[:,4].square()).mean()
    traction+=((upper[:,3]-top).square()+upper[:,4].square()).mean()+bottom[:,4].square().mean()
    nx,ny=p['normal'].T
    traction+=(((hole[:,2]*nx+hole[:,4]*ny).square()+(hole[:,4]*nx+hole[:,3]*ny).square())*p['hole_weight']).sum()
    disp=bottom[:,1].square().mean()+gauge[0,0].square()
    return dict(equilibrium=eq,constitutive=const,traction=traction,displacement=disp)


def objective(parts):
    return parts['equilibrium']+parts['constitutive']+10*parts['traction']+100*parts['displacement']
