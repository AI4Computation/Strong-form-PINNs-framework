"""Trainable coarse + compact local mixed PINN; sparse field and analytic Jacobian.

The geometry is fixed in this candidate. Jacobians retain parameter autograd,
including every window derivative. No first-round modules or FEM are imported.
"""
import numpy as np
import torch
from torch import nn
from cavity_cover import windows


def local_count(width):
    return width*width + 9*width + 5


def allocation(k, target=110705, global_width=32, seed=250927):
    available=target-local_count(global_width)
    width=max(1,int(np.floor((-9+np.sqrt(81+4*(available/k-5)))/2)))
    if k*local_count(width)>available:
        raise ValueError('Insufficient parameter budget.')
    delta=local_count(width+1)-local_count(width)
    upgraded=min(k,int((available-k*local_count(width))//delta))
    widths=np.full(k,width,dtype=int)
    widths[np.random.default_rng(seed).permutation(k)[:upgraded]]=width+1
    return widths


class ExpertBank(nn.Module):
    def __init__(self, count, width, dtype=torch.float64):
        super().__init__()
        self.w1=nn.Parameter(torch.empty(count,width,2,dtype=dtype))
        self.b1=nn.Parameter(torch.zeros(count,width,dtype=dtype))
        self.w2=nn.Parameter(torch.empty(count,width,width,dtype=dtype))
        self.b2=nn.Parameter(torch.zeros(count,width,dtype=dtype))
        self.w3=nn.Parameter(torch.empty(count,5,width,dtype=dtype))
        self.b3=nn.Parameter(torch.zeros(count,5,dtype=dtype))
        for weight,fan_in,fan_out in [(self.w1,2,width),(self.w2,width,width),(self.w3,width,5)]:
            nn.init.uniform_(weight,-np.sqrt(6/(fan_in+fan_out)),np.sqrt(6/(fan_in+fan_out)))

    def evaluate(self, z, expert, inverse_scale, derivative=True):
        w1,w2,w3=self.w1[expert],self.w2[expert],self.w3[expert]
        h1=torch.tanh(torch.bmm(w1,z[:,:,None])[:,:,0]+self.b1[expert])
        h2=torch.tanh(torch.bmm(w2,h1[:,:,None])[:,:,0]+self.b2[expert])
        q=torch.bmm(w3,h2[:,:,None])[:,:,0]+self.b3[expert]
        if not derivative:return q
        d1=(1-h1*h1)[:,:,None]*w1*inverse_scale[:,None,:]
        d2=(1-h2*h2)[:,:,None]*torch.bmm(w2,d1)
        return q,torch.bmm(w3,d2)


class SparseMixedPINN(nn.Module):
    def __init__(self, centres, halfwidths, target=110705, global_width=32, seed=250927):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer('centres',torch.as_tensor(centres,dtype=torch.float64))
        self.register_buffer('halfwidths',torch.as_tensor(halfwidths,dtype=torch.float64))
        self.widths=allocation(len(centres),target,global_width,seed)
        self.coarse=ExpertBank(1,global_width)
        unique=sorted(set(self.widths.tolist()))
        self.banks=nn.ModuleList([ExpertBank(int(np.sum(self.widths==w)),w) for w in unique])
        self.bank_indices=[]
        for w in unique:self.bank_indices.append(np.flatnonzero(self.widths==w))

    def prepare(self, xy):
        """Fixed collocation geometry; returns only active point/expert edges."""
        xy=np.asarray(xy,dtype=float)
        c=self.centres.detach().cpu().numpy();h=self.halfwidths.detach().cpu().numpy()
        w,g=windows(xy,c,h,True)
        x=torch.as_tensor(xy,dtype=self.centres.dtype,device=self.centres.device)
        edges=[]
        for indices in self.bank_indices:
            point,expert=np.nonzero(w[:,indices]>0)
            patch=indices[expert]
            tensor=lambda a:torch.as_tensor(a,dtype=x.dtype,device=x.device)
            edges.append(dict(point=torch.as_tensor(point,device=x.device),expert=torch.as_tensor(expert,device=x.device),
                              z=tensor((xy[point]-c[patch])/h[patch]),inverse_scale=tensor(1/h[patch]),
                              weight=tensor(w[point,patch]),gradient=tensor(g[point,patch])))
        return dict(x=x,edges=edges)

    def sparse(self, prepared):
        x=prepared['x'];zero=torch.zeros(len(x),dtype=torch.long,device=x.device)
        q,jac=self.coarse.evaluate(2*x,zero,torch.full_like(x,2))
        for bank,e in zip(self.banks,prepared['edges']):
            v,j=bank.evaluate(e['z'],e['expert'],e['inverse_scale'])
            q=q.index_add(0,e['point'],e['weight'][:,None]*v)
            complete=e['weight'][:,None,None]*j + v[:,:,None]*e['gradient'][:,None,:]
            jac=jac.index_add(0,e['point'],complete)
        return q,jac

    def dense(self,x):
        """Independent all-expert path, only for small numerical audits."""
        z=(x[:,None,:]-self.centres[None,:,:])/self.halfwidths[None,:,:]
        raw=torch.prod(torch.clamp(1-z*z,min=0)**3,dim=2)
        weight=raw/raw.sum(dim=1,keepdim=True)
        zero=torch.zeros(len(x),dtype=torch.long,device=x.device)
        result=self.coarse.evaluate(2*x,zero,torch.full_like(x,2),False)
        for bank,indices in zip(self.banks,self.bank_indices):
            for local,patch in enumerate(indices):
                idx=torch.full((len(x),),local,dtype=torch.long,device=x.device)
                result=result+weight[:,patch,None]*bank.evaluate(z[:,patch],idx,torch.ones_like(x),False)
        return result


def residual(q,jac,young=1.333,poisson=.3333):
    mu=young/(2*(1+poisson));lam=young*poisson/((1+poisson)*(1-2*poisson))
    ex,ey=jac[:,0,0],jac[:,1,1]
    return torch.stack([jac[:,2,0]+jac[:,4,1],jac[:,4,0]+jac[:,3,1],
                        q[:,2]-(lam+2*mu)*ex-lam*ey,q[:,3]-lam*ex-(lam+2*mu)*ey,
                        q[:,4]-mu*(jac[:,0,1]+jac[:,1,0])],dim=1)
