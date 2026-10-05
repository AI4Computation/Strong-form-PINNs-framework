"""Odd-count compatibility only; all frozen even-count models are delegated unchanged."""
import math
import numpy as np
import torch
from p2_components import FeatureMixedPINN
from shared_geometry_features import make_shared_model

class OddGeometryPINN(FeatureMixedPINN):
    def __init__(self,seed,centres,widths):
        count=len(centres)
        if not (0<count<1000 and count%2==1):raise ValueError('Odd fixture requires odd1<=K<1000')
        if np.asarray(centres).shape!=(count,2) or np.asarray(widths).shape!=(count,2):raise ValueError('Expected Kx2 centres and widths')
        if not np.isfinite(centres).all() or not np.isfinite(widths).all() or not (np.asarray(widths)>0).all():raise ValueError('Finite centres and positive widths required')
        super().__init__('fourier_half',seed)
        self.global_count=1000-count;self.sine_count=(self.global_count+1)//2;self.cosine_count=self.global_count//2
        self.B=self.B[:self.sine_count].clone()
        self.register_buffer('centres',torch.as_tensor(np.asarray(centres),dtype=torch.float64).clone())
        self.register_buffer('widths',torch.as_tensor(np.asarray(widths),dtype=torch.float64).clone())

    def features(self,x):
        phase=2*math.pi*x@self.B.T
        z=(x[:,None]-self.centres[None])/self.widths[None]
        return torch.cat([torch.sin(phase),torch.cos(phase[:,:self.cosine_count]),torch.exp(-.5*z.square().sum(2))],1)

    def prepare(self,xy):
        ref=next(self.parameters());x=torch.as_tensor(xy,dtype=ref.dtype,device=ref.device)
        phase=2*math.pi*x@self.B.T;z=(x[:,None]-self.centres[None])/self.widths[None];g=torch.exp(-.5*z.square().sum(2))
        f=torch.cat([torch.sin(phase),torch.cos(phase[:,:self.cosine_count]),g],1)
        d=torch.cat([2*math.pi*torch.cos(phase)[:,:,None]*self.B[None],-2*math.pi*torch.sin(phase[:,:self.cosine_count])[:,:,None]*self.B[None,:self.cosine_count],-g[:,:,None]*z/self.widths[None]],1)
        return dict(x=x,features=f,df=d)

def make_budget_compatible(method,seed,centres,widths,uniform=None):
    if method not in ['geometry_rbf_fourier','uniform_rbf_fourier']:raise ValueError('Wrapper supports the two Gaussian-Fourier representations only')
    if method=='uniform_rbf_fourier':
        if uniform is None:raise ValueError('Uniform construction required')
        centres,widths=uniform
    count=len(centres)
    if not 0<count<1000:raise ValueError('Feature budget requires1<=K<1000; no automatic deletion or coarsening')
    if count%2==0:return make_shared_model(method,seed,centres,widths,uniform)
    return OddGeometryPINN(seed,centres,widths)
