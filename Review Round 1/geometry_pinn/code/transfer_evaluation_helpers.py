"""Parameterized, independent new copies of verified interpolation/audit routines."""
import numpy as np
import torch
from p2_fem_interpolation import shape
from sparse_mixed_pinn import residual

def from_elements(data,points,elements,young,nu):
    """Known containing element; Newton invert coordinates, then interpolate."""
    u=np.empty((len(points),2));s=np.empty((len(points),3));max_position=0.
    mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))
    for start in range(0,len(points),16384):
        end=min(len(points),start+16384)
        ids=elements[start:end];types=data['element_types'][ids]
        for original_kind in np.unique(types):
            kind=str(original_kind);kind='CPE8R' if kind=='CPE8' else kind
            take=np.flatnonzero(types==original_kind);rows=start+take
            n={'CPE8R':8,'CPE6':6,'CPE4R':4,'CPE3':3}[kind]
            conn=data['connectivity'][ids[take],:n];xn=data['xy_u'][conn];un=data['u'][conn]
            rs=np.zeros((len(take),2));rs[:]=1/3 if n in [3,6] else 0
            for _ in range(12):
                N,dN=shape(kind,rs);position=np.einsum('ni,nij->nj',N,xn)
                J=np.einsum('nki,nij->nkj',dN,xn)
                step=np.linalg.solve(J.transpose(0,2,1),(points[rows]-position)[...,None])[...,0]
                rs+=step
                if np.max(abs(step))<1e-11:break
            N,dN=shape(kind,rs);position=np.einsum('ni,nij->nj',N,xn)
            max_position=max(max_position,float(np.max(abs(position-points[rows]))))
            J=np.einsum('nki,nij->nkj',dN,xn);gradN=np.linalg.solve(J,dN)
            gu=np.einsum('nki,nij->nkj',gradN,un)
            u[rows]=np.einsum('ni,nij->nj',N,un)
            xx,yy=gu[:,0,0],gu[:,1,1];tr=xx+yy
            s[rows]=np.column_stack([lam*tr+2*mu*xx,lam*tr+2*mu*yy,mu*(gu[:,0,1]+gu[:,1,0])])
    assert max_position<1e-9
    return u,s,max_position


def numerical_audit(model,xy,r,E,nu):
    idx=np.unique(np.r_[np.argsort(-np.sum(r*r,axis=1),kind='stable')[:8],np.random.default_rng(9274103).choice(len(xy),8,replace=False)])
    model.cpu().double()
    with torch.no_grad():q,j=model.sparse(model.prepare(xy[idx]));analytic=residual(q,j,E,nu).numpy()
    with torch.enable_grad():
        x=torch.tensor(xy[idx],dtype=torch.float64,requires_grad=True);v=model(x)
        jac=torch.stack([torch.autograd.grad(v[:,i].sum(),x,retain_graph=True)[0] for i in range(5)],1)
    v=v.detach().numpy();j=jac.detach().numpy();mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
    stress=np.column_stack([(lam+2*mu)*j[:,0,0]+lam*j[:,1,1],lam*j[:,0,0]+(lam+2*mu)*j[:,1,1],mu*(j[:,0,1]+j[:,1,0])])
    independent=np.column_stack([j[:,2,0]+j[:,4,1],j[:,4,0]+j[:,3,1],v[:,2:]-stress]);scale=max(1.,float(abs(independent).max()))
    e64=float(abs(analytic-independent).max()/scale);e32=float(abs(r[idx]-independent).max()/scale)
    assert e64<1e-9 and e32<3e-4,(e64,e32)
    return dict(float64_scaled_max_error=e64,float32_scaled_max_error=e32,passed=True),dict(indices=idx,xy=xy[idx],analytic_float64=analytic,autograd_float64=independent,saved_float32=r[idx])