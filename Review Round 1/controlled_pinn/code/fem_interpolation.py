"""Element-aware evaluation on fixed physical probes, using FE shape functions.

U is interpolated within the containing element. Stress is reconstructed from
its displacement gradient and the same plane-strain constitutive law. Exported
integration-point S remains the primary PINN reference; reconstruction accuracy
is separately measured at those integration points.
"""
import runtime
import numpy as np
from scipy.spatial import cKDTree

def shape(kind,rs):
    r,s=rs[:,0],rs[:,1]
    if kind in ('CPE4R','CPE8R'):
        ri=np.array([-1,1,1,-1])[None,:];si=np.array([-1,-1,1,1])[None,:]
        rr=r[:,None];ss=s[:,None]
        if kind=='CPE4R':
            N=(1+ri*rr)*(1+si*ss)/4
            dr=ri*(1+si*ss)/4;ds=si*(1+ri*rr)/4
        else:
            N=(1+ri*rr)*(1+si*ss)*(ri*rr+si*ss-1)/4
            dr=ri*(1+si*ss)*(2*ri*rr+si*ss)/4
            ds=si*(1+ri*rr)*(ri*rr+2*si*ss)/4
            N=np.column_stack([N,(1-r*r)*(1-s)/2,(1+r)*(1-s*s)/2,(1-r*r)*(1+s)/2,(1-r)*(1-s*s)/2])
            dr=np.column_stack([dr,-r*(1-s),(1-s*s)/2,-r*(1+s),-(1-s*s)/2])
            ds=np.column_stack([ds,-(1-r*r)/2,-s*(1+r),(1-r*r)/2,-s*(1-r)])
    elif kind in ('CPE3','CPE6'):
        l=np.column_stack([1-r-s,r,s]);dl=np.array([[-1,1,0],[-1,0,1]])
        if kind=='CPE3':
            N=l;dr=np.broadcast_to(dl[0],l.shape);ds=np.broadcast_to(dl[1],l.shape)
        else:
            N=l*(2*l-1);dr=(4*l-1)*dl[0];ds=(4*l-1)*dl[1]
            for a,b in [(0,1),(1,2),(2,0)]:
                N=np.column_stack([N,4*l[:,a]*l[:,b]])
                dr=np.column_stack([dr,4*(dl[0,a]*l[:,b]+l[:,a]*dl[0,b])])
                ds=np.column_stack([ds,4*(dl[1,a]*l[:,b]+l[:,a]*dl[1,b])])
    else:raise ValueError(kind)
    return N,np.stack([dr,ds],axis=1)

class Field:
    def __init__(self,data,E,nu):
        self.data=data;self.xy=data['xy_u'];self.u=data['u'];self.con=data['connectivity'];self.types=data['element_types']
        self.lam=E*nu/((1+nu)*(1-2*nu));self.mu=E/(2*(1+nu))
        self.centres=np.array([self.xy[c[c>=0]].mean(0) for c in self.con])
        self.tree=cKDTree(self.centres)

    def evaluate(self,points,k=16):
        points=np.asarray(points);n=len(points)
        _,candidates=self.tree.query(points,k=k)
        found=np.zeros(n,dtype=bool);displacement=np.zeros((n,2));stress=np.zeros((n,3))
        element_index=np.full(n,-1,dtype=int)
        for neighbour in range(k):
            pending=np.flatnonzero(~found)
            if not len(pending):break
            eid=candidates[pending,neighbour]
            for kind in np.unique(self.types[eid]):
                take=self.types[eid]==kind;rows=pending[take];els=eid[take]
                conn=self.con[els];nn=8 if kind=='CPE8R' else (6 if kind=='CPE6' else (4 if kind=='CPE4R' else 3))
                conn=conn[:,:nn];xnodes=self.xy[conn];unodes=self.u[conn]
                rs=np.zeros((len(rows),2))
                if nn in (3,6):rs[:]=1/3
                for _ in range(12):
                    N,dN=shape(kind,rs)
                    position=np.einsum('ni,nij->nj',N,xnodes)
                    J=np.einsum('nki,nij->nkj',dN,xnodes)
                    correction=np.linalg.solve(J.transpose(0,2,1), (points[rows]-position)[...,None])[...,0]
                    rs+=correction
                    if np.max(np.abs(correction))<1e-11:break
                if nn in (4,8):inside=(np.abs(rs)<=1+1e-7).all(1)
                else:inside=(rs>=-1e-7).all(1)&(rs.sum(1)<=1+1e-7)
                N,dN=shape(kind,rs)
                position=np.einsum('ni,nij->nj',N,xnodes)
                inside &= np.linalg.norm(position-points[rows],axis=1)<max(1.,np.abs(self.xy).max())*1e-7
                if not inside.any():continue
                rows=rows[inside];els=els[inside];N=N[inside];dN=dN[inside];xn=xnodes[inside];un=unodes[inside]
                J=np.einsum('nki,nij->nkj',dN,xn)
                gradN=np.linalg.solve(J,dN)
                gradU=np.einsum('nki,nij->nkj',gradN,un)
                exx=gradU[:,0,0];eyy=gradU[:,1,1];tr=exx+eyy
                displacement[rows]=np.einsum('ni,nij->nj',N,un)
                stress[rows]=np.column_stack([self.lam*tr+2*self.mu*exx,self.lam*tr+2*self.mu*eyy,
                                             self.mu*(gradU[:,0,1]+gradU[:,1,0])])
                found[rows]=True;element_index[rows]=els
        if not found.all():raise ValueError(f'{(~found).sum()} probes outside located elements; first: {points[~found][:5]}')
        return displacement,stress,element_index
