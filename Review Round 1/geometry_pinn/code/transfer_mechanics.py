"""Physical-data adapter for mixed strong-form PINNs; no fitting or file access."""
import numpy as np
import torch
from p2_components import shared_points,prepare_loss
from sparse_mixed_pinn import residual

def sample(domain,covers,seed,setting):
    p=shared_points(domain,covers,seed)
    curves=[c for c in domain.curves if c.hole];lengths=np.array([c.quadrature(32,5)['w'].sum() for c in curves]);desired=1000*lengths/lengths.sum();counts=np.floor(desired).astype(int)
    counts[np.argsort(-(desired-counts),kind='stable')[:1000-counts.sum()]]+=1
    labels=np.concatenate([np.repeat(c.tag,n) for c,n in zip(curves,counts)])
    p['hole_tags']=labels;p['hole_normal_stress']=np.array([setting['hole_normal_stress'][str(tag)] for tag in labels])
    # Each physical boundary condition is one mean, just as each outer edge is one mean.
    w=p['hole_weight'].copy()
    for tag in np.unique(labels):mask=labels==tag;w[mask]/=w[mask].sum()
    p['hole_weight']=w;return p

def prepare(model,points):
    p=prepare_loss(model,points);ref=next(model.parameters());p['hole_normal_stress']=torch.as_tensor(points['hole_normal_stress'],dtype=ref.dtype,device=ref.device);return p

def parts(model,p,setting):
    q,j=model.sparse(p['domain']);r=residual(q,j,setting['E'],setting['nu'])
    eq=(r[:,:2].square().sum(1)*p['domain_weight']).sum();co=(r[:,2:].square().sum(1)*p['domain_weight']).sum()
    o={k:model.sparse(p[k])[0] for k in ['left','right','top','bottom','hole','gauge']}
    traction=sum(((o[k][:,2]-setting['lateral']).square()+o[k][:,4].square()).mean() for k in ['left','right'])
    traction+=((o['top'][:,3]-setting['top']).square()+o['top'][:,4].square()).mean()+o['bottom'][:,4].square().mean()
    nx,ny=p['normal'].T;h=o['hole'];pressure=p['hole_normal_stress']
    tx=h[:,2]*nx+h[:,4]*ny-pressure*nx;ty=h[:,4]*nx+h[:,3]*ny-pressure*ny
    traction+=(p['hole_weight']*(tx.square()+ty.square())).sum()
    disp=o['bottom'][:,1].square().mean()+o['gauge'][0,0].square()
    return dict(equilibrium=eq,constitutive=co,traction=traction,displacement=disp)

def independent_objective(model,points,setting):
    ref=next(model.parameters());as_tensor=lambda v:torch.as_tensor(v,dtype=ref.dtype,device=ref.device)
    x=as_tensor(points['domain']).detach().clone().requires_grad_(True);q=model(x)
    j=torch.stack([torch.autograd.grad(q[:,k].sum(),x,create_graph=True,retain_graph=True)[0] for k in range(5)],1)
    E,nu=setting['E'],setting['nu'];lam=E*nu/((1+nu)*(1-2*nu));mu=E/(2*(1+nu));ex=j[:,0,0];ey=j[:,1,1]
    rs=torch.stack([j[:,2,0]+j[:,4,1],j[:,4,0]+j[:,3,1],q[:,2]-lam*(ex+ey)-2*mu*ex,q[:,3]-lam*(ex+ey)-2*mu*ey,q[:,4]-mu*(j[:,0,1]+j[:,1,0])],1)
    value=(as_tensor(points['domain_weight'])*rs.square().sum(1)).sum();out={k:model(as_tensor(points[k])) for k in ['left','right','top','bottom','hole','gauge']};t=0.
    for k,i,target in [('left',2,setting['lateral']),('right',2,setting['lateral']),('top',3,setting['top'])]:t+=((out[k][:,i]-target).square()+out[k][:,4].square()).mean()
    t+=out['bottom'][:,4].square().mean();h=out['hole'];n=as_tensor(points['normal']);sigma=torch.stack([torch.stack([h[:,2],h[:,4]],1),torch.stack([h[:,4],h[:,3]],1)],1)
    traction=torch.einsum('nij,nj->ni',sigma,n)-as_tensor(points['hole_normal_stress'])[:,None]*n
    t+=(as_tensor(points['hole_weight'])*traction.square().sum(1)).sum()
    return value+10*t+100*(out['bottom'][:,1].square().mean()+out['gauge'][0,0].square())

def distance_masks(xy,domain):
    d=np.full(len(xy),np.inf);corners=[]
    for h in domain.holes:
        if h['kind']=='polygon':
            v=np.asarray(h['vertices'])
            for i,(a,b) in enumerate(zip(v,np.roll(v,-1,axis=0))):
                t=np.clip(((xy-a)*(b-a)).sum(1)/np.sum((b-a)**2),0,1);d=np.minimum(d,np.linalg.norm(xy-a-t[:,None]*(b-a),axis=1))
                prev=a-v[i-1];nxt=b-a;angle=np.arccos(np.clip(prev@nxt/(np.linalg.norm(prev)*np.linalg.norm(nxt)),-1,1))
                if angle>np.pi/6:corners.append(a)
        else:
            angle=h.get('angle',0.);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);q=(xy-h['center'])@R;a=np.asarray(h['axes']);a2=a*a
            low=np.zeros(len(xy));high=np.linalg.norm(q*a,axis=1)
            for _ in range(52):
                mid=(low+high)/2;f=np.sum((a*q/(mid[:,None]+a2))**2,axis=1)>1
                low=np.where(f,mid,low);high=np.where(f,high,mid)
            closest=a2*q/(((low+high)/2)[:,None]+a2);d=np.minimum(d,np.linalg.norm(q-closest,axis=1))
    corner=np.zeros(len(xy),bool)
    for v in corners:corner|=np.linalg.norm(xy-v,axis=1)<.02
    return dict(near_wall=d<=.05,far_wall=d>.05,corner=corner,away_corner=~corner)

def wall_points(domain,count=2048):
    curves=[c for c in domain.curves if c.hole];lengths=np.array([c.quadrature(32,5)['w'].sum() for c in curves]);want=count*lengths/lengths.sum();counts=np.floor(want).astype(int);counts[np.argsort(-(want-counts),kind='stable')[:count-counts.sum()]]+=1
    xs=[];ns=[];ws=[];tags=[]
    for c,n in zip(curves,counts):
        assert n>0;x,normal,_,jac=c.evaluate((np.arange(n)+.5)/n);xs.append(x);ns.append(normal);ws.append(jac/n);tags.extend([c.tag]*n)
    primary=next(h for h in domain.holes if h['tag']=='cavity');v=np.asarray(primary['vertices']);ext=[]
    for dim,ismax in [(1,True),(1,False),(0,True),(0,False)]:
        target=v[:,dim].max() if ismax else v[:,dim].min();ext.append(v[abs(v[:,dim]-target)<1e-10].mean(0))
    return np.vstack(xs),np.vstack(ns),np.concatenate(ws),np.asarray(tags),np.array(ext),np.array([[0,-1],[0,1],[-1,0],[1,0]])
