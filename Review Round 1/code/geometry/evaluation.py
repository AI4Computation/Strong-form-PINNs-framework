"""Field, engineering and physical-residual evaluation."""
import numpy as np
import torch
from sparse_mixed_pinn import residual

def weighted_quantiles(x,w):
    order=np.argsort(x);cumulative=np.cumsum(w[order]);cumulative/=cumulative[-1]
    return np.interp([.5,.9,.95,.99],cumulative,x[order]).tolist()


def metrics(pred,true,w):
    error=np.linalg.norm(pred-true,axis=1);norm=np.linalg.norm(true,axis=1)
    return dict(relative_l2_percent=float(100*np.sqrt(np.sum(w*error**2)/np.sum(w*norm**2))),
                vector_mae=float(np.sum(w*error)/np.sum(w)),vector_absolute_error_quantiles=weighted_quantiles(error,w),
                max_vector_absolute_error=float(error.max()))


@torch.no_grad()
def predict(model,xy):
    out=np.empty((len(xy),5),dtype=np.float32)
    for start in range(0,len(xy),2048):
        sl=slice(start,start+2048)
        out[sl]=model.sparse(model.prepare(xy[sl]))[0].cpu().numpy()
    return out


def tail_mass(error,w,fraction=.01):
    order=np.argsort(-error,kind='stable');sw=w[order];se=error[order]
    prior=np.cumsum(sw)-sw;use=np.minimum(sw,np.maximum(0.,fraction*w.sum()-prior))
    return float(np.sum(use*se**2)/max(np.sum(w*error**2),1e-30))


def score(q,ref):
    assert all(np.isfinite(v).all() for v in q.values())
    w=ref['area_weight'];values={};errors={}
    for name,cols,truth in [('u',[0,1],ref['u_ip']),('s',[2,3,4],ref['s_ip'])]:
        pred=q['q_ip'][:,cols];e=np.linalg.norm(pred-truth,axis=1);errors[name]=e
        values[name+'_area']=metrics(pred,truth,w)
        values[name+'_point']=metrics(pred,truth,np.ones(len(w)))
        values[name+'_area']['worst_one_percent_area_squared_error_share']=tail_mass(e,w)
        for region in ['near_wall','far_wall','corner','away_corner']:
            mask=ref[region]
            if mask.any():values[name+'_'+region]=metrics(pred[mask],truth[mask],w[mask])
    values['u_original_nodes']=metrics(q['u_node'],ref['u_node'],np.ones(len(ref['u_node'])))
    values['wall_u']=metrics(q['u_wall'],ref['wall_u'],ref['wall_weight'])
    convergence=lambda u:np.array([u[0,1]-u[1,1],u[2,0]-u[3,0]])
    actual=q['u_extrema'];truth=ref['extrema_u'];ac=convergence(actual);tc=convergence(truth)
    values['engineering']=dict(extrema_order=['crown','invert','right','left'],extrema_prediction=actual.tolist(),
        extrema_reference=truth.tolist(),extrema_absolute_error=abs(actual-truth).tolist(),
        extrema_vector_error=np.linalg.norm(actual-truth,axis=1).tolist(),
        convergence_order=['vertical','horizontal'],convergence_prediction=ac.tolist(),
        convergence_reference=tc.tolist(),convergence_absolute_error=abs(ac-tc).tolist())
    errors['wall_u']=np.linalg.norm(q['u_wall']-ref['wall_u'],axis=1)
    return values,errors


def stats(r,w=None):
    if r.ndim==1:r=r[:,None]
    e=np.sum(r*r,axis=1);w=np.ones(len(e))/len(e) if w is None else w/w.sum()
    order=np.argsort(e,kind='stable');quant=np.interp([.5,.9,.95,.99,1.],np.cumsum(w[order]),np.sqrt(e[order]))
    tail=np.argsort(-e,kind='stable');prior=np.cumsum(w[tail])-w[tail];fraction=np.minimum(w[tail],np.maximum(0.,.01-prior))
    return dict(total_mean=float(w@e),component_mean=(w[:,None]*r*r).sum(0).tolist(),norm_quantiles=quant.tolist(),top_one_percent_mass=float(fraction@e[tail]/max(w@e,1e-30)))


def region_stats(r,masks):
    e=(r*r).sum(1);out={}
    for name,mask in masks.items():
        if mask.any():out[name]=dict(**stats(r[mask]),point_fraction=float(mask.mean()),residual_square_mass=float(e[mask].sum()/max(e.sum(),1e-30)))
    return out


@torch.no_grad()
def field(model,x):
    out=np.empty((len(x),5),float)
    for start in range(0,len(x),1024):
        sl=slice(start,start+1024);t=torch.as_tensor(x[sl],dtype=next(model.parameters()).dtype,device=next(model.parameters()).device);out[sl]=model(t).cpu().numpy()
    assert np.isfinite(out).all();return out


@torch.no_grad()
def domain_residual(model,xy,setting):
    out=np.empty((len(xy),5),float)
    for start in range(0,len(xy),1024):
        sl=slice(start,start+1024);q,j=model.sparse(model.prepare(xy[sl]));out[sl]=residual(q,j,setting['E'],setting['nu']).cpu().numpy()
    assert np.isfinite(out).all();return out


def independent_boundary(domain,setting):
    points={};weights={};holes=[];tags=[]
    for curve in domain.curves:
        length=float(curve.quadrature(16)['w'].sum());q=curve.quadrature(max(1,int(np.ceil(length/.002))),3)
        if curve.hole:holes.append(q);tags.extend([curve.tag]*len(q['xy']))
        else:points[curve.tag]=q['xy'];weights[curve.tag]=q['w']/q['w'].sum()
    points.update(hole=np.vstack([q['xy'] for q in holes]),normal=np.vstack([q['normal'] for q in holes]),hole_tags=np.array(tags),hole_normal_stress=np.array([setting['hole_normal_stress'][t] for t in tags]),gauge=np.array([[0.,-.5]]))
    weights['hole']=np.concatenate([q['w'] for q in holes])
    for tag in set(tags):m=points['hole_tags']==tag;weights['hole'][m]/=weights['hole'][m].sum()
    weights['bottom_shear']=weights['bottom'];weights['bottom_uy']=weights['bottom'];return points,weights


def boundary(model,points,setting,weights):
    q={k:field(model,points[k]) for k in ['left','right','top','bottom','hole','gauge']};r={}
    for k in ['left','right']:r[k]=np.column_stack([q[k][:,2]-setting['lateral'],q[k][:,4]])
    r['top']=np.column_stack([q['top'][:,3]-setting['top'],q['top'][:,4]])
    r.update(bottom_shear=q['bottom'][:,4,None],bottom_uy=q['bottom'][:,1,None],gauge_ux=q['gauge'][:,0,None])
    nx,ny=points['normal'].T;h=q['hole'];pressure=points['hole_normal_stress']
    r['hole']=np.column_stack([h[:,2]*nx+h[:,4]*ny-pressure*nx,h[:,4]*nx+h[:,3]*ny-pressure*ny])
    out={k:stats(v,weights.get(k)) for k,v in r.items() if k!='hole'}
    for tag in setting['hole_normal_stress']:
        mask=points['hole_tags']==tag;out['hole_'+tag]=stats(r['hole'][mask],weights['hole'][mask])
    traction=sum(out[k]['total_mean'] for k in ['left','right','top','bottom_shear'])+sum(out['hole_'+t]['total_mean'] for t in setting['hole_normal_stress'])
    return dict(traction=traction,displacement=out['bottom_uy']['total_mean']+out['gauge_ux']['total_mean'],constraints=out),r

