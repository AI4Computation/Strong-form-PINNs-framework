import runtime
from runtime import torch,DEVICE,DTYPE
import numpy as np
from models import legacy

def relative(reference,prediction):
    denominator=float(np.linalg.norm(reference))
    return None if denominator==0 else float(100*np.linalg.norm(prediction-reference)/denominator)

def field_metrics(reference,prediction,names):
    delta=prediction-reference
    out={'vector_pct':relative(reference,prediction),'rmse':float(np.sqrt(np.mean(delta**2))),
         'max_abs':float(np.abs(delta).max())}
    for j,name in enumerate(names):
        r,p=reference[:,j],prediction[:,j]
        out[name+'_pct']=relative(r,p)
        out[name+'_rmse']=float(np.sqrt(np.mean((r-p)**2)))
        out[name+'_max_abs']=float(np.abs(r-p).max())
        out[name+'_pearson']=float(np.corrcoef(r,p)[0,1]) if np.std(r)>0 and np.std(p)>0 else None
    return out

def predict(model,coordinates,kind,geometry,batch=4096):
    scale_x=75. if geometry=='tunnel' else 1.
    scale_out=(.01875 if kind=='u' else 2.5) if geometry=='tunnel' else 1.
    result=[]
    for start in range(0,len(coordinates),batch):
        xy=torch.as_tensor(coordinates[start:start+batch]/scale_x,dtype=DTYPE,device=DEVICE)
        if kind=='s' and isinstance(model,legacy.DEMNet):
            with torch.enable_grad():
                xy.requires_grad_(True);uv=model(xy)
                gu=torch.autograd.grad(uv[:,0].sum(),xy,retain_graph=True)[0]
                gv=torch.autograd.grad(uv[:,1].sum(),xy)[0]
                tr=gu[:,0]+gv[:,1]
                out=torch.stack([legacy.LAM*tr+2*legacy.MU*gu[:,0],
                    legacy.LAM*tr+2*legacy.MU*gv[:,1],legacy.MU*(gu[:,1]+gv[:,0])],1)
        else:
            with torch.no_grad():out=model(xy)[:,0:2 if kind=='u' else 5]
            if kind=='s':out=out[:,2:5]
        result.append(out.detach().cpu().numpy()*scale_out)
    return np.concatenate(result).astype(np.float64)

def evaluate(model,ref,geometry):
    u=predict(model,ref['xy_u'],'u',geometry)
    s=predict(model,ref['xy_s'],'s',geometry)
    out={f'u_{k}':v for k,v in field_metrics(ref['u'],u,['x','y']).items()}
    out.update({f's_{k}':v for k,v in field_metrics(ref['s'],s,['xx','yy','xy']).items()})
    if geometry=='circle':
        r=np.linalg.norm(ref['xy_s'],axis=1);mask=(r>.1)&(r<=.2)
        out['s_near_pct']=relative(ref['s'][mask],s[mask])
    return out
