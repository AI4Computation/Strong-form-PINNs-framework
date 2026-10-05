"""Independent CPU quadrature audit of every saved DEM endpoint; no retraining."""
import runtime
from runtime import torch
torch.set_num_threads(1)
from models import ROOT,legacy
from fem_interpolation import Field
from summarize import write_csv
import numpy as np,json,hashlib,time

def rule(order):
    a,w=np.polynomial.legendre.leggauss(order);a=a*.5;w=w*.5
    x,y=np.meshgrid(a,a,indexing='ij');wx,wy=np.meshgrid(w,w,indexing='ij')
    points=np.column_stack([x.ravel(),y.ravel()]).astype(np.float32)
    weight=(wx*wy).ravel();keep=np.linalg.norm(points,axis=1)>.1
    return points[keep],weight[keep],np.column_stack([np.full(order,-.5),a]),np.column_stack([np.full(order,.5),a]),np.column_stack([a,np.full(order,.5)]),w

def energy(model,quadrature,p,pt):
    xy,volume,left,right,top,wb=quadrature;internal=0.
    dtype=next(model.parameters()).dtype
    for start in range(0,len(xy),4096):
        x=torch.tensor(xy[start:start+4096],dtype=dtype,requires_grad=True)
        uv=model(x)
        gu=torch.autograd.grad(uv[:,0].sum(),x,retain_graph=True)[0]
        gv=torch.autograd.grad(uv[:,1].sum(),x)[0]
        exx,eyy=gu[:,0],gv[:,1];exy=(gu[:,1]+gv[:,0])*.5;tr=exx+eyy
        density=.5*legacy.LAM*tr.square()+legacy.MU*(exx.square()+eyy.square()+2*exy.square())
        internal+=float(np.dot(volume[start:start+4096],density.detach().numpy().astype(np.float64)))
    with torch.no_grad():
        ul=model(torch.tensor(left,dtype=dtype))[:,0].numpy().astype(np.float64)
        ur=model(torch.tensor(right,dtype=dtype))[:,0].numpy().astype(np.float64)
        vt=model(torch.tensor(top,dtype=dtype))[:,1].numpy().astype(np.float64)
    work=float(np.dot(wb,-p*ul+p*ur+pt*vt))
    return internal,work,internal-work

def main():
    manifest=json.loads((ROOT/'config/run_manifest.json').read_text());configs=[c for c in manifest if c['method']=='dem']
    assert len(configs)==46
    rules={n:rule(n) for n in [79,158,316]}
    _,_,left,right,top,wb=rules[316];boundary=np.vstack([left,right,top]);unit=[]
    for step in ['UnitL','UnitT']:
        data=dict(np.load(ROOT/'fem'/f'tr3_circle_sq0p0025_{step}.npz'))
        uv,_,_=Field(data,1.333,.3333).evaluate(boundary);unit.append(uv)
    rows=[];precision=[]
    for c in configs:
        folder=ROOT/'runs'/c['id'];r=json.loads((folder/'result.json').read_text())
        cp=folder/f"step_{r['accepted_steps']:05d}.pt"
        state=torch.load(cp,map_location='cpu',weights_only=True)['state_dict']
        model=legacy.DEMNet().to('cpu');model.load_state_dict(state)
        p,pt=c['p_lateral'],c['p_top'];uf=-p*unit[0]-pt*unit[1];n=316
        reference_work=float(np.dot(wb,-p*uf[:n,0]+p*uf[n:2*n,0]+pt*uf[2*n:,1]))
        for order,q in rules.items():
            internal,work,potential=energy(model,q,p,pt)
            rows.append({'id':c['id'],'case':c['case'],'seed':c['seed'],'order':order,'quadrature_points':len(q[0]),
                'quadrature_area':float(q[1].sum()),'exact_solid_area':1-np.pi*.1**2,
                'internal_energy':internal,'external_work':work,'potential_energy':potential,
                'recorded_training_final_loss':r['final_loss'],'fem_reference_potential_energy':-.5*reference_work,
                'stress_vector_error_pct':r['metrics']['s_vector_pct'],
                'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()})
        # Predefined numerical check of the two largest stress-error endpoints.
        if c['id'] in ['C7_dem_s45','C1_dem_s44']:
            model=model.double()
            for order in [79,316]:
                internal,work,potential=energy(model,rules[order],p,pt)
                precision.append({'id':c['id'],'order':order,'dtype':'float64','internal_energy':internal,
                    'external_work':work,'potential_energy':potential})
        print('DEM QUADRATURE '+c['id'],flush=True)
    write_csv('dem_quadrature_audit.csv',rows);write_csv('dem_precision_check.csv',precision)
    report={'complete':True,'n_endpoints':46,'orders':[79,158,316], 'device':'CPU','inference_dtype':'float32, with two endpoint float64 checks',
        'scope':'Independent energy evaluation of unchanged endpoint weights, not new training or replacement results. Tensor Gauss points outside the analytic circular void; increasing both volume and boundary rules. Fine FEM equilibrium potential estimated by -0.5 times boundary work. Neither the 316 rule nor its difference is claimed to be a strict integration error bound.',
        'source_sha256':hashlib.sha256(__file_bytes()).hexdigest()}
    (ROOT/'checks/dem_quadrature_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def __file_bytes():
    from pathlib import Path
    return Path(__file__).read_bytes()

if __name__=='__main__':main()
