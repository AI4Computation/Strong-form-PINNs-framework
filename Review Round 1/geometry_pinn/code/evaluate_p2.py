"""Evaluate all frozen P2A terminals; no training, selection or parameter changes."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from p2_components import make_model
from p2_fem_interpolation import Field,shape
from cavity_cover import normalized_domain

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parents[1]/'research'
BATCH=ROOT/'results/R2_P2A'
OUT=BATCH/'evaluation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}


def from_elements(data,points,elements):
    """Known containing element; Newton invert coordinates, then interpolate."""
    u=np.empty((len(points),2));s=np.empty((len(points),3));max_position=0.
    young,nu=1.333,.3333;mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))
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


def wall_points(domain,geometry):
    curves=[c for c in domain.curves if c.hole];lengths=np.array([c.quadrature(32,5)['w'].sum() for c in curves])
    counts=np.floor(1024*lengths/lengths.sum()).astype(int)
    counts[np.argsort(-(1024*lengths/lengths.sum()-counts),kind='stable')[:1024-counts.sum()]]+=1
    x,normal,w=[],[],[]
    for curve,count in zip(curves,counts):
        a,n,_,jac=curve.evaluate((np.arange(count)+.5)/count)
        x.append(a);normal.append(n);w.append(jac/count)
    # Fixed engineering points: crown, invert, right side, left side.
    a,b=(.1,.1) if geometry=='C1' else (.2,.015)
    extrema=np.array([[0,b],[0,-b],[a,0],[-a,0]])
    en=np.array([[0,-1],[0,1],[-1,0],[1,0]])
    return np.vstack(x),np.vstack(normal),np.concatenate(w),extrema,en


def distance_masks(xy,domain):
    h=domain.holes[0]
    if h['kind']=='ellipse':d=abs(np.linalg.norm(xy,axis=1)-h['axes'][0]);corner=np.zeros(len(xy),bool)
    else:
        v=np.asarray(h['vertices']);d=np.full(len(xy),np.inf)
        for a,b in zip(v,np.roll(v,-1,axis=0)):
            t=np.clip((xy-a)@(b-a)/np.sum((b-a)**2),0,1)
            d=np.minimum(d,np.linalg.norm(xy-a-t[:,None]*(b-a),axis=1))
        corner=np.min(np.linalg.norm(xy[:,None,:]-v[None,:,:],axis=2),axis=1)<.02
    return dict(near_wall=d<=.05,far_wall=d>.05,corner=corner,away_corner=~corner)


def main():
    m=read(BATCH/'manifest.json');assert m['status']=='fit_complete' and len(m['completed'])==15
    for name,files in m['terminal_sha256'].items():
        for file,h in files.items():assert sha(BATCH/name/file)==h
    for name,h in m['source_sha256'].items():assert sha(ROOT/'code'/name)==h
    assert not OUT.exists(),'Evaluation already started; inspect rather than overwrite.'
    OUT.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=read(ROOT/'protocols/R2_P2A_development.json');assert sha(ROOT/'protocols/R2_P2A_development.json')==m['protocol_sha256']
    geometries=read(ROOT/'inputs/cavity_geometries.json')
    report=dict(terminal_manifest_sha256=sha(BATCH/'manifest.json'),evaluator_sha256=sha(__file__),
                interpolation_sha256=sha(ROOT/'code/p2_fem_interpolation.py'),references={},cases={},
                all_terminals_frozen_before_fem_access=True,timing_valid_for_comparison=False)
    for case,setting in protocol['cases'].items():
        geometry=setting['geometry'];domain,_,_=normalized_domain(geometries[geometry])
        paths=[]
        if geometry=='C1':
            base=OLD/'controlled_pinn/fem/tr3_circle_sq0p0025_UnitL.npz'
            selection=OLD/f'controlled_pinn/fem/references/{case}.npz'
            data=load(base);ref=load(selection);paths=[base,selection]
            assert np.array_equal(data['xy_u'],ref['xy_u']) and np.array_equal(data['xy_s'],ref['xy_s'])
            data.update(u=ref['u'],s=ref['s'])
        else:
            base=OLD/'validation_and_cost/shape_references/full_integration/tsf_slender_g3_Load.npz'
            meshfile=OLD/'validation_and_cost/shape_references/tsr_slender_g3_mesh.npz'
            data=load(base);mesh=load(meshfile);paths=[base,meshfile]
            assert np.array_equal(data['connectivity'],mesh['connectivity']-1)
            # Exact input coordinates avoid tiny-cell derivative amplification.
            data['xy_u']=mesh['xy']
        ids=data['element_ids'];order=np.argsort(ids)
        elements=order[np.searchsorted(ids[order],data['ip_id'][:,0])]
        assert np.array_equal(ids[elements],data['ip_id'][:,0])
        uip,scheck,pos=from_elements(data,data['xy_s'],elements)
        w=data['volume'];reconstruction=metrics(scheck,data['s'],w)['relative_l2_percent']
        assert reconstruction<.02,(case,'stress reconstruction %',reconstruction)
        assert (w>0).all() and abs(w.sum()-domain.area)<1e-5
        wall,normal,ww,extrema,en=wall_points(domain,geometry)
        actual=np.vstack([wall,extrema]);normals=np.vstack([normal,en])
        def boundary_reference(offset):
            query=actual-offset*normals
            if geometry=='C1':return field.evaluate(query,k=32)[0]
            i=np.searchsorted(mesh['x'],query[:,0],side='right')-1
            j=np.searchsorted(mesh['y'],query[:,1],side='right')-1
            eid=mesh['grid_element_ids'][i,j];assert (eid>0).all()
            return from_elements(data,query,eid-1)[0]
        if geometry=='C1':field=Field(data,1.333,.3333)
        boundary_u=boundary_reference(1e-8);shifted=boundary_reference(2e-8)
        wall_shift=float(np.max(abs(boundary_u-shifted)))
        assert wall_shift<1e-5
        reference_path=OUT/f'{case}_reference.npz'
        masks=distance_masks(data['xy_s'],domain)
        np.savez_compressed(reference_path,xy_ip=data['xy_s'],area_weight=w,u_ip=uip,s_ip=data['s'],
                            xy_node=data['xy_u'],u_node=data['u'],wall=wall,wall_weight=ww,wall_u=boundary_u[:-4],
                            extrema=extrema,extrema_u=boundary_u[-4:],**masks)
        report['references'][case]=dict(source_sha256={str(f):sha(f) for f in paths},derived_sha256=sha(reference_path),
                                       stress_reconstruction_percent=reconstruction,maximum_position_defect=pos,
                                       wall_reference_solid_offset=1e-8,wall_offset_sensitivity_max=wall_shift,area=float(w.sum()),integration_points=len(w))
        with np.load(BATCH/f'{geometry}_covers.npz') as z:covers=[(z['centres'],z['halfwidths']),(z['uniform_centres'],z['uniform_halfwidths'])]
        case_results={};errors={}
        for method in protocol['methods']:
            name=case+'_'+method;model=make_model(method,protocol['seed'],covers,'cuda').float()
            model.load_state_dict(torch.load(BATCH/name/'terminal.pt',map_location='cuda',weights_only=True));model.eval()
            start=time.perf_counter()
            q=predict(model,data['xy_s']);un=predict(model,data['xy_u'])[:,:2];wu=predict(model,actual)[:,:2]
            predictions=OUT/f'{name}_predictions.npz'
            np.savez_compressed(predictions,q_ip=q,u_node=un,u_wall=wu[:-4],u_extrema=wu[-4:])
            values={}
            for fieldname,cols,truth in [('u',[0,1],uip),('s',[2,3,4],data['s'])]:
                values[fieldname+'_area']=metrics(q[:,cols],truth,w)
                values[fieldname+'_point']=metrics(q[:,cols],truth,np.ones(len(w)))
                for region,mask in masks.items():
                    if mask.any():values[fieldname+'_'+region]=metrics(q[mask][:,cols],truth[mask],w[mask])
                errors[(method,fieldname)]=np.linalg.norm(q[:,cols]-truth,axis=1)
            values['u_original_nodes']=metrics(un,data['u'],np.ones(len(un)))
            values['wall_u']=metrics(wu[:-4],boundary_u[:-4],ww)
            def convergence(u):return np.array([u[0,1]-u[1,1],u[2,0]-u[3,0]])
            values['engineering']=dict(extrema_order=['crown','invert','right','left'],extrema_reference=boundary_u[-4:].tolist(),
                                       extrema_prediction=wu[-4:].tolist(),extrema_absolute_error=abs(wu[-4:]-boundary_u[-4:]).tolist(),
                                       convergence_reference=convergence(boundary_u[-4:]).tolist(),convergence_prediction=convergence(wu[-4:]).tolist(),
                                       convergence_absolute_error=abs(convergence(wu[-4:])-convergence(boundary_u[-4:])).tolist())
            case_results[method]=dict(metrics=values,predictions_sha256=sha(predictions),development_evaluation_seconds=time.perf_counter()-start)
            print(case,method,'area U/S %',values['u_area']['relative_l2_percent'],values['s_area']['relative_l2_percent'],flush=True)
            del model;torch.cuda.empty_cache()
        for method in protocol['methods']:
            case_results[method]['area_fraction_lower_vector_error']={control:{field:float(np.sum(w*(errors[(method,field)]<errors[(control,field)]))/w.sum()) for field in ['u','s']} for control in protocol['methods'] if control!=method}
        report['cases'][case]=case_results
        write(OUT/f'{case}_metrics.json',case_results)
        write(OUT/'evaluation_progress.json',dict(completed=list(report['cases']),active=None))
        del data,uip,scheck,errors
    ratios={}
    for case,results in report['cases'].items():
        ratios[case]={control:{key:results['geometry_local']['metrics'][key]['relative_l2_percent']/results[control]['metrics'][key]['relative_l2_percent'] for key in ['u_area','s_area','wall_u']} for control in ['uniform_local','fourier_half']}
    mean_ratio=float(np.exp(np.mean([np.log(ratios[c]['uniform_local'][k]) for c in ratios for k in ['u_area','s_area']])))
    independent=mean_ratio<=.90 and all(ratios[c]['uniform_local'][k]<=1.10 for c in ratios for k in ['u_area','s_area','wall_u'])
    close=all(ratios[c]['fourier_half'][k]<=1.10 for c in ratios for k in ['u_area','s_area'])
    better=sum(min(ratios[c]['fourier_half'][k] for k in ['u_area','s_area'])<=.90 and max(ratios[c]['fourier_half'][k] for k in ['u_area','s_area'])<=1.10 for c in ratios)>=2
    report['screening']=dict(ratios=ratios,geometric_mean_uniform_ratio=mean_ratio,independent_gain_gate=independent,
                             half_fourier_gate=close or better,advance=independent and (close or better),single_seed_development_only=True)
    write(OUT/'analysis.json',report)
    print(json.dumps(report['screening']),flush=True)


if __name__=='__main__':main()
