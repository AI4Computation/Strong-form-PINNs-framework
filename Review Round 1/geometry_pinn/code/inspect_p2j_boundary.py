"""Mesh boundary versus analytic cavity, geometry only and no PINN values."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import json,hashlib
import numpy as np
from cavity_cover import normalized_domain
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parents[1]/'research';OUT=ROOT/'results/R2_P2J_reference_repair'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def distance(x,h):
    if h['kind']=='polygon':
        v=np.asarray(h['vertices']);d=np.full(len(x),np.inf)
        for a,b in zip(v,np.roll(v,-1,axis=0)):
            t=np.clip(((x-a)*(b-a)).sum(1)/np.sum((b-a)**2),0,1);d=np.minimum(d,np.linalg.norm(x-a-t[:,None]*(b-a),axis=1))
        return d
    angle=h.get('angle',0.);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);q=(x-h['center'])@rot;a,b=h['axes'];theta=np.arctan2(q[:,1]/b,q[:,0]/a)
    for _ in range(12):
        co,si=np.cos(theta),np.sin(theta);p=np.column_stack([a*co,b*si]);dp=np.column_stack([-a*si,b*co]);dd=-p;change=((p-q)*dp).sum(1)/((dp*dp).sum(1)+((p-q)*dd).sum(1));theta-=change
        if abs(change).max()<1e-14:break
    p=np.column_stack([a*np.cos(theta),b*np.sin(theta)]);return np.linalg.norm(p-q,axis=1)
def main():
    out={}
    for case in ['T1','S1']:
        path=OUT/'T1_exact_mesh.npz' if case=='T1' else OLD/'validation_and_cost/shape_references/tsr_square_g3_mesh.npz'
        with np.load(path) as z:xy=z['xy'];con=z['connectivity'] if case=='T1' else z['connectivity']-1
        edgespec=[(0,1,3),(1,2,4),(2,0,5)] if case=='T1' else [(0,1,4),(1,2,5),(2,3,6),(3,0,7)]
        edges=np.concatenate([con[:,[a,b,m]] for a,b,m in edgespec]);pair=np.sort(edges[:,:2],axis=1);_,first,counts=np.unique(pair,axis=0,return_index=True,return_counts=True);boundary=edges[first[counts==1]];coords=xy[boundary];outer=(np.isclose(abs(coords),.5,atol=1e-12,rtol=0).all(1)).any(1);boundary=boundary[~outer];coords=xy[boundary]
        t=np.linspace(0,1,33);N=np.column_stack([(1-t)*(1-2*t),t*(2*t-1),4*t*(1-t)]);samples=np.einsum('qi,eij->eqj',N,coords);flat=samples.reshape(-1,2)
        domain,_,_=normalized_domain(read(ROOT/'inputs/cavity_geometries.json')[case]);ds=np.stack([distance(flat,h) for h in domain.holes]);d=ds.min(0);label=ds.argmin(0)
        maxgap=float(d.max());precision_floor=4*np.finfo(np.float32).eps*float(abs(xy).max()) if case=='T1' else 4*np.finfo(np.float64).eps*float(abs(xy).max())
        offset=max(1e-8,4*maxgap,precision_floor)
        out[case]=dict(mesh_source=str(path.resolve()),mesh_sha256=sha(path),quadratic_hole_edges=len(boundary),samples_per_edge=33,max_sampled_boundary_gap=maxgap,per_hole_max_gap={h['tag']:float(d[label==i].max()) for i,h in enumerate(domain.holes)},coordinate_precision_floor=precision_floor,proposed_solid_offset=offset,rule='max(1e-8, 4*max_sampled_boundary_gap, 4*coordinate_epsilon*max_abs_normalized_coordinate); 33 equally spaced samples per quadratic hole edge; coordinate epsilon is float32 for T1 meshing/export and float64 for exact S1 generator')
        np.savez_compressed(OUT/f'{case}_boundary_geometry.npz',edges=boundary,edge_samples=samples,distance_to_analytic=d.reshape(len(boundary),33),hole_index=label.reshape(len(boundary),33));print(case,out[case],flush=True)
    (OUT/'boundary_geometry.json').write_text(json.dumps(dict(source_code_sha256=sha(__file__),models_read=False,cases=out),indent=2),encoding='utf-8')
if __name__=='__main__':main()
