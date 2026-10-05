"""Geometry-only localization diagnosis. No model checkpoint or ranking is read."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from p2_fem_interpolation import shape
from cavity_cover import normalized_domain
from transfer_mechanics import wall_points
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parents[1]/'research';OUT=ROOT/'results/R2_P2J_reference_repair'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def locate(xy,con,points,k=64):
    tree=cKDTree(xy[con[:,:6]].mean(1));_,cand=tree.query(points,k=k);found=np.zeros(len(points),bool);els=np.full(len(points),-1);minimum=np.full(len(points),np.inf);position_defect=np.full(len(points),np.inf)
    for col in range(k):
        rows=np.flatnonzero(~found)
        if not len(rows):break
        xn=xy[con[cand[rows,col],:6]];rs=np.full((len(rows),2),1/3)
        for _ in range(12):
            N,dN=shape('CPE6',rs);pos=np.einsum('ni,nij->nj',N,xn);J=np.einsum('nki,nij->nkj',dN,xn);change=np.linalg.solve(J.transpose(0,2,1),(points[rows]-pos)[...,None])[...,0];rs+=change
            if abs(change).max()<1e-11:break
        N,_=shape('CPE6',rs);defect=np.linalg.norm(np.einsum('ni,nij->nj',N,xn)-points[rows],axis=1)
        violation=np.maximum(np.maximum(-rs[:,0],-rs[:,1]),rs.sum(1)-1);better=violation<minimum[rows];minimum[rows[better]]=violation[better];position_defect[rows[better]]=defect[better]
        inside=(violation<=1e-7)&(defect<1e-7);els[rows[inside]]=cand[rows[inside],col];found[rows[inside]]=True
    return dict(found=found,element_index=els,minimum_natural_coordinate_violation=minimum,position_defect=position_defect)

def main():
    assert not OUT.exists();OUT.mkdir();source=OLD/'controlled_pinn/fem/tr3_tunnel_tq0p125_Load.npz';inp=OLD/'controlled_pinn/fem/abaqus/tr3_tunnel_tq0p125.inp'
    inv=read(ROOT.parent/'00_基线与规则/第一轮冻结清单.json');hashes={x['relative_path']:x['sha256'] for x in inv['files']}
    for f in [source,inp]:assert sha(f)==hashes[f.relative_to(OLD).as_posix()]
    suspended=ROOT/'results/R2_P2I_shape_transfer/suspension.json';s=read(suspended)
    for f,h in s['files_sha256'].items():assert sha(suspended.parent/f)==h
    for f,h in s['supporting_files_sha256'].items():assert sha(f)==h
    with np.load(source) as z:xy=z['xy_u'];con=z['connectivity'];types=z['element_types']
    assert set(types)=={'CPE6'};exact=np.full_like(xy,np.nan,dtype=float);exactcon=np.full_like(con,-1);mode=None;nodes=0;elements=0
    with inp.open(encoding='utf-8-sig') as f:
      for raw in f:
        line=raw.strip()
        if not line or line.startswith('**'):continue
        if line.startswith('*'):
            mode='node' if line.lower()=='*node' else ('element' if line.lower().startswith('*element,') else None);continue
        if mode=='node':
            v=line.split(',');i=int(v[0])-1;exact[i]=[float(v[1]),float(v[2])];nodes+=1
        elif mode=='element':
            v=[int(x.strip()) for x in line.split(',') if x.strip()];exactcon[v[0]-1,:len(v)-1]=np.array(v[1:])-1;elements+=1
    assert nodes==len(xy) and elements==len(con) and np.isfinite(exact).all() and np.array_equal(exactcon,con)
    delta=xy-exact;rounding=float(abs(delta).max());is_float32=bool(np.array_equal(exact.astype(np.float32).astype(float),xy));assert rounding<4e-6,(rounding,is_float32)
    np.savez_compressed(OUT/'T1_exact_mesh.npz',xy=exact/75,connectivity=exactcon)
    domain,_,_=normalized_domain(read(ROOT/'inputs/cavity_geometries.json')['T1']);wall,normal,w,tags,ext,en=wall_points(domain);actual=np.vstack([wall,ext]);normals=np.vstack([normal,en]);labels=np.r_[tags,np.repeat('extrema',4)];rows={};arrays=dict(actual=actual,normals=normals,tags=labels,wall_weight=w)
    for coordinate_name,coordinates in [('exported',xy/75),('exact_input',exact/75)]:
      for offset in [1e-8,2e-8]:
        result=locate(coordinates,con,actual-offset*normals);tag=f'{coordinate_name}_{offset:g}';failed=np.flatnonzero(~result['found']);rows[tag]=dict(failed_count=len(failed),failed_indices=failed.tolist(),failed_tags=sorted(set(labels[failed])),max_violation_failed=float(result['minimum_natural_coordinate_violation'][failed].max()) if len(failed) else 0.)
        arrays.update({tag+'_'+k:v for k,v in result.items()});print(tag,rows[tag],flush=True)
        if coordinate_name=='exported' and offset==1e-8:
            expanded=locate(coordinates,con,(actual-offset*normals)[failed],512);rows['expanded_512_candidates']=dict(failed_count=int((~expanded['found']).sum()),tested=len(failed));arrays.update({'expanded_'+k:v for k,v in expanded.items()})
    np.savez_compressed(OUT/'localization_arrays.npz',**arrays)
    report=dict(source_sha256={str(f.resolve()):sha(f) for f in [source,inp]},source_code_sha256=sha(__file__),suspended_batch_sha256=sha(suspended),nodes=nodes,elements=elements,max_export_input_coordinate_difference_physical=rounding,export_exactly_float32_rounded_input=is_float32,checks=rows,models_read=False,training_runs=0,formal_timing=False)
    (OUT/'diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('DIAGNOSIS_COMPLETE',is_float32,rounding,flush=True)
if __name__=='__main__':main()
