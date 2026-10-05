"""Freeze stage-3 inputs without executing legacy training scripts."""
from pathlib import Path
import ast, hashlib, json, platform, os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import torch
import numpy as np
from matplotlib.path import Path as Polygon

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def literals(p, names):
    values = {}
    for node in ast.parse(p.read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names and target.id not in values:
                    values[target.id] = ast.literal_eval(node.value)
    return values

def segment_distance(points, polygon):
    result = np.full(len(points), np.inf)
    for a,b in zip(polygon[:-1],polygon[1:]):
        v=b-a
        t=np.clip((points-a)@v/(v@v),0,1)
        result=np.minimum(result,np.linalg.norm(points-a-t[:,None]*v,axis=1))
    return result

def main():
    for d in ('config','fem','runs','checks','summary','figures'):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    source=PROJECT/'代码源文件/【隧道+溶洞】'
    raw=literals(source/'高级PINN - 隧道散点双向求导.py',{'_x_raw','_y_raw'})
    fe=literals(source/'py - abaqus.py',{'x_coords','y_coords'})
    polygon=np.column_stack([raw['_x_raw'],raw['_y_raw']])
    assert np.array_equal(polygon,np.column_stack([fe['x_coords'],fe['y_coords']]))
    points=np.loadtxt(source/'displacement-karst.csv',delimiter=',',skiprows=1)[:,:2]
    # The legacy file is in physical metres; establish this explicitly.
    assert np.max(np.abs(points))>30
    pphys=polygon*75
    d_full=segment_distance(points,pphys)
    d_half=segment_distance(points,pphys/2)
    full_boundary=points[d_full<1e-4]
    in_full=Polygon(pphys).contains_points(points)
    in_half=Polygon(pphys/2).contains_points(points)
    signed_area=float(np.sum(polygon[:-1,0]*polygon[1:,1]-polygon[1:,0]*polygon[:-1,1])/2)
    checks={
        'tunnel_full_dimensions_m':np.ptp(pphys,axis=0).tolist(),
        'pinn_abaqus_vertices_identical':True,'polygon_signed_area_normalized':signed_area,
        'reference_nodes':len(points),'reference_extent_m':[points.min(0).tolist(),points.max(0).tolist()],
        'reference_nodes_on_full_boundary_tol_1e_4_m':len(full_boundary),
        'reference_nodes_on_half_boundary_tol_1e_4_m':int((d_half<1e-4).sum()),
        'reference_nodes_strictly_between_half_and_full_boundary':int((in_full&~in_half&(d_full>1e-4)).sum()),
        'reference_full_boundary_extent_m':[full_boundary.min(0).tolist(),full_boundary.max(0).tolist()],
        'bias_variance_box':2*20**2*.5**2/9,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                       'device':torch.cuda.get_device_name(0),'numpy':np.__version__},
        'inputs':{str(p.relative_to(PROJECT)):digest(p) for p in (
            PROJECT/'再次提交TUST/manuscript.docx',PROJECT/'再次提交TUST/TUST-D-26-02864.pdf',
            source/'高级PINN - 隧道散点双向求导.py',source/'py - abaqus.py',
            source/'displacement-karst.csv',source/'stress-karst.csv',
            PROJECT/'代码源文件/审稿修改-基线对比/benchmark_suite.py')}
    }
    (ROOT/'checks/input_audit.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
    geometry={'tunnel_vertices_normalized':polygon.tolist(),'length_scale_m':75.,'stress_scale_MPa':2.5,
              'modulus_scale_MPa':10000.,'displacement_scale_m':.01875,'E_MPa':10000.,'nu':.26,
              'ellipse_center_m':[15.,15.],'ellipse_semiaxes_m':[7.5,3.75],'ellipse_angle_deg':-45.,
              'outer_halfwidth_m':37.5,'side_pressure_MPa':10.,'top_pressure_MPa':10.,'water_pressure_MPa':2.5}
    (ROOT/'config/geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
    cases=[('C1',-1.,-5.),('C2',-2.,-4.),('C3',-8/3,-10/3),('C4',-3.,-3.),
           ('C5',-10/3,-8/3),('C6',-4.,-2.),('C7',-5.,-1.),('C8',-4.,-4.)]
    runs=[]
    for case,p,pt in cases:
        for seed in range(41,49 if case in ('C1','C8') else 46):
            methods=['vanilla','anchored','fourier','xpinn','dem']
            if case in ('C1','C8'):
                methods+=['independent_gaussian','independent_marginal','vanilla_matched','fourier_half','fourier_double']
            for method in methods:
                runs.append(dict(id=f'{case}_{method}_s{seed}',geometry='circle',case=case,seed=seed,
                                 method=method,p_lateral=p,p_top=pt,max_iter=1000 if method=='dem' else 2000))
    for seed in range(41,49):
        for method in ('anchored','fourier','vanilla_matched'):
            runs.append(dict(id=f'T1_{method}_s{seed}',geometry='tunnel',case='T1',seed=seed,method=method,
                             p_lateral=-4.,p_top=-4.,p_water=-1.,max_iter=10000))
    assert len(runs)==334
    # Deterministic order balances methods over time; never selected from outcomes.
    order=np.random.default_rng(20260920).permutation(len(runs))
    runs=[runs[i] for i in order]
    (ROOT/'config/run_manifest.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k!='inputs'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
