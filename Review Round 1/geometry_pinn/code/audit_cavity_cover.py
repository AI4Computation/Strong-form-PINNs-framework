"""Registered P1 geometry audit. All output is confined to the round-two tree."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import json, hashlib, copy, sys
from datetime import datetime
import numpy as np
from cavity_cover import normalized_domain, construct, boundary_samples, windows, support_components

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT/'geometry_pinn/protocols/R2_P1_geometry.json'
DATA = ROOT/'geometry_pinn/inputs/cavity_geometries.json'
OUT = ROOT/'geometry_pinn/results/R2_P1'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'This result directory is immutable once created.'
    OUT.mkdir()
    p = json.loads(PROTO.read_text(encoding='utf-8'))
    cases = json.loads(DATA.read_text(encoding='utf-8'))
    result = dict(id=p['id'], protocol_sha256=digest(PROTO), input_sha256=digest(DATA),
                  source_sha256={f.name:digest(f) for f in [Path(__file__),Path(__file__).with_name('cavity_cover.py'),Path(__file__).with_name('geometry_primitives.py')]},
                  python=sys.executable, cases={}, fem_access=False, training_runs=0, timing_valid_for_comparison=False)
    for case, data in cases.items():
        domain,_,_ = normalized_domain(data)
        cover = construct(domain,p)
        c,h = cover['centres'],cover['halfwidths']
        rng = np.random.default_rng(p['checks']['interior_seed'])
        interior = domain.random_interior(p['checks']['interior_points'],rng)
        boundary = boundary_samples(domain,p['checks']['boundary_audit_spacing_over_outer_length'],False)
        epsilon = 1e-7
        near = boundary['x'] - epsilon*boundary['normal']
        vertices = np.asarray(domain.corners)
        probes = np.vstack([interior,boundary['x'],near,vertices])
        minima=[]; partition_error=[]; overlap=[]
        for chunk in np.array_split(probes,max(1,int(np.ceil(len(probes)/512)))):
            w,tot,n = windows(chunk,c,h)
            minima.append(float(tot.min())); partition_error.append(float(np.max(np.abs(w.sum(axis=1)-1))));overlap.append(int(n.max()))
        components = support_components(domain,c,h)
        b,n = cover['boundary'],cover['normal']
        gap,chord = cover['solid_gap'],cover['void_chord']
        ray_solid=[];ray_void=[]
        for fraction in p['checks']['ray_steps_for_medium_check']:
            ray_solid.append(int(np.count_nonzero(~domain.contains(b-fraction*gap[:,None]*n))))
            ray_void.append(int(np.count_nonzero(domain.contains(b+fraction*chord[:,None]*n))))
        moved=copy.deepcopy(data)
        scale=p['checks']['scale_translation_factor'];shift=np.asarray(p['checks']['scale_translation_shift'])
        moved['outer']=(np.asarray(moved['outer'])*scale+shift).tolist()
        for hole in moved['holes']:
            if hole['kind']=='polygon':hole['vertices']=(np.asarray(hole['vertices'])*scale+shift).tolist()
            else:
                hole['center']=(np.asarray(hole['center'])*scale+shift).tolist();hole['axes']=(np.asarray(hole['axes'])*scale).tolist()
        shifted,_,_=normalized_domain(moved);other=construct(shifted,p)
        invariant = c.shape==other['centres'].shape and np.allclose(c,other['centres'],atol=1e-12,rtol=0) and np.allclose(h,other['halfwidths'],atol=1e-12,rtol=0)
        # Points strictly inside rock; independent spatial finite-difference derivative check.
        points=interior[:31];w,g=windows(points,c,h,True);fd=np.empty_like(g);step=1e-7
        for axis in range(2):
            delta=np.zeros(2);delta[axis]=step
            fd[:,:,axis]=(windows(points+delta,c,h)[0]-windows(points-delta,c,h)[0])/(2*step)
        error=float(np.max(np.abs(g-fd)))
        checks={
            'rays_finite_positive':bool(np.all(np.isfinite(gap)) and np.all(np.isfinite(chord)) and min(gap.min(),chord.min())>0),
            'rock_normal_segments_solid':sum(ray_solid)==0,
            'void_normal_segments_void':sum(ray_void)==0,
            'near_wall_normals_into_solid':bool(np.all(domain.contains(near))),
            'coverage':min(minima)>=p['checks']['raw_weight_sum_minimum'],
            'partition_identity':max(partition_error)<=p['checks']['partition_error_maximum'],
            'connected_support_two_grids':bool(np.all(components==1)),
            'patch_budget':len(c)<=p['checks']['max_patch_count'],
            'overlap_budget':max(overlap)<=p['checks']['max_overlap_count'],
            'scale_translation_equivariance':bool(invariant),
            'window_derivative':error<=1e-5,
            'gradient_partition_identity':float(np.max(np.abs(g.sum(axis=1))))<1e-10,
        }
        np.savez_compressed(OUT/f'{case}_cover.npz',**cover,support_components=components,interior_probes=interior,boundary_probes=boundary['x'])
        row=dict(passed=all(checks.values()),checks=checks,patches=len(c),sharp_corners=len(cover['corners']),depth_counts={str(i):int(np.sum(cover['depths']==i)) for i in np.unique(cover['depths'])},
                 minimum_raw_weight_sum=min(minima),maximum_overlap=max(overlap),derivative_max_error=error,
                 minimum_solid_gap=float(gap.min()),minimum_void_chord=float(chord.min()),target_min=float(cover['target'].min()),target_max=float(cover['target'].max()),
                 support_component_failures=[np.where(x!=1)[0].tolist() for x in components],ray_solid_failures=ray_solid,ray_void_failures=ray_void,
                 probe_count=len(probes),area=domain.area,geometry_archive_sha256=digest(OUT/f'{case}_cover.npz'))
        result['cases'][case]=row
        (OUT/f'{case}_audit.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(case, json.dumps(row),flush=True)
    result['passed']=all(x['passed'] for x in result['cases'].values())
    result['completed_local']=datetime.now().astimezone().isoformat()
    result['scope']='Sampled coverage/connectivity feasibility, not a topological proof, accuracy result or method novelty claim.'
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
