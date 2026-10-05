"""Mesh/order convergence on common physical probes; never compare node indices."""
import runtime
from models import ROOT,in_rock,POLYGON,ellipse_points
from fem_interpolation import Field
from metrics import relative
import numpy as np,json,csv

def probes(geometry):
    if geometry=='circle':
        g=np.linspace(-.495,.495,101)
        gx,gy=np.meshgrid(g,g)
        xy=np.column_stack([gx.ravel(),gy.ravel()]);xy=xy[np.linalg.norm(xy,axis=1)>.105]
        theta=np.arange(360)*2*np.pi/360
        paths=np.concatenate([np.column_stack([r*np.cos(theta),r*np.sin(theta)]) for r in [.105,.11,.125,.15,.2]])
        return xy,paths
    g=np.linspace(-37.25,37.25,101);gx,gy=np.meshgrid(g,g)
    xy=np.column_stack([gx.ravel(),gy.ravel()]);xy=xy[in_rock(xy/75,'tunnel')]
    # Polygon tangent bisector points are offset 0.5 m into the rock.
    p=POLYGON[:-1]*75
    tangent=np.roll(p,-1,0)-np.roll(p,1,0)
    outward=np.column_stack([-tangent[:,1],tangent[:,0]])
    outward/=np.linalg.norm(outward,axis=1)[:,None]
    tpath=p+.5*outward
    ep,en=ellipse_points(361);ep=ep[:-1]*75-.5*en[:-1]
    direction=np.array([1.,1.])/np.sqrt(2.)
    exits=[]
    for a,b in zip(p,np.roll(p,-1,0)):
        matrix=np.column_stack([direction,-(b-a)])
        if abs(np.linalg.det(matrix))<1e-12:continue
        t,q=np.linalg.solve(matrix,a)
        if t>0 and 0<=q<=1:exits.append(t)
    tunnel_exit=min(exits)
    ellipse_entry=np.linalg.norm([15.,15.])-3.75
    bridge=np.linspace(tunnel_exit+.5,ellipse_entry-.5,51)[:,None]*direction
    assert in_rock(bridge/75,'tunnel').all()
    return xy,np.vstack([tpath,ep,bridge])

def main():
    manifest=json.loads((ROOT/'fem/mesh_manifest.json').read_text())
    records=[];superposition=[];reconstruction=[]
    for geometry in ['circle','tunnel']:
        xy,path=probes(geometry);np.savez_compressed(ROOT/f'fem/{geometry}_common_probes.npz',field=xy,paths=path)
        refname='tr3_circle_sq0p0025_Check' if geometry=='circle' else 'tr3_tunnel_tq0p125_Load'
        reference=dict(np.load(ROOT/'fem'/f'{refname}.npz'))
        E,nu=(1.333,.3333) if geometry=='circle' else (10000.,.26)
        ref=Field(reference,E,nu)
        refu,refs,_=ref.evaluate(xy);pu,ps,_=ref.evaluate(path)
        for row in manifest:
            if row['geometry']!=geometry:continue
            step='Check' if geometry=='circle' else 'Load'
            data=dict(np.load(ROOT/'fem'/f"{row['job']}_{step}.npz"))
            field=Field(data,E,nu)
            u,s,_=field.evaluate(xy);up,sp,_=field.evaluate(path)
            record={**{k:row[k] for k in ('job','geometry','h','quadratic','nodes','elements')},
                'u_common_pct':relative(refu,u),'s_common_pct':relative(refs,s),
                'u_path_pct':relative(pu,up),'s_path_pct':relative(ps,sp)}
            records.append(record);print(json.dumps(record),flush=True)
            selection=np.linspace(0,len(data['s'])-1,min(3000,len(data['s'])),dtype=int)
            _,reconstructed,_=field.evaluate(data['xy_s'][selection])
            reconstruction.append({'job':row['job'],'s_reconstruction_pct':relative(data['s'][selection],reconstructed)})
            if geometry=='circle':
                first=np.load(ROOT/'fem'/f"{row['job']}_UnitL.npz")
                second=np.load(ROOT/'fem'/f"{row['job']}_UnitT.npz")
                assert np.array_equal(first['xy_u'],second['xy_u']) and np.array_equal(first['xy_s'],second['xy_s'])
                superposition.append({'job':row['job'],'u_pct':relative(data['u'],first['u']+5*second['u']),
                    's_pct':relative(data['s'],first['s']+5*second['s'])})
        np.savez_compressed(ROOT/f'fem/{geometry}_probe_reference.npz',xy=xy,u=refu,s=refs,path=path,path_u=pu,path_s=ps)
    report={'mesh_order_comparison':records,'linear_superposition':superposition,'stress_reconstruction_check':reconstruction,
        'note':'Common-probe stress is computed from FE displacement gradients; primary PINN errors use exported integration-point stresses.'}
    (ROOT/'checks/fem_convergence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (ROOT/'summary/fem_convergence.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    print(json.dumps({'superposition':superposition,'reconstruction':reconstruction},indent=2))

if __name__=='__main__':main()
