"""Freeze validated reference fields. All cases use the same fine FE mesh."""
import runtime
from models import ROOT
import numpy as np,json,hashlib,argparse
from fem_interpolation import Field
from check_fem import probes
from metrics import relative

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',choices=['circle','tunnel'],required=True)
    args=parser.parse_args()
    case_manifest=json.loads((ROOT/'config/run_manifest.json').read_text())
    target=ROOT/'fem/references';target.mkdir(exist_ok=True)
    selection_path=ROOT/'config/reference_selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {'circle':{'validated':False},'tunnel':{'validated':False}}
    if args.geometry=='circle':
        mesh='tr3_circle_sq0p0025';medium='tr3_circle_sq0p005'
        fine=[dict(np.load(ROOT/'fem'/f'{mesh}_{step}.npz')) for step in ('UnitL','UnitT')]
        mid=[dict(np.load(ROOT/'fem'/f'{medium}_{step}.npz')) for step in ('UnitL','UnitT')]
        xy,path=probes('circle');allpoints=np.vstack([xy,path]);fields={}
        for tag,data in [('fineL',fine[0]),('fineT',fine[1]),('midL',mid[0]),('midT',mid[1])]:
            u,s,_=Field(data,1.333,.3333).evaluate(allpoints);fields[tag]=(u,s)
        checks={}
        cases={c['case']:c for c in case_manifest if c['geometry']=='circle'}
        for name,c in cases.items():
            pl,pt=-c['p_lateral'],-c['p_top']
            u=pl*fine[0]['u']+pt*fine[1]['u'];s=pl*fine[0]['s']+pt*fine[1]['s']
            fu=pl*fields['fineL'][0]+pt*fields['fineT'][0];fs=pl*fields['fineL'][1]+pt*fields['fineT'][1]
            mu=pl*fields['midL'][0]+pt*fields['midT'][0];ms=pl*fields['midL'][1]+pt*fields['midT'][1]
            check={'u_common_pct':relative(fu[:len(xy)],mu[:len(xy)]),'s_common_pct':relative(fs[:len(xy)],ms[:len(xy)]),
                   'u_path_pct':relative(fu[len(xy):],mu[len(xy):]),'s_path_pct':relative(fs[len(xy):],ms[len(xy):])}
            assert check['u_common_pct']<.1 and check['s_common_pct']<.5 and check['s_path_pct']<.5
            checks[name]=check
            np.savez_compressed(target/f'{name}.npz',xy_u=fine[0]['xy_u'],u=u,xy_s=fine[0]['xy_s'],s=s)
        entry={'validated':True,'mesh':mesh,'comparison_mesh':medium,'convergence':checks,
               'uncertainty_rule':'A method difference comparable to the reference change is unresolved; use one-fifth of the observed gap as the sensitivity screen, not as a rigorous error bound.',
               'stress_source':'integration-point S11,S22,S12, no averaging or shear weighting'}
    else:
        mesh='tr3_tunnel_tq0p125';medium='tr3_tunnel_aq0p25'
        fine=dict(np.load(ROOT/'fem'/f'{mesh}_Load.npz'));mid=dict(np.load(ROOT/'fem'/f'{medium}_Load.npz'))
        xy,path=probes('tunnel');points=np.vstack([xy,path]);n=len(xy)
        fu,fs,_=Field(fine,10000.,.26).evaluate(points);mu,ms,_=Field(mid,10000.,.26).evaluate(points)
        checks={'u_common_pct':relative(fu[:n],mu[:n]),'s_common_pct':relative(fs[:n],ms[:n]),
                'u_path_pct':relative(fu[n:],mu[n:]),'s_path_pct':relative(fs[n:],ms[n:])}
        assert checks['u_common_pct']<.1 and checks['s_common_pct']<.5 and checks['s_path_pct']<.5,checks
        np.savez_compressed(target/'T1.npz',**{k:fine[k] for k in ('xy_u','u','xy_s','s')})
        entry={'validated':True,'mesh':mesh,'comparison_mesh':medium,'convergence':checks,
               'stress_source':'integration-point S11,S22,S12; fixed polygon corners retained'}
    entry['reference_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*.npz')
        if (p.stem.startswith('C') if args.geometry=='circle' else p.stem=='T1')}
    selection[args.geometry]=entry
    selection_path.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps(entry,indent=2))

if __name__=='__main__':main()
