"""Boundary-matched Kirsch auxiliary check, separate from submitted BCs.

Itasca's plane-strain hole solution gives stress and excavation displacement.
Add homogeneous far-field displacement to obtain total displacement; prescribe
that exact total field on the finite square boundary. No remote-boundary or
bottom-support approximation is then compared against an infinite-domain field.
Source: https://docs.itascacg.com/itasca900/3dec/docproject/source/examples/CylindricalHoleInAnIEM.html
"""
import runtime
import numpy as np,json,hashlib
from models import ROOT
from pathlib import Path

def exact(xy,p=-1.,q=-5.,E=1.333,nu=.3333,a=.1):
    x,y=xy[:,0],xy[:,1];r=np.sqrt(x*x+y*y);c=x/r;s=y/r
    c2=c*c-s*s;s2=2*c*s;t=a*a/(r*r)
    mean=(p+q)/2;deviator=(p-q)/2;mu=E/(2*(1+nu));kappa=3-4*nu
    sr=mean*(1-t)+deviator*(1-4*t+3*t*t)*c2
    st=mean*(1+t)-deviator*(1+3*t*t)*c2
    tau=-deviator*(1+2*t-3*t*t)*s2
    stress=np.column_stack([sr*c*c+st*s*s-2*tau*c*s,
                            sr*s*s+st*c*c+2*tau*c*s,(sr-st)*c*s+tau*(c*c-s*s)])
    ur=(mean*((kappa-1)*r/2+a*a/r)+deviator*(r+(kappa+1)*a*a/r-a**4/r**3)*c2)/(2*mu)
    ut=-deviator*(r+(kappa-1)*a*a/r+a**4/r**3)*s2/(2*mu)
    return np.column_stack([ur*c-ut*s,ur*s+ut*c]),stress

def verify():
    rng=np.random.default_rng(20260920);xy=rng.uniform(-.5,.5,(500,2));xy=xy[np.linalg.norm(xy,axis=1)>.105]
    u,s=exact(xy);grad=[];div=[]
    for axis in [0,1]:
        z=xy.astype(complex);z[:,axis]+=1e-25j
        uc,sc=exact(z);grad.append(uc.imag/1e-25);div.append(sc.imag/1e-25)
    lam=1.333*.3333/((1+.3333)*(1-2*.3333));mu=1.333/(2*(1+.3333))
    exx,eyy=grad[0][:,0],grad[1][:,1];tr=exx+eyy
    constitutive=np.column_stack([lam*tr+2*mu*exx,lam*tr+2*mu*eyy,mu*(grad[1][:,0]+grad[0][:,1])])
    equilibrium=np.column_stack([div[0][:,0]+div[1][:,2],div[0][:,2]+div[1][:,1]])
    th=np.arange(360)*2*np.pi/360;normal=np.column_stack([np.cos(th),np.sin(th)])
    _,hole=exact(.1*normal)
    traction=np.column_stack([hole[:,0]*normal[:,0]+hole[:,2]*normal[:,1],hole[:,2]*normal[:,0]+hole[:,1]*normal[:,1]])
    checks={'max_constitutive_residual':float(np.abs(constitutive-s).max()),
            'max_equilibrium_residual':float(np.abs(equilibrium).max()),'max_hole_traction':float(np.abs(traction).max())}
    assert max(checks.values())<1e-10,checks
    (ROOT/'checks/kirsch_formula.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    return checks

def build():
    print(verify())
    jobdir=Path(r'D:\Abaqus\_projects\jobs\tust_revision3')
    manifest=[]
    for code,h in [('0p01',.01),('0p005',.005)]:
        source='tr3_circle_sq'+code;name='tr3_kirsch_q'+code
        text=(jobdir/(source+'.inp')).read_text()
        before=text.split('*Step,',1)[0]
        lines=[];skip=False
        for line in before.splitlines():
            if line.lower().startswith('*boundary'):
                skip=True;continue
            if line.startswith('*') and not line.startswith('**'):skip=False
            if not skip:lines.append(line)
        coordinates={};in_nodes=False
        for line in before.splitlines():
            if line.lower()=='*node':in_nodes=True;continue
            if line.startswith('*'):in_nodes=False
            if in_nodes:
                tokens=line.split(',');coordinates[int(tokens[0])]=[float(tokens[1]),float(tokens[2])]
        ids=np.array(list(coordinates));xy=np.array(list(coordinates.values()))
        selected=np.max(np.abs(xy),axis=1)>.5-1e-10
        u,_=exact(xy[selected])
        lines+=['*Step, name=Kirsch, nlgeom=NO','*Static','1., 1., 1e-05, 1.','*Boundary']
        for label,uv in zip(ids[selected],u):
            for component,value in enumerate(uv,1):lines.append(f'ROCK-1.{label}, {component}, {component}, {value:.16g}')
        lines+=['*Output, field, frequency=99999','*Node Output','COORD, RF, U','*Element Output, directions=YES','COORD, IVOL, S','*End Step']
        path=jobdir/(name+'.inp');path.write_text('\n'.join(lines)+'\n')
        meta=json.loads((ROOT/'fem'/f'{source}.json').read_text())
        manifest.append({'job':name,'geometry':'kirsch','h':h,'quadratic':True,'nodes':meta['nodes'],
                         'elements':meta['elements'],'input_file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (jobdir/'kirsch_manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'fem/kirsch_manifest.json').write_text(json.dumps(manifest,indent=2))

def compare():
    from metrics import relative
    out=[]
    for row in json.loads((ROOT/'fem/kirsch_manifest.json').read_text()):
        d=np.load(ROOT/'fem'/f"{row['job']}_Kirsch.npz")
        u,_=exact(d['xy_u']);_,s=exact(d['xy_s']);near=np.linalg.norm(d['xy_s'],axis=1)<=.2
        out.append({'job':row['job'],'h':row['h'],'u_vector_pct':relative(u,d['u']),
                    's_vector_pct':relative(s,d['s']),'s_near_pct':relative(s[near],d['s'][near])})
    (ROOT/'checks/kirsch_comparison.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    import sys
    compare() if '--compare' in sys.argv else build()
