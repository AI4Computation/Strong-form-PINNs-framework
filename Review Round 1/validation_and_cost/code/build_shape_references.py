"""Nested, corner-graded CPE8R reference meshes for two held-out cavity shapes.

Generates Abaqus input only. Actual analysis is submitted via the Abaqus MCP.
Same tensor-grid construction and loads for both shapes; no learned field read.
"""
from pathlib import Path
import numpy as np
import json,hashlib,sys,shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'shape_references'
WORK=Path('D:/Abaqus/_projects/jobs/tust_shape_reference_20260921')
SHAPES={'square':(.1,.1),'slender':(.2,.015)}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def axis(a,level):
    breaks=[-.5,-a,a,.5];pieces=[]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        n=max(8,int(np.ceil((hi-lo)/.02)))
        n+=n%2;n*=2**level
        t=np.arange(n+1)/n
        x=lo+(hi-lo)*.5*(1-np.cos(np.pi*t))
        x[0]=lo;x[-1]=hi
        pieces.append(x if not pieces else x[1:])
    x=np.concatenate(pieces);x[np.abs(x)<1e-15]=0.
    assert np.min(np.diff(x))>0
    return x


def generate(name,level):
    a,b=SHAPES[name];x,y=axis(a,level),axis(b,level)
    nodes=[];index={};elements=[];grid_ids=np.zeros((len(x)-1,len(y)-1),dtype=np.int32)
    edges={k:[] for k in ['LEFT','RIGHT','TOP']}
    def node(i,j):
        key=(i,j)
        if key not in index:
            xx=x[i//2] if i%2==0 else .5*(x[i//2]+x[i//2+1])
            yy=y[j//2] if j%2==0 else .5*(y[j//2]+y[j//2+1])
            index[key]=len(nodes)+1;nodes.append((xx,yy))
        return index[key]
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            if -a<(x[i]+x[i+1])*.5<a and -b<(y[j]+y[j+1])*.5<b:continue
            con=[node(2*i,2*j),node(2*i+2,2*j),node(2*i+2,2*j+2),node(2*i,2*j+2),
                 node(2*i+1,2*j),node(2*i+2,2*j+1),node(2*i+1,2*j+2),node(2*i,2*j+1)]
            elements.append(con);eid=len(elements);grid_ids[i,j]=eid
            if i==0:edges['LEFT'].append(eid)
            if i==len(x)-2:edges['RIGHT'].append(eid)
            if j==len(y)-2:edges['TOP'].append(eid)
    xy=np.asarray(nodes);con=np.asarray(elements,dtype=np.int32)
    area=float(sum((x[i+1]-x[i])*(y[j+1]-y[j]) for i,j in zip(*np.nonzero(grid_ids))))
    assert abs(area-(1-4*a*b))<1e-10
    bottom=np.flatnonzero(xy[:,1]==-.5)+1
    pin=np.flatnonzero((xy[:,0]==0)&(xy[:,1]==-.5))+1;assert len(pin)==1
    aspect=np.array([max((x[i+1]-x[i])/(y[j+1]-y[j]),(y[j+1]-y[j])/(x[i+1]-x[i])) for i,j in zip(*np.nonzero(grid_ids))])
    job=f'tsr_{name}_g{level}'
    input_path=WORK/(job+'.inp');assert not input_path.exists()
    with input_path.open('w',encoding='ascii') as f:
        f.write('*Heading\nTUST automatic shape reference: linear plane strain; no learned field\n*Preprint,echo=NO,model=NO,history=NO,contact=NO\n*Part,name=Rock\n*Node\n')
        for i,(xx,yy) in enumerate(xy,1):f.write(f'{i},{xx:.17g},{yy:.17g}\n')
        f.write('*Element,type=CPE8R,elset=ALL\n')
        for i,row in enumerate(con,1):f.write(str(i)+','+','.join(map(str,row))+'\n')
        def aset(kind,label,ids):
            f.write(f'*{kind},{"nset" if kind=="Nset" else "elset"}={label}\n')
            for k in range(0,len(ids),16):f.write(','.join(map(str,ids[k:k+16]))+'\n')
        aset('Nset','BOTTOM',bottom);aset('Nset','PIN',pin)
        for label,side in [('LEFT','S4'),('RIGHT','S2'),('TOP','S3')]:
            aset('Elset',label+'_E',edges[label]);f.write(f'*Surface,type=ELEMENT,name={label}\n{label}_E,{side}\n')
        f.write('*Solid Section,elset=ALL,material=ROCK_MAT\n1.\n*End Part\n*Assembly,name=Assembly\n*Instance,name=ROCK-1,part=Rock\n*End Instance\n*End Assembly\n')
        f.write('*Material,name=ROCK_MAT\n*Elastic\n1.333,0.3333\n*Boundary\nROCK-1.BOTTOM,2,2,0.\nROCK-1.PIN,1,1,0.\n')
        f.write('*Step,name=Load,nlgeom=NO\n*Static\n1.,1.,1e-05,1.\n*Dsload\nROCK-1.LEFT,P,1.\nROCK-1.RIGHT,P,1.\nROCK-1.TOP,P,5.\n')
        f.write('*Output,field,frequency=1\n*Node Output\nU,RF,COORD\n*Element Output\nS,IVOL,COORD\n*Output,history,frequency=1\n*Energy Output\nALLSE,ALLWK\n*End Step\n')
    mesh=BATCH/(job+'_mesh.npz')
    np.savez_compressed(mesh,x=x,y=y,grid_element_ids=grid_ids,xy=xy,connectivity=con)
    return dict(job=job,geometry=name,level=level,quadratic=True,element_type='CPE8R',nodes=len(xy),elements=len(con),
        area=area,minimum_edge=float(min(np.diff(x).min(),np.diff(y).min())),maximum_aspect_ratio=float(aspect.max()),
        input_file=str(input_path),input_sha256=sha(input_path),mesh_file=str(mesh),mesh_sha256=sha(mesh))


def main():
    assert not BATCH.exists(),'Existing reference batch must not be overwritten.'
    BATCH.mkdir();WORK.mkdir(parents=True,exist_ok=False)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),geometries=SHAPES,levels=[0,1,2,3],
        material=dict(E=1.333,nu=.3333),loads=dict(side=1.,top=5.,cavity=0.),
        support='Bottom Uy=0; bottom-centre Ux=0; all other unlisted tractions zero.',
        analysis='Small-strain static linear isotropic plane strain, unit out-of-plane thickness, no geometric nonlinearity.',
        mesh='Same corner-graded nested tensor CPE8R rule for both axis-aligned rectangular cavities. Piecewise cosine grading between outer and cavity coordinate lines, common coarse size .02, even segment count >=8, doubled per level.',
        reference_scope='Independent Abaqus references only. No new-method square/slender accuracy claim until automatic-source and neural solves are frozen and evaluated.',
        convergence='Check reaction, integrated area, positive integration weights, stress reconstruction, common-fine-point mesh differences and strain energy; report singular-corner sensitivity, never compare peak stress for convergence.',
        selection_uses_fem=False,timing_valid_for_comparison=False,cpus=2,source_sha256=sha(__file__))
    (BATCH/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[generate(name,level) for name in SHAPES for level in range(4)]
    (BATCH/'mesh_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (WORK/'mesh_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    for name in SHAPES:
        for level in range(3):
            a=np.load(BATCH/f'tsr_{name}_g{level}_mesh.npz');b=np.load(BATCH/f'tsr_{name}_g{level+1}_mesh.npz')
            assert np.max(np.abs(a['x']-b['x'][::2]))<1e-15 and np.max(np.abs(a['y']-b['y'][::2]))<1e-15
    print(json.dumps(records,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
