"""Abaqus/CAE: reproducible geometry, meshes and independent pressure steps.

Run from the Abaqus MCP workspace. geometry.json resides beside this script.
Produces inputs only; analysis submission is a separate recorded operation.
"""
from abaqus import mdb, Mdb
from abaqusConstants import *
from caeModules import *
import mesh, regionToolset
import json, math, os

BASE = r'D:\Abaqus\_projects'
OUT = os.path.join(BASE,'jobs','tust_revision3')
os.makedirs(OUT,exist_ok=True)
os.chdir(OUT)
with open(os.path.join(BASE,'scripts','tust_revision3','geometry.json')) as f:
    GEO=json.load(f)
Mdb()
records=[]
previous_path=os.path.join(OUT,'mesh_manifest.json')
previous=json.load(open(previous_path)) if os.path.exists(previous_path) else []

def create(kind,h,quadratic=False):
    code=('q' if quadratic else 'l')+str(h).replace('.','p')
    if kind=='circle':code='s'+code
    if kind=='tunnel' and quadratic and h==.25:code='a'+code
    if kind=='tunnel' and quadratic and h<.25:code='t'+code
    name='tr3_'+kind+'_'+code
    existing=[r for r in previous if r['job']==name]
    if existing and os.path.exists(os.path.join(OUT,name+'.odb')):
        records.append(existing[0]);return
    m=mdb.Model(name=name)
    size=.5 if kind=='circle' else 37.5
    sk=m.ConstrainedSketch(name='profile',sheetSize=5*size)
    sk.rectangle(point1=(-size,-size),point2=(size,size))
    if kind=='circle':
        sk.CircleByCenterPerimeter(center=(0.,0.),point1=(.1,0.))
        E,nu=1.333,.3333
    else:
        pts=[(75*p[0],75*p[1]) for p in GEO['tunnel_vertices_normalized']]
        for a,b in zip(pts[:-1],pts[1:]):sk.Line(point1=a,point2=b)
        th=-math.pi/4
        sk.EllipseByCenterPerimeter(center=(15.,15.),
            axisPoint1=(15+7.5*math.cos(th),15+7.5*math.sin(th)),
            axisPoint2=(15+3.75*math.cos(th-math.pi/2),15+3.75*math.sin(th-math.pi/2)))
        E,nu=10000.,.26
    p=m.Part(name='Rock',dimensionality=TWO_D_PLANAR,type=DEFORMABLE_BODY)
    p.BaseShell(sketch=sk)
    # Exactly the original bottom-centre horizontal rigid-mode constraint.
    if kind=='circle':
        partition=m.ConstrainedSketch(name='radial_partition',sheetSize=5*size)
        for k in range(8):
            angle=k*math.pi/4
            ux,uy=math.cos(angle),math.sin(angle)
            outer=size/max(abs(ux),abs(uy))
            partition.Line(point1=(.1*ux,.1*uy),point2=(outer*ux,outer*uy))
        p.PartitionFaceBySketch(faces=p.faces[:],sketch=partition)
    else:
        p.PartitionEdgeByParam(edges=p.edges.findAt(((0.,-size,0.),)),parameter=.5)
    mat=m.Material(name='RockMaterial');mat.Elastic(table=((E,nu),))
    m.HomogeneousSolidSection(name='Solid',material='RockMaterial',thickness=None)
    p.SectionAssignment(region=regionToolset.Region(faces=p.faces[:]),sectionName='Solid')
    p.setMeshControls(regions=p.faces[:],elemShape=QUAD,technique=STRUCTURED if kind=='circle' else FREE)
    if kind=='tunnel' and quadratic and h==.25:
        p.setMeshControls(regions=p.faces[:],elemShape=QUAD_DOMINATED,technique=FREE,algorithm=ADVANCING_FRONT)
    if kind=='tunnel' and quadratic and h<.25:
        p.setMeshControls(regions=p.faces[:],elemShape=TRI,technique=FREE)
    types=(mesh.ElemType(elemCode=CPE8R if quadratic else CPE4R,elemLibrary=STANDARD),
           mesh.ElemType(elemCode=CPE6 if quadratic else CPE3,elemLibrary=STANDARD))
    p.setElementType(regions=(p.faces[:],),elemTypes=types)
    p.seedPart(size=h,deviationFactor=.1,minSizeFactor=.1)
    p.generateMesh()
    quality=p.verifyMeshQuality(criterion=ANALYSIS_CHECKS)
    failed=quality.get('failedElements',[])
    assert not len(failed),(name,'Failed mesh quality',len(failed))
    a=m.rootAssembly
    inst=a.Instance(name='ROCK-1',part=p,dependent=ON)
    bottom=inst.edges.findAt(((-size/2,-size,0.),),((size/2,-size,0.),))
    centre=inst.vertices.findAt(((0.,-size,0.),))
    m.DisplacementBC(name='BottomUy',createStepName='Initial',region=a.Set(name='Bottom',edges=bottom),u2=0.)
    m.DisplacementBC(name='PinUx',createStepName='Initial',region=a.Set(name='Pin',vertices=centre),u1=0.)
    tol=size*1e-6
    left=inst.edges.getByBoundingBox(xMin=-size-tol,xMax=-size+tol,yMin=-size-tol,yMax=size+tol)
    right=inst.edges.getByBoundingBox(xMin=size-tol,xMax=size+tol,yMin=-size-tol,yMax=size+tol)
    side=left+right
    top=inst.edges.getByBoundingBox(xMin=-size-tol,xMax=size+tol,yMin=size-tol,yMax=size+tol)
    a.Surface(name='Sides',side1Edges=side);a.Surface(name='Top',side1Edges=top)
    if kind=='circle':
        m.StaticStep(name='UnitL',previous='Initial',nlgeom=OFF)
        m.StaticStep(name='UnitT',previous='UnitL',nlgeom=OFF)
        m.StaticStep(name='Check',previous='UnitT',nlgeom=OFF)
        m.Pressure(name='PL',createStepName='UnitL',region=a.surfaces['Sides'],magnitude=1.)
        m.Pressure(name='PT',createStepName='UnitT',region=a.surfaces['Top'],magnitude=1.)
        m.loads['PL'].setValuesInStep(stepName='UnitT',magnitude=0.)
        m.loads['PL'].setValuesInStep(stepName='Check',magnitude=1.)
        m.loads['PT'].setValuesInStep(stepName='Check',magnitude=5.)
    else:
        m.StaticStep(name='Load',previous='Initial',nlgeom=OFF)
        m.Pressure(name='PL',createStepName='Load',region=a.surfaces['Sides'],magnitude=10.)
        m.Pressure(name='PT',createStepName='Load',region=a.surfaces['Top'],magnitude=10.)
        th=-math.pi/4
        ep=(15+7.5*math.cos(th),15+7.5*math.sin(th),0.)
        a.Surface(name='Water',side1Edges=inst.edges.findAt((ep,)))
        m.Pressure(name='PW',createStepName='Load',region=a.surfaces['Water'],magnitude=2.5)
    firststep='UnitL' if kind=='circle' else 'Load'
    if 'F-Output-1' in m.fieldOutputRequests:
        m.fieldOutputRequests['F-Output-1'].setValues(variables=('S','U','RF','COORD','IVOL'),frequency=LAST_INCREMENT)
    else:
        m.FieldOutputRequest(name='F-Output-1',createStepName=firststep,variables=('S','U','RF','COORD','IVOL'),frequency=LAST_INCREMENT)
    j=mdb.Job(name=name,model=name,numCpus=4,numDomains=4,memory=80,memoryUnits=PERCENTAGE,
              nodalOutputPrecision=FULL,description='TUST revision 3: linear plane strain, verified source geometry')
    j.writeInput(consistencyChecking=ON)
    path=os.path.join(OUT,name+'.inp')
    records.append(dict(job=name,geometry=kind,h=h,quadratic=quadratic,
                        nodes=len(p.nodes),elements=len(p.elements),input_file=path,
                        failed_quality_elements=len(failed)))
    print(json.dumps(records[-1]),flush=True)

for kind,hs,qh in [('circle',[.01,.005,.0025],.005),('tunnel',[1.,.5,.25],.5)]:
    for h in hs:create(kind,h)
    for h in hs:create(kind,h,True)
    if kind=='tunnel':create(kind,.125,True)
mdb.saveAs(pathName=os.path.join(OUT,'tust_revision3_structured.cae'))
with open(os.path.join(OUT,'mesh_manifest.json'),'w') as f:json.dump(records,f,indent=2)
print('BUILD_COMPLETE',flush=True)
