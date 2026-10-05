"""Abaqus Python export: nodal U, integration-point S, reference coordinates.

No nodal stress averaging/extrapolation. Isoparametric coordinates are checked
against the COORD field; initial geometry is always used for PINN evaluation.
"""
from odbAccess import openOdb
from abaqusConstants import INTEGRATION_POINT, NODAL
import numpy as np
import os,json,math

BASE=r'D:\Abaqus\_projects\jobs\tust_shape_reference_full_20260921'
OUT=r'D:\Abaqus\_projects\results\tust_shape_reference_full_20260921'
os.makedirs(OUT,exist_ok=True)

def data(v):
    try:return v.dataDouble
    except Exception:return v.data

def shape(kind,ip):
    if kind=='CPE4R':return np.full(4,.25)
    if kind=='CPE3':return np.full(3,1/3.)
    if kind in ('CPE8R','CPE8'):
        g=math.sqrt(3/5.) if kind=='CPE8' else 1/math.sqrt(3.)
        coordinates=[(r,s) for s in [-g,0.,g] for r in [-g,0.,g]] if kind=='CPE8' else [(-g,-g),(g,-g),(g,g),(-g,g)]
        r,s=coordinates[ip-1]
        return np.array([-(1-r)*(1-s)*(1+r+s)/4,
            -(1+r)*(1-s)*(1-r+s)/4,-(1+r)*(1+s)*(1-r-s)/4,
            -(1-r)*(1+s)*(1+r-s)/4,(1-r*r)*(1-s)/2,
            (1+r)*(1-s*s)/2,(1-r*r)*(1+s)/2,(1-r)*(1-s*s)/2])
    if kind=='CPE6':
        r,s=[(1/6.,1/6.),(2/3.,1/6.),(1/6.,2/3.)][ip-1]
        l=1-r-s
        return np.array([l*(2*l-1),r*(2*r-1),s*(2*s-1),4*l*r,4*r*s,4*s*l])
    raise ValueError(kind)

def export(record):
    name=record['job']
    status=open(os.path.join(BASE,name+'.sta')).read()
    assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in status,name
    odb=openOdb(os.path.join(BASE,name+'.odb'),readOnly=True)
    inst=odb.rootAssembly.instances['ROCK-1']
    nodeids=np.array([n.label for n in inst.nodes],dtype=np.int32)
    nodeindex={int(n):i for i,n in enumerate(nodeids)}
    odb_xy=np.array([n.coordinates[:2] for n in inst.nodes],dtype=np.float64)
    exact_mesh=np.load(record['mesh_file'])
    assert np.array_equal(nodeids,np.arange(1,len(nodeids)+1))
    xy=exact_mesh['xy'].astype(np.float64)
    coordinate_rounding=float(np.max(np.abs(xy-odb_xy)))
    assert coordinate_rounding<5e-8
    elements={e.label:(e.type,np.array([nodeindex[n] for n in e.connectivity])) for e in inst.elements}
    eids=np.array(list(elements),dtype=np.int32)
    con=np.full((len(elements),8),-1,dtype=np.int32)
    types=[]
    for i,e in enumerate(eids):
        kind,ids=elements[int(e)];con[i,:len(ids)]=ids;types.append(kind)
    exported={**record,'steps':{},'odb_coordinate_rounding':coordinate_rounding,'element_types':{k:types.count(k) for k in set(types)}}
    for stepname,step in odb.steps.items():
        frame=step.frames[-1]
        u=np.zeros((len(xy),2))
        for value in frame.fieldOutputs['U'].getSubset(region=inst,position=NODAL).values:
            u[nodeindex[value.nodeLabel]]=data(value)[:2]
        rf=np.zeros((len(xy),2))
        for value in frame.fieldOutputs['RF'].getSubset(region=inst,position=NODAL).values:
            rf[nodeindex[value.nodeLabel]]=data(value)[:2]
        sf=frame.fieldOutputs['S'].getSubset(region=inst,position=INTEGRATION_POINT)
        labels=sf.componentLabels
        idx=[labels.index(k) for k in ('S11','S22','S12')]
        values=sf.values
        ck={(v.elementLabel,v.integrationPoint):np.array(data(v)[:2]) for v in
             frame.fieldOutputs['COORD'].getSubset(region=inst,position=INTEGRATION_POINT).values}
        # Establish Abaqus IP numbering from the actual reference COORD output,
        # independently checking every point below against shape interpolation.
        permutations={}
        for el,(kind,conn) in elements.items():
            if kind in permutations:continue
            count=9 if kind=='CPE8' else (4 if kind=='CPE8R' else (3 if kind=='CPE6' else 1))
            candidates=np.array([shape(kind,i+1)@xy[conn] for i in range(count)])
            actual=np.array([ck[(el,i+1)] for i in range(count)])
            distance=np.linalg.norm(actual[:,None,:]-candidates[None,:,:],axis=2)
            order=np.argmin(distance,axis=1)
            assert len(set(order))==count
            assert distance[np.arange(count),order].max()<max(1.,np.abs(xy).max())*2e-6
            permutations[kind]=(order+1).tolist()
        ipxy=np.zeros((len(values),2));s=np.zeros((len(values),3));ipu=np.zeros_like(ipxy)
        ipid=np.zeros((len(values),2),dtype=np.int32)
        for i,value in enumerate(values):
            el,ip=value.elementLabel,value.integrationPoint
            kind,conn=elements[el];N=shape(kind,permutations[kind][ip-1])
            ipxy[i]=N@xy[conn];ipu[i]=N@u[conn]
            s[i]=np.asarray(data(value))[idx];ipid[i]=el,ip
        coords=np.array([ck[tuple(key)] for key in ipid])
        err_initial=float(np.max(np.linalg.norm(coords-ipxy,axis=1)))
        err_deformed=float(np.max(np.linalg.norm(coords-ipxy-ipu,axis=1)))
        # COORD may be output in reference or current configuration. Either must
        # agree with the independently reconstructed isoparametric coordinates.
        assert min(err_initial,err_deformed)<max(1.,np.max(np.abs(xy)))*2e-6,(name,stepname,err_initial,err_deformed)
        iv={(v.elementLabel,v.integrationPoint):float(data(v)) for v in
            frame.fieldOutputs['IVOL'].getSubset(region=inst,position=INTEGRATION_POINT).values}
        volume=np.array([iv[tuple(key)] for key in ipid])
        assert (volume>0).all()
        outfile=name+'_'+stepname+'.npz'
        np.savez_compressed(os.path.join(OUT,outfile),xy_u=xy,u=u,xy_s=ipxy,s=s,ip_id=ipid,
                            volume=volume,node_ids=nodeids,element_ids=eids,connectivity=con,
                            element_types=np.array(types),rf=rf)
        exported['steps'][stepname]={'file':outfile,'integration_points':len(s),'area':float(volume.sum()),
            'coord_error_initial':err_initial,'coord_error_deformed':err_deformed,'ip_shape_permutation':permutations,
            'sum_reaction':rf.sum(0).tolist(),'max_displacement':np.abs(u).max(0).tolist(),
            'energies':{key:float(value.data[-1][1]) for region in step.historyRegions.values() for key,value in region.historyOutputs.items() if key in ['ALLSE','ALLWK']}}
    odb.close()
    with open(os.path.join(OUT,name+'.json'),'w') as f:json.dump(exported,f,indent=2)
    print(name+' EXPORTED',flush=True)

records=json.load(open(os.path.join(BASE,'mesh_manifest.json')))
aux=os.path.join(BASE,'kirsch_manifest.json')
if os.path.exists(aux):records+=json.load(open(aux))
for record in records:
    if not os.path.exists(os.path.join(OUT,record['job']+'.json')):export(record)
