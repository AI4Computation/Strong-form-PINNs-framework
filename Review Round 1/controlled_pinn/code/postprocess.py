"""Post-training diagnostics, never included in optimizer timing."""
import runtime
from runtime import torch,DEVICE
from models import ROOT,POLYGON,build_model,samples_for,parts_loss,WEIGHTS
from metrics import predict
from summarize import write_csv
import numpy as np,json,gc
from scipy.spatial import cKDTree

def gradients(config,folder):
    rows=[];pairs=[]
    if config['case'] not in ('C1','C8') or config['method'] not in ('anchored','independent_gaussian','independent_marginal','fourier','vanilla','vanilla_matched'):
        return rows,pairs
    model=build_model(config['method'],config['seed'],config['geometry']);s=samples_for(config['seed'],'circle')
    parameters=[p for p in model.parameters() if p.requires_grad];count=sum(p.numel() for p in parameters)
    for path in sorted(folder.glob('step_*.pt')):
        checkpoint=torch.load(path,map_location=DEVICE,weights_only=True);model.load_state_dict(checkpoint['state_dict'])
        step=checkpoint['accepted_step']
        terms=parts_loss(model,s,config['p_lateral'],config['p_top'])
        vectors={}
        for name,value in terms.items():
            grad=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
            vector=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(parameters,grad)])
            norm=float(torch.linalg.vector_norm(vector));vectors[name]=vector.detach()
            rows.append({'case':config['case'],'method':config['method'],'seed':config['seed'],'accepted_step':step,
                'loss_term':name,'loss':float(value.detach()),'gradient_norm':norm,'weight':WEIGHTS[name],
                'weighted_gradient_norm':norm*WEIGHTS[name],'gradient_norm_per_sqrt_parameter':norm/np.sqrt(count)})
        names=list(vectors)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                va,vb=vectors[a],vectors[b];den=torch.linalg.vector_norm(va)*torch.linalg.vector_norm(vb)
                cosine=float(torch.dot(va,vb)/den) if float(den)>0 else None
                pairs.append({'case':config['case'],'method':config['method'],'seed':config['seed'],'accepted_step':step,
                              'term_a':a,'term_b':b,'cosine':cosine,'negative':None if cosine is None else cosine<0})
        del terms,vectors
    return rows,pairs

def main():
    manifest=json.loads((ROOT/'config/run_manifest.json').read_text())
    selection=json.loads((ROOT/'config/reference_selection.json').read_text())
    refs={};volumes={}
    for geometry,step in [('circle','UnitL'),('tunnel','Load')]:
        file=ROOT/'fem'/f"{selection[geometry]['mesh']}_{step}.npz"
        with np.load(file) as data:volumes[geometry]=data['volume']
    tunnel=dict(np.load(ROOT/'fem/references/T1.npz'))
    left=POLYGON[np.argmin(POLYGON[:,0])]*75;right=POLYGON[np.argmax(POLYGON[:,0])]*75
    crown=POLYGON[np.argmax(POLYGON[:,1])]*75;measure=np.vstack([crown,left,right])
    distance,indices=cKDTree(tunnel['xy_u']).query(measure)
    assert distance.max()<1e-5,(distance,measure)
    e=(right-left)/np.linalg.norm(right-left)
    def quantities(uv):return -uv[0,1],-float((uv[2]-uv[1])@e)
    true_settlement,true_convergence=quantities(tunnel['u'][indices])
    engineering=[];weighted=[];gradrows=[];cosrows=[]
    for config in manifest:
        folder=ROOT/'runs'/config['id'];result=json.loads((folder/'result.json').read_text())
        if config['case'] not in refs:refs[config['case']]=dict(np.load(ROOT/'fem/references'/f"{config['case']}.npz"))
        ref=refs[config['case']]
        checkpoint=torch.load(folder/f"step_{result['accepted_steps']:05d}.pt",map_location=DEVICE,weights_only=True)
        model=build_model(config['method'],config['seed'],config['geometry']);model.load_state_dict(checkpoint['state_dict'])
        pred=predict(model,ref['xy_s'],'s',config['geometry']);v=volumes[config['geometry']]
        area_error=float(100*np.sqrt(np.sum(v[:,None]*(pred-ref['s'])**2)/np.sum(v[:,None]*ref['s']**2)))
        weighted.append({'id':config['id'],'case':config['case'],'method':config['method'],'seed':config['seed'],
                         'area_weighted_stress_pct':area_error,'primary_unweighted_stress_pct':result['metrics']['s_vector_pct']})
        if config['geometry']=='tunnel':
            steps=sorted(set([2000,result['accepted_steps']]))
            for step in steps:
                path=folder/f'step_{step:05d}.pt'
                if not path.exists():continue
                checkpoint=torch.load(path,map_location=DEVICE,weights_only=True);model.load_state_dict(checkpoint['state_dict'])
                uv=predict(model,measure,'u','tunnel');settlement,convergence=quantities(uv)
                engineering.append({'id':config['id'],'method':config['method'],'seed':config['seed'],'accepted_step':step,
                    'is_final':step==result['accepted_steps'],'crown_settlement_mm':settlement*1000,
                    'reference_settlement_mm':true_settlement*1000,'settlement_abs_error_mm':abs(settlement-true_settlement)*1000,
                    'horizontal_convergence_mm':convergence*1000,'reference_convergence_mm':true_convergence*1000,
                    'convergence_abs_error_mm':abs(convergence-true_convergence)*1000})
        del pred,model,checkpoint
        gr,co=gradients(config,folder);gradrows+=gr;cosrows+=co
        print('POST '+config['id'],flush=True)
    write_csv('area_weighting_sensitivity.csv',weighted);write_csv('engineering_quantities.csv',engineering)
    write_csv('loss_gradient_norms.csv',gradrows);write_csv('within_model_gradient_cosines.csv',cosrows)
    (ROOT/'summary/engineering_measurement_definition.json').write_text(json.dumps({
        'points_m':{'crown':crown.tolist(),'left_wall':left.tolist(),'right_wall':right.tolist()},
        'settlement':'-uy(crown)','convergence':'-(u_right-u_left) dot e_left_to_right',
        'reference_node_distances_m':distance.tolist(),'area_weighting':'Supplementary spatial-weighting sensitivity; registered primary remains unweighted IP vector norm.'},indent=2))
    from postprocess_supplement import main as supplement
    supplement(manifest)

if __name__=='__main__':main()
