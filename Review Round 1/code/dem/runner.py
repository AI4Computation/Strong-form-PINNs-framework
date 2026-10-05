"""Numerically checked DEM comparator; FEM is used only after training."""
from common import *
from solver import solve
import stable_dem

def train(c,out,device):
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    p=read(ROOT/'configs/dem.json')
    synchronize(device);start=time.perf_counter()
    solve(c,dict(completed=[]),p,out,device)
    synchronize(device)
    record=read(out/'result.json')
    record.update(elapsed_seconds=time.perf_counter()-start,environment=resources(device))
    write(out/'result.json',record)

def evaluate(c,out,device,all_checkpoints=False,save_fields=False):
    model=stable_dem.build(c['seed'],device=device)
    load_weights(model,out/'model.pt',device);model.eval()
    ref=load_npz(ROOT/f'data/fem/benchmark/references/{c["case"]}.npz')
    values={};fields={}
    for name,cols in [('u',slice(0,2)),('s',slice(2,5))]:
        pred=stable_dem.predict(model,ref['xy_'+name])[:,cols];truth=ref[name]
        values[name+'_relative_l2_percent']=float(100*np.linalg.norm(pred-truth)/np.linalg.norm(truth))
        fields[name]=pred
    if 'volume' in ref and len(ref['volume'])==len(fields['s']):
        w=ref['volume'].reshape(-1)
        values['s_area_relative_l2_percent']=float(100*np.sqrt((w[:,None]*(fields['s']-ref['s'])**2).sum()/(w[:,None]*ref['s']**2).sum()))
    write(out/'evaluation.json',dict(configuration=c,metrics=values))
    if save_fields:np.savez_compressed(out/'fields.npz',**fields)
    print(json.dumps(values),flush=True)
