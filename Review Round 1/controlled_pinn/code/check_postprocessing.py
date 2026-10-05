"""Exercise postprocessing on one real checkpoint set; diagnostic output stays separate."""
import runtime
from runtime import torch
import models,postprocess
from models import ROOT
import numpy as np,json
from scipy.spatial import cKDTree

def main():
    torch.set_num_threads(1)
    config=json.loads((ROOT/'runs/C1_anchored_s41/result.json').read_text())['configuration']
    # Generate the exact original CUDA RNG collocation once, then carry out the
    # functional check on CPU so it does not contend with formal GPU training.
    original=models.samples_for(41,'circle')
    cpu_samples={k:v.cpu() for k,v in original.items()}
    del original
    models.DEVICE=torch.device('cpu');postprocess.DEVICE=torch.device('cpu')
    postprocess.samples_for=lambda *args,**kwargs:cpu_samples
    rows,pairs=postprocess.gradients(config,ROOT/'runs/C1_anchored_s41')
    assert len(rows)==20 and len(pairs)==30
    assert all(np.isfinite(r['gradient_norm']) for r in rows)
    assert all(r['cosine'] is None or -1.000001<=r['cosine']<=1.000001 for r in pairs)
    ref=np.load(ROOT/'fem/references/T1.npz')
    p=models.POLYGON*75
    measure=np.vstack([p[np.argmax(p[:,1])],p[np.argmin(p[:,0])],p[np.argmax(p[:,0])]])
    distance,indices=cKDTree(ref['xy_u']).query(measure)
    assert distance.max()<1e-5
    uv=ref['u'][indices];e=(measure[2]-measure[1])/np.linalg.norm(measure[2]-measure[1])
    result={'passed':True,'gradient_rows_checked':len(rows),'cosine_rows_checked':len(pairs),
        'diagnostic_execution_device':'CPU; functional check only, not included in final gradient analysis',
        'engineering_node_distances_m':distance.tolist(),'reference_crown_settlement_mm':float(-uv[0,1]*1000),
        'reference_horizontal_convergence_mm':float(-(uv[2]-uv[1])@e*1000)}
    (ROOT/'checks/postprocessing_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
