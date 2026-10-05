"""Read existing 36-case checkpoints on CPU; correct undefined zero-load metrics.

No training and no rewriting original results. These describe the old checkpoints
on their original FE reference and must not be mixed with the new 334-run sample.
"""
import runtime
from runtime import torch
torch.set_num_threads(1)
from torch import nn
from models import ROOT,PROJECT
from metrics import field_metrics
import numpy as np,json,csv,hashlib

class PINN_Network(nn.Module):
    def forward(self,x):
        if hasattr(self,'fixed_layer'):
            return self.trainable_net(self.activation(self.fixed_layer(x)))
        return self.net(x)

def main():
    basedir=PROJECT/'代码源文件/【abaqus-36】'
    configs=[('2out_hard',basedir/'sensitivity_results+仅输出位移+硬约束',True),
             ('5out_hard',basedir/'sensitivity_results+输出五个变量+硬约束位移',True),
             ('5out_soft',basedir/'sensitivity_results+位移软约束',False),
             ('anchored_legacy',PROJECT/'代码源文件/【abaqus-36 - validation】/sensitivity_results',False)]
    rows=[]
    for name,folder,hard in configs:
        for p in range(0,-6,-1):
            for pt in range(0,-6,-1):
                label=f'P_{p}_Ptop_{pt}';path=folder/label/(label+'.pth')
                if not path.exists():raise FileNotFoundError(path)
                token=lambda v:str(abs(v)) if v else '1e-300'
                refpath=PROJECT/'abaqus-36/processed/displacement'/f'abaqus-p_lateral{token(p)}-p_top{token(pt)}.csv'
                data=np.loadtxt(refpath,delimiter=',',skiprows=1)
                xy=torch.tensor(data[:,:2],dtype=torch.float32)
                model=torch.load(path,map_location='cpu',weights_only=False).eval()
                with torch.no_grad():
                    uv=model(xy)[:,:2]
                    if hard:
                        uv=uv*torch.cat([xy[:,0:1]**2+(xy[:,1:2]+.5)**2,xy[:,1:2]+.5],1)
                pred=uv.numpy().astype(np.float64)
                row={'architecture':name,'p_lateral':p,'p_top':pt,'relative_defined':bool(np.linalg.norm(data[:,2:4])>0),
                     **field_metrics(data[:,2:4],pred,['u','v']),
                     'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                rows.append(row)
        print(name+' 36 existing checkpoints evaluated on CPU',flush=True)
    with (ROOT/'summary/legacy_36case_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    zero=[r for r in rows if not r['relative_defined']]
    assert len(zero)==4 and all(r['vector_pct'] is None for r in zero)
    summary={name:{'total_cases':36,'nonzero_cases':35,'zero_load':next(r for r in zero if r['architecture']==name),
              'mean_nonzero_u_vector_pct':float(np.mean([r['vector_pct'] for r in rows if r['architecture']==name and r['relative_defined']]))}
             for name,_,_ in configs}
    (ROOT/'summary/legacy_zero_load_audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
