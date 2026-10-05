"""E5 fixed-map diagnostics. No PDE training or selected-best seed."""
import runtime
from models import ROOT,FixedFeatures,generator
from runtime import torch
import numpy as np,json,csv

def scale_comparison(rows=None):
    if rows is None:
        with (ROOT/'summary/feature_diagnostics.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
    output=[{'method':r['method'],'seed':int(r['seed']),'features':1000,
        'mean_squared_feature_gradient':float(r['derivative_energy_mean']),
        'spatial_rule':'mean over fixed solid grid','sigma_over_default':None} for r in rows]
    for method,factor in [('fourier_half',.5),('fourier',1.),('fourier_double',2.)]:
        for seed in range(41,49):
            model=FixedFeatures(method,seed,100)
            energy=float((2*np.pi)**2*np.square(model.B.numpy().astype(np.float64)).sum()/1000)
            output.append({'method':method,'seed':seed,'features':1000,
                'mean_squared_feature_gradient':energy,
                'spatial_rule':'exact sin/cos pair identity, independent of x','sigma_over_default':factor})
    from summarize import write_csv
    write_csv('feature_scale_comparison.csv',output)

def main():
    axis=np.linspace(-.495,.495,64)
    xx,yy=np.meshgrid(axis,axis);points=np.column_stack([xx.ravel(),yy.ravel()])
    mask=np.linalg.norm(points,axis=1)>.1
    xy=points[mask];near=np.linalg.norm(xy,axis=1)<=.2
    rows=[];archive={'axis':axis,'xy':xy,'mask':mask}
    for seed in range(41,49):
        for method in ('anchored','independent_gaussian','independent_marginal'):
            m=FixedFeatures(method,seed,100)
            W=m.W.numpy().astype(np.float64);b=m.b.numpy().astype(np.float64)
            with torch.no_grad():
                z32=torch.tensor(xy,dtype=torch.float32)@m.W.T+m.b
                z=z32.numpy().astype(np.float64);h=torch.tanh(z32).numpy().astype(np.float64)
            active=np.abs(z)<=1.;coverage=active.mean(1)
            energy=((1-h*h)**2*(W*W).sum(1)[None,:]).mean(1)
            centered=h-h.mean(0);norm=np.linalg.norm(centered,axis=0)
            alive=norm>1e-12
            q=np.zeros_like(centered);q[:,alive]=centered[:,alive]/norm[alive]
            gram=q.T@q
            corr=np.abs(gram[np.ix_(alive,alive)])
            offdiag=corr[~np.eye(len(corr),dtype=bool)]
            eigen=np.maximum(np.linalg.eigvalsh(gram),0.)
            largest=eigen[-1];cut=1e-6*largest;positive=eigen[eigen>cut]
            prob=eigen[eigen>0]/eigen.sum()
            row={'method':method,'seed':seed,'features':1000,'samples':len(xy),'constant_columns':int((~alive).sum()),
                 'ridge_intersection_fraction':float((np.abs(b)<=.5*np.abs(W).sum(1)).mean()),
                 'coverage_mean':float(coverage.mean()),'coverage_near_q05':float(np.quantile(coverage[near],.05)),
                 'coverage_near_mean':float(coverage[near].mean()),'derivative_energy_mean':float(energy.mean()),
                 'coherence_max':float(offdiag.max()),'coherence_q95':float(np.quantile(offdiag,.95)),
                 'numerical_rank':len(positive),'effective_rank':float(np.exp(-np.sum(prob*np.log(prob)))),
                 'regularized_condition':float((largest+cut)/(eigen[0]+cut)),
                 'effective_spectral_condition':float(largest/positive[0])}
            centers=(torch.rand((1000,2),generator=generator(seed,400000))-.5).numpy()
            row['anchor_centers_in_solid_fraction']=float((np.linalg.norm(centers,axis=1)>.1).mean()) if method=='anchored' else None
            rows.append(row)
            archive[f'{method}_s{seed}_active_fraction']=active.mean(0)
            archive[f'{method}_s{seed}_eigenvalues']=eigen
            if seed==42:
                archive[method+'_coverage']=coverage;archive[method+'_energy']=energy
                archive[method+'_W']=W;archive[method+'_b']=b
            print(json.dumps({k:row[k] for k in ['method','seed','ridge_intersection_fraction','coverage_near_q05','effective_rank']}),flush=True)
    with (ROOT/'summary/feature_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(ROOT/'summary/feature_diagnostics.npz',**archive)
    note={'grid':'64x64 uniform grid on [-0.495,0.495]^2 excluding r<=0.1','transition_threshold':1.,
        'center_columns':True,'constant_norm_threshold':1e-12,'constant_columns':'kept as zero Gram columns; omitted from undefined pair correlations',
        'feature_evaluation':'CPU PyTorch float32 activations; centering, Gram matrix and spectrum evaluated in float64',
        'rank_relative_threshold':1e-6,'energy':'mean over features of squared spatial gradient norm',
        'spatial_replicates_are_not_independent_training_replicates':True}
    (ROOT/'summary/feature_diagnostic_definitions.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    scale_comparison(rows)

if __name__=='__main__':main()
