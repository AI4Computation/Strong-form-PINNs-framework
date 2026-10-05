"""Numerical checks for observational optimizer and controlled experiment setup."""
import runtime
from runtime import torch,DEVICE
from models import *
from observed_lbfgs import ObservedLBFGS
from metrics import field_metrics
import json

def optimizer_check(dtype,device):
    initial=torch.tensor([-1.2,1.0,.7],dtype=dtype,device=device)
    traces=[];parameters=[];accepted=[]
    for cls in [torch.optim.LBFGS,ObservedLBFGS]:
        x=nn.Parameter(initial.clone());trace=[];states=[]
        kwargs=dict(max_iter=80,history_size=50,line_search_fn='strong_wolfe',tolerance_change=1e-12,tolerance_grad=1e-7)
        if cls is ObservedLBFGS:kwargs['observer']=lambda *args:states.append(args)
        opt=cls([x],**kwargs)
        def closure():
            opt.zero_grad(set_to_none=True)
            loss=(1-x[0])**2+100*(x[1]-x[0]**2)**2+.3*(x[2]-.1)**2
            loss.backward();trace.append(float(loss.detach()));return loss
        opt.step(closure)
        traces.append(trace);parameters.append(x.detach().clone())
        if states:accepted=states
    assert traces[0]==traces[1],('changed_closures',dtype,device)
    assert torch.equal(*parameters),('changed_parameters',dtype,device)
    assert len(accepted)<len(traces[0])+1
    return dict(dtype=str(dtype),device=str(device),closure_evaluations=len(traces[0]),
                accepted_states=len(accepted),bitwise_equal=True)

def main():
    results={'optimizer_equivalence':[optimizer_check(torch.float64,'cpu'),optimizer_check(torch.float32,DEVICE)]}
    counts={}
    for geometry in ['circle','tunnel']:
        methods=['anchored','fourier','vanilla_matched']
        if geometry=='circle':methods+=['vanilla','independent_gaussian','independent_marginal','xpinn','dem']
        for method in methods:
            m=build_model(method,42,geometry)
            counts[geometry+'_'+method]=sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert counts['circle_anchored']==110705 and counts['circle_vanilla_matched']==110913
    assert counts['tunnel_anchored']==145285 and counts['tunnel_vanilla_matched']==145253
    a=build_model('anchored',42);g=build_model('independent_gaussian',42);b=build_model('independent_marginal',42)
    assert torch.equal(a.W,g.W) and torch.equal(a.W,b.W)
    for other in [g,b,build_model('fourier',42),build_model('fourier_half',42),build_model('fourier_double',42)]:
        assert all(torch.equal(v,other.net.state_dict()[k]) for k,v in a.net.state_dict().items())
    s=samples_for(42,'circle',128,16)
    old=legacy.mixed_loss(a,legacy.SampleSet(**s),-1.,-5.)
    new=objective(a,s,dict(method='anchored',geometry='circle',p_lateral=-1.,p_top=-5.))
    assert torch.isclose(old,new,rtol=2e-7,atol=1e-5),(old.item(),new.item())
    ts=tunnel_samples(42,8000)
    assert len(ts['domain'])==8000 and in_rock(ts['domain'].cpu().numpy(),'tunnel').all()
    assert array_hash(ts['domain'])==array_hash(tunnel_samples(42,8000)['domain'])
    zero=field_metrics(np.zeros((4,2)),np.ones((4,2)),['x','y'])
    assert zero['vector_pct'] is None and zero['x_pearson'] is None and zero['rmse']==1.
    results.update(parameter_counts=counts,shared_weights_and_downstream=True,
                   circle_objective_equivalence=True,paired_collocation=True,zero_reference_handling=zero)
    (ROOT/'checks/core_verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
