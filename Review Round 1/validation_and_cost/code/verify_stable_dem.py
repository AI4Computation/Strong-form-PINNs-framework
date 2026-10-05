"""Verify original initialization, analytic Jacobian and parameter gradients."""
from stable_dem import *
from types import SimpleNamespace


def main():
    model=build(42,'cpu');assert sum(p.numel() for p in model.parameters())==81402
    prior=ROOT.parent/'controlled_pinn/runs/C1_dem_s42/step_00000.pt'
    old=torch.load(prior,map_location='cpu',weights_only=True)['state_dict']
    assert all(torch.equal(v.float(),old[k]) for k,v in model.state_dict().items())
    xy=torch.tensor(np.random.default_rng(20261106).uniform(-.5,.5,(31,2)),dtype=torch.float64,requires_grad=True)
    uv=model(xy);gu=torch.autograd.grad(uv[:,0].sum(),xy,create_graph=True)[0];gv=torch.autograd.grad(uv[:,1].sum(),xy,create_graph=True)[0]
    trace=gu[:,0]+gv[:,1]
    stress=torch.stack([legacy.LAM*trace+2*legacy.MU*gu[:,0],legacy.LAM*trace+2*legacy.MU*gv[:,1],legacy.MU*(gu[:,1]+gv[:,0])],1)
    expected=torch.cat([uv,stress],1);actual,density=field(model,xy.detach())
    field_difference=float(torch.max(torch.abs(expected-actual)).detach());assert field_difference<1e-11
    exact_density=.5*legacy.LAM*trace.square()+legacy.MU*(gu[:,0].square()+gv[:,1].square()+.5*(gu[:,1]+gv[:,0]).square())
    model.zero_grad(set_to_none=True);exact_density.sum().backward()
    reference_grad=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()]).clone()
    model.zero_grad(set_to_none=True);density.sum().backward()
    actual_grad=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()])
    gradient_difference=float((reference_grad-actual_grad).abs().max());assert gradient_difference<1e-10
    other=build(42);probe=xy.detach().numpy()
    cpu_gpu=float(np.max(np.abs(predict(model,probe)-predict(other,probe))));assert cpu_gpu<1e-10
    volume=[]
    for base in [3,4]:
        q=rule(base);area_error=abs(q['area']-(1-np.pi*.1**2))
        assert area_error<1e-7
        volume.append(dict(base=base,points=len(q['xy']),area_error=area_error))
    sample=rule(2);n=len(sample['boundary_weights'])
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float64)
    shared=SimpleNamespace(quad_domain=tensor(sample['xy']),quad_domain_weights=tensor(sample['w']),
        quad_left=tensor(sample['boundary'][:n]),quad_right=tensor(sample['boundary'][n:2*n]),
        quad_top=tensor(sample['boundary'][2*n:]),quad_boundary_weights=tensor(sample['boundary_weights']))
    model.zero_grad(set_to_none=True);original_loss=legacy.dem_loss(model,shared,-1.,-5.);original_loss.backward()
    original_gradient=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()]).detach().numpy().copy()
    value=energy(model,sample,-1.,-5.,True)
    integrated_loss_difference=abs(float(original_loss.detach())-value['potential_energy'])
    integrated_gradient_difference=float(np.max(np.abs(original_gradient-value['gradient'])))
    assert integrated_loss_difference<1e-10 and integrated_gradient_difference<1e-10
    optimizer=ObservedLBFGS(list(model.parameters()),max_iter=2,max_eval=5,line_search_fn='strong_wolfe')
    def closure():return torch.tensor(energy(model,sample,-1.,-5.,True)['potential_energy'],dtype=torch.float64)
    optimizer.step(closure);after=energy(model,sample,-1.,-5.)['potential_energy']
    assert optimizer.accepted_steps==2 and after<value['potential_energy']
    result=dict(passed=True,original_float32_initial_weights_preserved=True,parameters=81402,
        field_autograd_max_difference=field_difference,parameter_gradient_max_difference=gradient_difference,
        cpu_gpu_max_difference=cpu_gpu,quadrature=volume,
        legacy_integrated_loss_difference=integrated_loss_difference,legacy_integrated_parameter_gradient_difference=integrated_gradient_difference,
        observed_optimizer_verified_steps=optimizer.accepted_steps,optimizer_energy_before=value['potential_energy'],optimizer_energy_after=after,
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('stable_dem.py'),LEGACY]},
        reference_initial_checkpoint_sha256=sha(prior))
    write(ROOT/'checks/stable_dem_verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
