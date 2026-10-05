"""Bounded single-GPU development batch; no FEM imports or reads."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import hashlib,json,time,sys,traceback
from datetime import datetime,timezone
import numpy as np
import torch
from p2_components import prepare_loss,parts_loss,objective
from shared_geometry_features import make_shared_model

def make_model(method,seed,covers,device):
    return make_shared_model(method,seed,*covers[0],uniform=covers[1]).to(device)
from p2_observed_lbfgs import ObservedLBFGS

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'results/R2_P2F_repetition_budget'
PROTOCOL=ROOT/'protocols/R2_P2F_repetition_budget.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not (BATCH/'manifest.json').exists(),'Existing batch: inspect rather than restart.'
    pre=load(BATCH/'preflight.json');assert pre['passed']
    assert sha(PROTOCOL)==pre['protocol_sha256']
    for name,h in pre['source_sha256'].items():assert sha(ROOT/'code'/name)==h
    for name,h in pre['inputs_sha256'].items():assert sha(name)==h
    for name,h in pre['prepared_sha256'].items():assert sha(BATCH/name)==h
    p=load(PROTOCOL)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    manifest=dict(id=p['id'],status='running',created_utc=utc(),active=None,completed=[],
                  source_sha256={**pre['source_sha256'],Path(__file__).name:sha(__file__)},
                  prepared_sha256=pre['prepared_sha256'],protocol_sha256=sha(PROTOCOL),
                  preflight_sha256=sha(BATCH/'preflight.json'),terminal_sha256={},
                  python=sys.executable,torch=torch.__version__,device=torch.cuda.get_device_name(),
                  fem_read=False,timing_valid_for_comparison=False)
    write(BATCH/'manifest.json',manifest)
    try:
        cases={f'{case}_seed{seed}':dict(setting,seed=seed,original_case=case) for seed in p['seeds'] for case,setting in p['case_settings'].items()}
        for ci,(case,setting) in enumerate(cases.items()):
            geometry=setting['geometry']
            with np.load(BATCH/f'{geometry}_seed{setting["seed"]}_points.npz') as z:points={k:z[k] for k in z.files}
            with np.load(BATCH/f'{geometry}_covers.npz') as z:covers=[(z['centres'],z['halfwidths']),(z['uniform_centres'],z['uniform_halfwidths'])]
            shift=ci%len(p['methods']);order=p['methods'][shift:]+p['methods'][:shift]
            for method in order:
                name=case+'_'+method;out=BATCH/name;out.mkdir()
                manifest['active']=name;write(BATCH/'manifest.json',manifest)
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                prep_start=time.perf_counter()
                model=make_model(method,setting['seed'],covers,'cuda').float()
                prepared=prepare_loss(model,points);torch.cuda.synchronize()
                prep_seconds=time.perf_counter()-prep_start
                trace=[];saved=[];closures=0
                def save_weights(label):
                    dest=out/(label+'.pt');assert not dest.exists()
                    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},dest)
                start=time.perf_counter()
                def observer(step,evaluations,loss,gradient):
                    if not np.isfinite([loss,gradient]).all():raise FloatingPointError('Nonfinite accepted state')
                    item=dict(accepted_step=step,closure_evaluations=evaluations,loss=loss,max_gradient=gradient,
                              development_elapsed_seconds=time.perf_counter()-start)
                    trace.append(item)
                    if step in p['checkpoints']:
                        save_weights(f'step_{step:04d}');saved.append(step)
                    if step%400==0:
                        write(out/'live.json',item)
                        print(json.dumps(dict(run=name,**item)),flush=True)
                cfg={k:v for k,v in p['optimizer'].items() if k not in ['name','budget_note']}
                optimizer=ObservedLBFGS(model.parameters(),observer=observer,**cfg)
                def closure():
                    nonlocal closures
                    optimizer.zero_grad(set_to_none=True)
                    value=objective(parts_loss(model,prepared,setting['lateral'],setting['top']))
                    if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite trial objective')
                    value.backward();closures+=1
                    if not all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters()):
                        raise FloatingPointError('Nonfinite trial gradient')
                    return value
                optimizer.step(closure);torch.cuda.synchronize()
                training_seconds=time.perf_counter()-start
                save_weights('terminal')
                with torch.no_grad():
                    final={k:float(v) for k,v in parts_loss(model,prepared,setting['lateral'],setting['top']).items()}
                result=dict(case=case,method=method,seed=setting['seed'],parameters=sum(v.numel() for v in model.parameters()),
                            accepted_steps=optimizer.accepted_steps,closure_evaluations=closures,stop_reason=optimizer.stop_reason,
                            loss_parts=final,final_loss=final['equilibrium']+final['constitutive']+10*final['traction']+100*final['displacement'],
                            saved_checkpoints=saved,finished_utc=utc(),terminal_sha256=sha(out/'terminal.pt'),
                            prepared_points_sha256=sha(BATCH/f'{geometry}_seed{setting["seed"]}_points.npz'),
                            preparation_seconds=prep_seconds,optimization_including_checkpoint_io_seconds=training_seconds,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            fem_read=False,timing_valid_for_comparison=False)
                write(out/'trace.json',trace);write(out/'result.json',result)
                manifest['completed'].append(name)
                manifest['terminal_sha256'][name]={file:sha(out/file) for file in ['terminal.pt','result.json','trace.json']+[f.name for f in out.glob('step_*.pt')]}
                write(BATCH/'manifest.json',manifest)
                print('FROZEN',name,'loss',result['final_loss'],flush=True)
                del optimizer,prepared,model
                torch.cuda.empty_cache()
        manifest.update(status='fit_complete',active=None,finished_utc=utc())
        write(BATCH/'manifest.json',manifest)
        print('ALL_27_TRAJECTORIES_AND_REGISTERED_CHECKPOINTS_FROZEN_NO_FEM_READ',flush=True)
    except BaseException as exc:
        manifest.update(status='failed',failure=str(exc),failed_utc=utc())
        write(BATCH/'manifest.json',manifest)
        write(BATCH/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),active=manifest['active']))
        raise


if __name__=='__main__':main()
