"""One warmed-process cost replay; all output belongs to the new P2L batch."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,time,json,hashlib,threading,ctypes
from pathlib import Path
import numpy as np
import psutil
import torch
from cavity_cover import normalized_domain,construct
from shared_geometry_features import make_shared_model,uniform_centres
from p2_components import uniform_cover,objective
from transfer_mechanics import sample,prepare,parts
from p2_observed_lbfgs import ObservedLBFGS
from evaluate_p2 import predict
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2L_formal_cost';P=ROOT/'protocols/R2_P2L_formal_cost.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Power(ctypes.Structure):
    _fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
def ac():
    p=Power();assert ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p));return int(p.ACLineStatus)
def main(index):
    entered=time.perf_counter();p=read(P);job=p['jobs'][index];case,seed,method=[job[k] for k in ['case','seed','method']];setting=p['cases'][case];name=f'{case}_seed{seed}_{method}_uniform_refresh';out=B/f'{index:02d}_{name}';assert not out.exists();out.mkdir()
    source=ROOT/'results/R2_P2I_shape_transfer' if seed==260930 else ROOT/f'results/R2_P2K_transfer_replication/seed{seed}'
    frozen=read(source/'manifest.json');expected=load(ROOT/f'results/R2_P2D_shared_features/{case}_covers.npz');original=[load(source/f'{case}_block{i}_points.npz') for i in range(4)]
    assert sha(P)==read(B/'manifest.json')['protocol_sha256']
    for f,h in read(B/'manifest.json')['source_sha256'].items():assert sha(ROOT/'code'/f)==h
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    warm_start=time.perf_counter();warm=make_shared_model('fourier_half',424242,expected['centres'],expected['halfwidths']).float().cuda();wp=prepare(warm,original[0])
    for _ in range(50):warm.zero_grad(set_to_none=True);v=objective(parts(warm,wp,setting));v.backward()
    torch.cuda.synchronize();del v,wp,warm;torch.cuda.empty_cache();warm_seconds=time.perf_counter()-warm_start;assert ac()==1
    process=psutil.Process();samples=[];stop=threading.Event();baseline_rss=process.memory_info().rss
    def monitor():
        while not stop.is_set():
            samples.append(dict(time=time.perf_counter(),rss=process.memory_info().rss,system_cpu_percent=psutil.cpu_percent(None),ac=ac()));stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();components={};trace=[];blocks=[]
    tick=time.perf_counter();geo=read(ROOT/'inputs/cavity_geometries.json')[case];domain,_,_=normalized_domain(geo);cover=construct(domain,read(ROOT/'protocols/R2_P1_geometry.json'));c,h=cover['centres'],cover['halfwidths'];pc,ph,_=uniform_cover(domain,len(c));components['common_geometry_seconds']=time.perf_counter()-tick
    assert np.array_equal(c,expected['centres']) and np.array_equal(h,expected['halfwidths'])
    tick=time.perf_counter();base=sample(domain,[(c,h),(pc,ph)],seed,setting);components['initial_points_seconds']=time.perf_counter()-tick;n=int(base['uniform_count'])
    tick=time.perf_counter();uniform=uniform_centres(len(c),domain) if method=='uniform_rbf_fourier' else None;model=make_shared_model(method,seed,c,h,uniform).float().cuda();torch.cuda.synchronize();components['representation_initialization_seconds']=time.perf_counter()-tick
    if uniform is not None:assert np.array_equal(uniform[0],expected['uniform_centres']) and np.array_equal(uniform[1],expected['uniform_halfwidths'])
    total=0;closures_total=0
    for block in range(4):
        tick=time.perf_counter();points={k:v.copy() for k,v in base.items()}
        if block:points['domain'][:n]=domain.random_interior(n,np.random.default_rng(100000+seed+1000000*block))
        generation=time.perf_counter()-tick;assert all(np.array_equal(v,original[block][k]) for k,v in points.items())
        tick=time.perf_counter();pr=prepare(model,points);torch.cuda.synchronize();preparation=time.perf_counter()-tick;closures=0
        def observer(step,evaluations,loss,gradient):
            assert np.isfinite([loss,gradient]).all();trace.append(dict(block=block,block_step=step,cumulative_step=total+step,closure_evaluations=closures_total+evaluations,loss=loss,max_gradient=gradient))
        tick=time.perf_counter();opt=ObservedLBFGS(model.parameters(),observer=observer,**p['optimizer']);setup=time.perf_counter()-tick
        def closure():
            nonlocal closures
            opt.zero_grad(set_to_none=True);v=objective(parts(model,pr,setting));assert torch.isfinite(v);v.backward();closures+=1;assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in model.parameters());return v
        torch.cuda.synchronize();tick=time.perf_counter();opt.step(closure);torch.cuda.synchronize();optimization=time.perf_counter()-tick;assert opt.accepted_steps==400;(total,closures_total)=(total+400,closures_total+closures)
        compute_elapsed=time.perf_counter()-start;tick=time.perf_counter();state={k:v.detach().cpu() for k,v in model.state_dict().items()};torch.save(state,out/f'step_{total:04d}.pt');write(out/'trace.json',trace);io=time.perf_counter()-tick
        blocks.append(dict(block=block,cumulative_step=total,accepted_steps=400,closures=closures,refresh_generation_seconds=generation,cache_preparation_seconds=preparation,optimizer_setup_seconds=setup,optimization_seconds=optimization,checkpoint_trace_io_seconds=io,elapsed_before_checkpoint_io_seconds=compute_elapsed,end_to_end_seconds=time.perf_counter()-start))
        print(index,name,'TIMED',total,round(blocks[-1]['end_to_end_seconds'],3),flush=True);del opt,pr,state
    torch.cuda.synchronize();solve_seconds=time.perf_counter()-start;allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();stop.set();thread.join()
    assert samples and all(x['ac']==1 for x in samples);components['solve_end_to_end_seconds']=solve_seconds
    # Reference-free output evaluation, separate from training and target times.
    tick=time.perf_counter();query=domain.random_interior(10000,np.random.default_rng(67001));query_generation=time.perf_counter()-tick;torch.cuda.synchronize();tick=time.perf_counter();q=predict(model,query);torch.cuda.synchronize();inference=time.perf_counter()-tick;np.savez_compressed(out/'output_queries.npz',xy=query,q=q)
    # Only after timing: verify every state against the already frozen science run.
    verify_start=time.perf_counter();matches={}
    for step in [400,800,1200,1600]:
        oldpath=source/name/f'step_{step:04d}.pt';assert sha(oldpath)==frozen['terminal_sha256'][name][oldpath.name];old=torch.load(oldpath,map_location='cpu',weights_only=True);new=torch.load(out/oldpath.name,map_location='cpu',weights_only=True);equal=set(old)==set(new) and all(torch.equal(v,new[k]) for k,v in old.items());matches[str(step)]=equal;assert equal,(name,step,'not exact replay')
    record=read(source/name/'result.json');assert closures_total==record['closure_evaluations'] and [x['closures'] for x in blocks]==[x['closures'] for x in record['blocks']]
    for x in samples:x['seconds_from_solve_start']=x.pop('time')-start
    write(out/'resource_samples.json',samples)
    result=dict(**job,name=name,formal_timing=True,parameters=sum(v.numel() for v in model.parameters()),accepted_steps=total,closure_evaluations=closures_total,components=components,blocks=blocks,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,baseline_rss_bytes=baseline_rss,peak_sampled_rss_bytes=max(x['rss'] for x in samples),warmup_seconds=warm_seconds,pre_solve_python_setup_and_warmup_seconds=start-entered,post_solve_verification_seconds=time.perf_counter()-verify_start,query_count=len(query),query_generation_seconds=query_generation,query_inference_seconds=inference,exact_replay=matches,source_scientific_run=str((source/name).resolve()),all_ac_samples_on=True,protocol_sha256=sha(P));write(out/'result.json',result);print('TIMING_EXACT_REPLAY_PASS',index,flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
