"""Serial formal cost replays; no reference evaluation during timing."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,hashlib,time,subprocess,ctypes,platform,traceback
from pathlib import Path
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2L_formal_cost';P=ROOT/'protocols/R2_P2L_formal_cost.json'
SMI=Path('C:/Windows/system32/nvidia-smi.exe')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
class Power(ctypes.Structure):
    _fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
def ac():
    p=Power();assert ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p));return int(p.ACLineStatus)
def query(fields):
    return subprocess.check_output([str(SMI),'--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True,creationflags=subprocess.CREATE_NO_WINDOW).strip()
def main():
    assert not B.exists(),'Existing batch must not be overwritten';p=read(P)
    # Verify frozen ancestry before any timed solve, not in parallel with one.
    ancestry={}
    for name,key in [('R2_P2J_reference_repair','P2J_completion_sha256'),('R2_P2K_transfer_replication','P2K_completion_sha256')]:
        folder=ROOT/'results'/name;assert sha(folder/'completion.json')==p[key];closed=read(folder/'completion.json')
        for f,h in closed['files_sha256'].items():assert sha(folder/f)==h,(name,f)
        for f,h in closed.get('supporting_files_sha256',{}).items():assert sha(f)==h,f
        ancestry[name]=dict(completion_sha256=sha(folder/'completion.json'),verified_files=len(closed['files_sha256']))
    sources=['bench_p2l.py','bench_p2l_worker.py','transfer_mechanics.py','p2_components.py','shared_geometry_features.py','sparse_mixed_pinn.py','geometry_primitives.py','cavity_cover.py','p2_observed_lbfgs.py','evaluate_p2.py','p2_fem_interpolation.py','support_probes.py']
    inputs=[ROOT/'inputs/cavity_geometries.json',ROOT/'protocols/R2_P1_geometry.json']+[ROOT/f'results/R2_P2D_shared_features/{c}_covers.npz' for c in p['cases']]
    B.mkdir();m=dict(id=p['id'],status='running',created_utc=utc(),protocol_sha256=sha(P),source_sha256={f:sha(ROOT/'code'/f) for f in sources},inputs_sha256={str(f.resolve()):sha(f) for f in inputs},ancestry=ancestry,completed=[],active=None,gate_records=[],run_records=[],formal_timing=True,author_idle_confirmation=p['author_idle_confirmation'])
    write(B/'manifest.json',m)
    write(B/'environment.json',dict(recorded_utc=utc(),python=sys.version,executable=sys.executable,platform=platform.platform(),logical_cpus=psutil.cpu_count(),physical_cpus=psutil.cpu_count(logical=False),ram_bytes=psutil.virtual_memory().total,gpu=query('name,driver_version,memory.total,power.limit'),cpu_threads=2,environment={k:os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_THREADING_LAYER','CUBLAS_WORKSPACE_CONFIG']}))
    telemetry=None;stream=None
    try:
        stream=(B/'gpu_telemetry.csv').open('w',encoding='utf-8')
        telemetry=subprocess.Popen([str(SMI),'--query-gpu=timestamp,index,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.current.graphics,clocks.current.memory','--format=csv','--loop-ms=2000'],stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        for job in p['jobs']:
            if job['index']:time.sleep(8)
            gate=dict(index=job['index'],utc=utc(),system_cpu_percent=psutil.cpu_percent(interval=2),ac=ac(),gpu=query('utilization.gpu,temperature.gpu,power.draw,memory.used'))
            gate['passed']=gate['ac']==1 and gate['system_cpu_percent']<=15 and float(gate['gpu'].split(',')[0])<=10
            m['gate_records'].append(gate);write(B/'manifest.json',m);assert gate['passed'],('idle gate failed',gate)
            assert telemetry.poll() is None,'GPU telemetry process stopped'
            m['active']=job;write(B/'manifest.json',m);before=utc();tick=time.perf_counter()
            with (B/f"job{job['index']:02d}.stdout.log").open('w',encoding='utf-8') as log:
                proc=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'code/bench_p2l_worker.py'),str(job['index'])],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            elapsed=time.perf_counter()-tick;name=f"{job['index']:02d}_{job['case']}_seed{job['seed']}_{job['method']}_uniform_refresh"
            m['run_records'].append(dict(index=job['index'],folder=name,started_utc=before,finished_utc=utc(),process_envelope_seconds=elapsed,exit_code=proc.returncode));write(B/'manifest.json',m)
            assert proc.returncode==0,('worker failed',job['index'])
            result=read(B/name/'result.json');assert all(result['exact_replay'].values()) and result['accepted_steps']==1600
            m['completed'].append(name);m['active']=None;write(B/'manifest.json',m);print('FORMAL_REPLAY_PASS',len(m['completed']),'/18',name,round(elapsed,3),flush=True)
        m.update(status='timing_complete',active=None,finished_utc=utc());write(B/'manifest.json',m);print('ALL_18_TIMED_REPLAYS_COMPLETE',flush=True)
    except BaseException as exc:
        m.update(status='failed',failure=repr(exc),traceback=traceback.format_exc());write(B/'manifest.json',m);raise
    finally:
        if telemetry is not None and telemetry.poll() is None:telemetry.terminate();telemetry.wait(timeout=10)
        if stream is not None:stream.close()
if __name__=='__main__':main()
