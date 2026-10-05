"""Run the finite, frozen coverage pipeline through evaluation and reporting."""
from run_stable_dem_coverage import *
import subprocess
import os


def main():
    manifest=initialize();state_path=BATCH/'pipeline_status.json'
    stages=['run_stable_dem_coverage.py','evaluate_stable_dem_coverage.py','finalize_stable_dem_coverage.py']
    state=dict(status='running',pid=os.getpid(),started_utc=utc(),completed_stages=[],active=None,
        pipeline_sha256=sha(__file__),source_archive_sha256=manifest['source_archive_sha256'],timing_valid_for_comparison=False)
    write(state_path,state)
    try:
        for filename in stages:
            state.update(active=filename,updated_utc=utc());write(state_path,state)
            with (BATCH/(Path(filename).stem+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name(filename))],cwd=str(ROOT.parents[1]),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            state['completed_stages'].append(filename);write(state_path,state)
        completion=read(BATCH/'completion.json');assert completion['passed']
        state.update(status='complete',active=None,completed_utc=utc(),completion_sha256=sha(BATCH/'completion.json'))
        write(state_path,state)
    except Exception as error:
        state.update(status='needs_attention',error=repr(error),updated_utc=utc());write(state_path,state)
        current=read(ROOT/'进度.json');current.update(current_jobs=[],milestone='stable_dem_coverage_pipeline_needs_attention',
            stable_dem_pipeline_error=repr(error),updated_utc=utc());write(ROOT/'进度.json',current)
        raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8');main()
