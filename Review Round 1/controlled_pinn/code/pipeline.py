"""Execute the authorized 334-run protocol, then audit and postprocess it."""
import runtime
from models import ROOT
from train import run_one,write_json,code_hashes
from summarize import summarize
import json,datetime,traceback,os

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    manifest=json.loads((ROOT/'config/run_manifest.json').read_text())
    assert len(manifest)==334
    selection=json.loads((ROOT/'config/reference_selection.json').read_text())
    assert selection['circle']['validated'] and selection['tunnel']['validated']
    # FEM geometry approval order is fixed before this batch: circle, then tunnel.
    ordered=[c for g in ('circle','tunnel') for c in manifest if c['geometry']==g]
    status=ROOT/'进度.json';started=now()
    done=sum((ROOT/'runs'/c['id']/'result.json').exists() for c in manifest)
    frozen=code_hashes()
    try:
        for config in ordered:
            assert code_hashes()==frozen,'Training source changed during the formal batch.'
            write_json(status,{'status':'training','started_utc':started,'updated_utc':now(),'pid':os.getpid(),
                'completed':done,'planned':334,'current':config['id']})
            existed=(ROOT/'runs'/config['id']/'result.json').exists()
            print('START '+config['id'],flush=True);run_one(config)
            if not existed:done+=1
            if done%10==0:summarize()
        write_json(status,{'status':'postprocessing','started_utc':started,'updated_utc':now(),'pid':os.getpid(),
                          'completed':done,'planned':334})
        summarize(require_complete=True)
        from audit_results import audit
        audit()
        from postprocess import main as postprocess
        postprocess()
        write_json(status,{'status':'computed_pending_scientific_and_visual_review','started_utc':started,
            'updated_utc':now(),'completed':done,'planned':334,'note':'Training and automated checks complete; manuscript conclusions and figure QA still require review.'})
    except Exception:
        error=traceback.format_exc()
        write_json(status,{'status':'failed','started_utc':started,'updated_utc':now(),'completed':done,'planned':334,'error':error})
        raise

if __name__=='__main__':main()
