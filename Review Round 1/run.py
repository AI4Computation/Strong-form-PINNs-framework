"""Train or evaluate the supplied research experiments using package-local files."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRESETS = ('benchmark', 'dem', 'representations', 'budgets', 'sampling', 'transfer')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))



def jobs(preset):
    if preset in ('benchmark', 'dem'):
        configs = read(ROOT / 'configs/benchmark_runs.json')
        if preset == 'dem':
            configs = [dict(c, id=c['id'].replace('_dem_', '_verified_dem_'))
                       for c in configs if c['method'] == 'dem']
        return [dict(c, preset=preset) for c in configs]
    return [c for c in read(ROOT / 'configs/geometry_runs.json') if c['preset'] == preset]

def output(c):
    return ROOT / 'results' / c['preset'] / c['id']

def launch(action, c, a):
    command = [sys.executable, '-B', str(Path(__file__).resolve()), action,
               '--preset', c['preset'], '--id', c['id'], '--device', a.device, '--worker']
    if action == 'evaluate' and a.all_checkpoints:
        command.append('--all-checkpoints')
    if action == 'evaluate' and a.save_fields:
        command.append('--save-fields')
    subprocess.run(command, cwd=ROOT, check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['list', 'train', 'evaluate'])
    p.add_argument('--preset', required=True, choices=PRESETS)
    p.add_argument('--id')
    p.add_argument('--case')
    p.add_argument('--method')
    p.add_argument('--seed', type=int)
    p.add_argument('--arm', choices=['continuous', 'fixed_reset', 'uniform_refresh'])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--skip-completed', action='store_true', help='Skip successful training runs only.')
    p.add_argument('--all-checkpoints', action='store_true', help='Evaluate all saved benchmark checkpoints.')
    p.add_argument('--save-fields', action='store_true', help='Save full regenerated predictions and errors.')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    selected = [c for c in jobs(a.preset) if all(getattr(a, key) is None or c.get(key) == getattr(a, key)
                for key in ['id', 'case', 'method', 'seed', 'arm'])]
    if not selected:
        p.error('No experiment matches these filters. Use list --preset PRESET to see available experiments.')
    if a.action == 'list':
        for c in selected:
            print(json.dumps(c, ensure_ascii=True))
        return
    if not a.worker:
        # Separate processes prevent baseline and geometry module-name collisions.
        for c in selected:
            folder = output(c)
            if a.action == 'train':
                if (folder / 'result.json').is_file() and a.skip_completed:
                    print('SKIP', c['id'], flush=True)
                    continue
                if folder.exists():
                    raise FileExistsError('Existing output; move it aside or use --skip-completed: ' + str(folder))
                if c['preset'] == 'sampling':
                    prefix = next(j for j in jobs('budgets') if all(j[k] == c[k] for k in ['case','method','seed']))
                    if not (output(prefix) / 'result.json').is_file():
                        if output(prefix).exists():
                            raise RuntimeError('Incomplete prefix run; inspect results/budgets before continuing.')
                        launch('train', prefix, a)
                    if not (output(prefix) / 'step_0400.pt').is_file():
                        raise RuntimeError('Continuous prefix did not reach 400 accepted updates.')
            launch(a.action, c, a)
        return
    if len(selected) != 1:
        p.error('A worker requires exactly one experiment.')
    c = selected[0]
    threads = 4 if c['preset'] == 'benchmark' else 2
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = str(threads)
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TUST_DEVICE'] = a.device
    sys.path.insert(0, str(ROOT / 'code'))
    family = 'benchmark' if c['preset'] == 'benchmark' else 'dem' if c['preset'] == 'dem' else 'geometry'
    sys.path.insert(0, str(ROOT / 'code' / family))
    import torch
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable. Choose --device cpu or a compatible CUDA runtime.') ###
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from runner import train, evaluate
    if a.action == 'train':
        train(c, output(c), a.device)
    else:
        evaluate(c, output(c), a.device, a.all_checkpoints, a.save_fields)

if __name__ == '__main__':
    main()
