"""Package-relative paths and output serialization."""
from pathlib import Path
import json
import platform
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def save(model, path, actual_steps):
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    actual_steps=actual_steps), path)

def load_weights(model, path, device):
    record = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(record.get('state_dict', record), strict=True)
    return record.get('actual_steps')

def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()

def resources(device):
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
                cpu_threads=torch.get_num_threads(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else None,
                formal_timing=False)

def finite_loss_and_gradients(loss, model):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite objective.')
    loss.backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite parameter gradient.')
