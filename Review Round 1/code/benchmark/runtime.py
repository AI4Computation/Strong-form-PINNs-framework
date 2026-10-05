"""Process-only runtime settings. Never modifies the installed environment."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if os.environ.get('KMP_DUPLICATE_LIB_OK','').lower() in ('true','1','yes'):
    raise RuntimeError('Remove KMP_DUPLICATE_LIB_OK from the launch process; conflicts must not be ignored.')
import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
DEVICE=torch.device(os.environ.get('TUST_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
DTYPE=torch.float32

def sync():
    if DEVICE.type == 'cuda': torch.cuda.synchronize()

