"""Generate an observational copy of the installed PyTorch L-BFGS step.

The mathematical operations and stopping order stay unchanged. Insertions only
notify accepted states and name existing stopping conditions. The output retains
the installed source hash, and is verified against native L-BFGS before use.
PyTorch source is BSD-3-Clause: https://github.com/pytorch/pytorch/blob/main/LICENSE
"""
from pathlib import Path
import runtime
import torch, inspect, hashlib, textwrap

def main():
    src=inspect.getsource(torch.optim.LBFGS.step)
    original=src
    def replace(old,new):
        nonlocal src
        assert src.count(old)==1,(old,src.count(old))
        src=src.replace(old,new)
    replace('        # optimal condition\n        if opt_cond:\n            return orig_loss',
            '        self.accepted_steps = 0\n        self.stop_reason = None\n'
            '        self.observe(0, state["func_evals"], loss, float(flat_grad.abs().max()))\n'
            '        # optimal condition\n        if opt_cond:\n'
            '            self.stop_reason = "initial_gradient_tolerance"\n            return orig_loss')
    replace('            if gtd > -tolerance_change:\n                break',
            '            if gtd > -tolerance_change:\n                self.stop_reason = "directional_derivative_tolerance"\n                break')
    replace('            state["func_evals"] += ls_func_evals\n',
            '            state["func_evals"] += ls_func_evals\n'
            '            self.accepted_steps += 1\n'
            '            self.observe(self.accepted_steps, state["func_evals"], loss, float(flat_grad.abs().max()))\n')
    for condition,reason in [('n_iter == max_iter','max_iter'),('current_evals >= max_eval','max_eval'),
                             ('opt_cond','gradient_tolerance'),('d.mul(t).abs().max() <= tolerance_change','step_tolerance'),
                             ('abs(loss - prev_loss) < tolerance_change','loss_change_tolerance')]:
        replace(f'            if {condition}:\n                break',
                f'            if {condition}:\n                self.stop_reason = "{reason}"\n                break')
    prefix=('"""Observed installed PyTorch L-BFGS; see prepare_optimizer.py. BSD-3-Clause upstream.\n'
            'Original step SHA-256: '+hashlib.sha256(original.encode()).hexdigest()+'\n"""\n'
            'import torch\nfrom torch.optim.lbfgs import _strong_wolfe, _to_scalar\n\n'
            'class ObservedLBFGS(torch.optim.LBFGS):\n'
            '    def __init__(self, *args, observer=None, **kwargs):\n'
            '        super().__init__(*args, **kwargs)\n'
            '        self.observe = observer or (lambda *args: None)\n\n')
    path=Path(__file__).with_name('observed_lbfgs.py')
    path.write_text(prefix+src,encoding='utf-8')
    print('Generated',path.name,'from installed PyTorch; no environment files changed.')

if __name__=='__main__':main()
