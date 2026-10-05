"""Fixed cavity-scale Gaussian inputs to ONE fully trainable mixed PINN.

No output windows, per-patch networks, FEM, optimizer or file writes.
Coordinates and widths are in the common normalized domain.
"""
import math
import numpy as np
import torch
from p2_components import FeatureMixedPINN


def uniform_centres(count, domain):
    """Exact-count deterministic farthest-point design in the enclosing square.

    Domain membership rejects void centres, but local wall scales do not enter.
    Ties resolve by lexicographic candidate index, not case-specific choices.
    """
    n = int(np.ceil(2 * np.sqrt(count)))
    axis = (np.arange(n) + .5) / n - .5
    x, y = np.meshgrid(axis, axis, indexing='ij')
    grid = np.column_stack([x.ravel(), y.ravel()])
    grid = grid[domain.contains(grid)]
    assert len(grid) >= count
    distance = np.full(len(grid), np.inf)
    selected = []
    index = 0
    for _ in range(count):
        selected.append(index)
        distance = np.minimum(distance, ((grid-grid[index])**2).sum(1))
        distance[selected] = -1.
        index = int(np.argmax(distance))
    return grid[selected], np.full((count, 2), .75 / np.sqrt(count))


class SharedGeometryPINN(FeatureMixedPINN):
    def __init__(self, base, seed, centres, widths):
        super().__init__(base, seed)
        count = len(centres)
        assert 0 < count < 1000 and count % 2 == 0
        assert np.asarray(centres).shape == np.asarray(widths).shape == (count, 2)
        assert np.isfinite(centres).all() and np.isfinite(widths).all() and (widths > 0).all()
        self.global_count = 1000-count
        if hasattr(self, 'B'):
            self.B = self.B[:self.global_count//2].clone()
        else:
            self.W = self.W[:self.global_count].clone()
            self.b = self.b[:self.global_count].clone()
        self.register_buffer('centres', torch.tensor(centres, dtype=torch.float64))
        self.register_buffer('widths', torch.tensor(widths, dtype=torch.float64))

    def features(self, x):
        base = super().features(x)
        z = (x[:, None, :] - self.centres[None]) / self.widths[None]
        rbf = torch.exp(-.5 * z.square().sum(2))
        return torch.cat([base, rbf], 1)

    def prepare(self, xy):
        ref = next(self.parameters())
        x = torch.as_tensor(xy, dtype=ref.dtype, device=ref.device)
        if hasattr(self, 'B'):
            phase = 2*math.pi*x@self.B.T
            base = torch.cat([torch.sin(phase), torch.cos(phase)], 1)
            derivative = 2*math.pi*torch.cat([
                torch.cos(phase)[:, :, None]*self.B[None],
                -torch.sin(phase)[:, :, None]*self.B[None]], 1)
        else:
            base = torch.tanh(x@self.W.T+self.b)
            derivative = (1-base.square())[:, :, None]*self.W[None]
        z = (x[:, None, :] - self.centres[None])/self.widths[None]
        rbf = torch.exp(-.5*z.square().sum(2))
        drbf = -rbf[:, :, None]*z/self.widths[None]
        return dict(x=x, features=torch.cat([base, rbf], 1),
                    df=torch.cat([derivative, drbf], 1))


def make_shared_model(method, seed, centres, widths, uniform=None):
    if method in ['fourier_half', 'anchored', 'independent_marginal']:
        return FeatureMixedPINN(method, seed)
    if method == 'uniform_rbf_fourier':
        assert uniform is not None
        centres, widths = uniform
        base = 'fourier_half'
    else:
        base = {'geometry_rbf_fourier': 'fourier_half',
                'geometry_rbf_anchored': 'anchored',
                'geometry_rbf_independent': 'independent_marginal'}[method]
    return SharedGeometryPINN(base, seed, centres, widths)
