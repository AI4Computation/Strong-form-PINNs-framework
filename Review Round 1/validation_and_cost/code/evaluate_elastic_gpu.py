"""Float64 GPU field evaluation only; checked against independent NumPy fields."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
torch.set_num_threads(2)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64, device='cuda')


def kelvin(xy, sources, E, nu):
    r = xy[:, None] - sources[None]
    r2 = torch.sum(r * r, dim=2)
    inv = 1 / r2
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    c, k = 1 / (8 * np.pi * mu * (1 - nu)), 3 - 4 * nu
    eye = torch.eye(2, dtype=xy.dtype, device=xy.device)
    U = c * (-.5 * k * torch.log(r2)[:, :, None, None] * eye + r[:, :, :, None] * r[:, :, None, :] * inv[:, :, None, None])
    gradient = torch.empty((len(xy), len(sources), 2, 2, 2), dtype=xy.dtype, device=xy.device)
    for i in range(2):
        for j in range(2):
            for a in range(2):
                gradient[:, :, i, j, a] = c * (-k * eye[i, j] * r[:, :, a] * inv
                    + (eye[i, a] * r[:, :, j] + r[:, :, i] * eye[j, a]) * inv
                    - 2 * r[:, :, i] * r[:, :, j] * r[:, :, a] * inv ** 2)
    trace = gradient[:, :, 0, :, 0] + gradient[:, :, 1, :, 1]
    S = torch.stack([lam * trace + 2 * mu * gradient[:, :, 0, :, 0],
                     lam * trace + 2 * mu * gradient[:, :, 1, :, 1],
                     mu * (gradient[:, :, 0, :, 1] + gradient[:, :, 1, :, 0])], dim=2)
    return torch.cat([U, S], dim=2).permute(0, 2, 1, 3)


class GPUFields:
    def __init__(self, problem, sources, coefficients):
        self.E, self.nu = problem.E, problem.nu
        self.sources, self.coeff = tensor(sources), tensor(coefficients)
        self.plane = getattr(problem, 'roller_plane', None)
        if self.plane:
            from roller_sources import images
            self.n, self.t = tensor(self.plane['normal']), tensor(self.plane['tangent'])
            self.origin, self.gauge = tensor(self.plane['origin']), tensor(self.plane['gauge'])
            self.mirror = tensor(images(sources, self.plane))
            self.reflection = torch.eye(2, dtype=torch.float64, device='cuda') - 2 * torch.outer(self.n, self.n)
            at_gauge = kelvin(self.gauge[None], self.sources, self.E, self.nu)
            at_gauge += kelvin(self.gauge[None], self.mirror, self.E, self.nu) @ self.reflection
            self.shift = torch.einsum('i,imj->mj', self.t, at_gauge[0, :2]).reshape(-1)

    @torch.no_grad()
    def __call__(self, xy):
        x = tensor(xy)
        mu = self.E / (2 * (1 + self.nu))
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        source = kelvin(x, self.sources, self.E, self.nu)
        if self.plane:
            source = (source + kelvin(x, self.mirror, self.E, self.nu) @ self.reflection).reshape(len(x), 5, -1)
            source[:, :2] -= self.t[None, :, None] * self.shift[None, None]
            affine = torch.zeros((len(x), 5, 2), dtype=x.dtype, device=x.device)
            affine[:, :2, 0] = ((x - self.gauge) @ self.t)[:, None] * self.t
            affine[:, :2, 1] = ((x - self.origin) @ self.n)[:, None] * self.n
            for j, direction in enumerate([self.t, self.n]):
                stress = lam * torch.eye(2, dtype=x.dtype, device=x.device) + 2 * mu * torch.outer(direction, direction)
                affine[:, 2:, j] = torch.stack([stress[0, 0], stress[1, 1], stress[0, 1]])
        else:
            source = source.reshape(len(x), 5, -1)
            affine = torch.zeros((len(x), 5, 6), dtype=x.dtype, device=x.device)
            affine[:, 0, 0] = 1.; affine[:, 1, 1] = 1.
            affine[:, 0, 2:4] = x; affine[:, 1, 4:6] = x
            affine[:, 2, 2] = lam + 2 * mu; affine[:, 2, 5] = lam
            affine[:, 3, 2] = lam; affine[:, 3, 5] = lam + 2 * mu
            affine[:, 4, 3] = mu; affine[:, 4, 4] = mu
        return (torch.cat([affine, source], dim=2) @ self.coeff)[:, :, 0].cpu().numpy()
